#![cfg_attr(feature = "axstd", no_std)]
#![cfg_attr(feature = "axstd", no_main)]
#![feature(asm_const)]
#![feature(riscv_ext_intrinsics)]

extern crate alloc;
#[cfg(feature = "axstd")]
extern crate axstd as std;
#[macro_use]
extern crate axlog;

mod csrs;
mod loader;
mod regs;
mod sbi;
mod task;
mod vcpu;

use alloc::boxed::Box;
use axhal::arch::{local_irq_restore, local_irq_save_and_disable};
use axhal::mem::{phys_to_virt, virt_to_phys, PhysAddr, PAGE_SIZE_4K};
use axmm::AddrSpace;
use csrs::defs::hstatus;
use csrs::{RiscvCsrTrait, CSR};
use loader::load_vm_image;
use regs::GprIndex::{self, A0, A1, A7};
use riscv::register::{htval, scause, sstatus, stval};
use sbi::SbiMessage;
use tock_registers::LocalRegisterCopy;
use vcpu::{_run_guest, VmCpuRegisters};

const VM_ENTRY: usize = 0x8020_0000;
const GUEST_LOAD_ADDR: usize = 64;
const GUEST_LOAD_VALUE: usize = 0x6688;
const GUEST_HART_ID: usize = 0x1234;
const INSN_SIZE: usize = 4;
const RD_MASK: u32 = 0x1f << 7;
// Instruction encodings with the rd field cleared.
const READ_MHARTID: u32 = 0xf140_2073; // csrrs rd, mhartid, zero
const LOAD_GUEST_VALUE: u32 = 0x0400_3003; // ld rd, 64(zero)

// Sv39x4 has a 16 KiB root; Sv39 lower-level tables have the same format.
#[repr(align(16384))]
struct GuestStageRoot([u8; 4 * PAGE_SIZE_4K]);

#[cfg_attr(feature = "axstd", no_mangle)]
fn main() {
    ax_println!("Hypervisor ...");

    // Only the guest image is mapped; GPA 64 is handled as an emulated read.
    let mut uspace = AddrSpace::new_empty(0.into(), VM_ENTRY + PAGE_SIZE_4K).unwrap();

    // Load vm binary file into address space.
    if let Err(e) = load_vm_image("/sbin/skernel2", &mut uspace) {
        panic!("Cannot load app! {:?}", e);
    }

    // Setup context to prepare to enter guest mode.
    let mut ctx = VmCpuRegisters::default();
    prepare_guest_context(&mut ctx);

    // Setup pagetable for 2nd address mapping.
    let mut root = Box::new(GuestStageRoot([0; 4 * PAGE_SIZE_4K]));
    let root_ptr = root.0.as_mut_ptr();
    // The guest uses only the first 512 root entries. Keep `uspace` alive for
    // the lower-level tables these entries point to.
    unsafe {
        core::ptr::copy_nonoverlapping(
            phys_to_virt(uspace.page_table_root()).as_ptr(),
            root_ptr,
            PAGE_SIZE_4K,
        );
    }
    prepare_vm_pgtable(virt_to_phys((root_ptr as usize).into()));

    while !run_guest(&mut ctx, &uspace) {}

    // Disable guest translation before dropping its page tables.
    unsafe {
        core::arch::asm!("csrw hgatp, zero");
        core::arch::riscv64::hfence_gvma_all();
    }
}

fn prepare_vm_pgtable(ept_root: PhysAddr) {
    let hgatp = 8usize << 60 | usize::from(ept_root) >> 12;
    unsafe {
        core::arch::asm!(
            "csrw hgatp, {hgatp}",
            hgatp = in(reg) hgatp,
        );
        core::arch::riscv64::hfence_gvma_all();
    }
}

fn run_guest(ctx: &mut VmCpuRegisters, uspace: &AddrSpace) -> bool {
    // HS interrupts are otherwise enabled while executing in VS mode.
    let irq_flags = local_irq_save_and_disable();
    let sie = CSR.sie.atomic_replace(0);
    unsafe {
        _run_guest(ctx);
    }
    let done = vmexit_handler(ctx, uspace);
    CSR.sie.write_value(sie);
    local_irq_restore(irq_flags);
    done
}

fn emulate_read(ctx: &mut VmCpuRegisters, uspace: &AddrSpace, opcode: u32, value: usize) {
    let pc = ctx.guest_regs.sepc;
    let mut bytes = [0; INSN_SIZE];
    uspace
        .read(pc.into(), &mut bytes)
        .expect("Cannot read guest instruction");
    let instruction = u32::from_le_bytes(bytes);
    assert_eq!(
        instruction & !RD_MASK,
        opcode,
        "Unexpected instruction {instruction:#x} at {pc:#x}"
    );
    let rd = GprIndex::from_raw((instruction & RD_MASK) >> 7).unwrap();
    ctx.guest_regs.gprs.set_reg(rd, value);
    ctx.guest_regs.sepc += INSN_SIZE;
}

fn vmexit_handler(ctx: &mut VmCpuRegisters, uspace: &AddrSpace) -> bool {
    use scause::{Exception, Trap};

    let scause = scause::read();
    match scause.cause() {
        Trap::Exception(Exception::VirtualSupervisorEnvCall) => {
            let sbi_msg =
                SbiMessage::from_regs(ctx.guest_regs.gprs.a_regs()).expect("Bad guest SBI message");
            ax_println!("VmExit Reason: VSuperEcall: {:?}", sbi_msg);
            match sbi_msg {
                SbiMessage::Reset(_)
                    if ctx.guest_regs.gprs.reg(A7) == sbi_spec::legacy::LEGACY_SHUTDOWN =>
                {
                    let a0 = ctx.guest_regs.gprs.reg(A0);
                    let a1 = ctx.guest_regs.gprs.reg(A1);
                    ax_println!("a0 = {:#x}, a1 = {:#x}", a0, a1);
                    assert_eq!(a0, GUEST_LOAD_VALUE);
                    assert_eq!(a1, GUEST_HART_ID);
                    ax_println!("Shutdown vm normally!");
                    true
                }
                _ => panic!("Unsupported guest SBI message: {sbi_msg:?}"),
            }
        }
        Trap::Exception(Exception::IllegalInstruction) => {
            emulate_read(ctx, uspace, READ_MHARTID, GUEST_HART_ID);
            false
        }
        Trap::Exception(Exception::LoadGuestPageFault) => {
            let vaddr = stval::read();
            let gpa = htval::read() << 2 | vaddr & 0x3;
            assert_eq!(vaddr, GUEST_LOAD_ADDR, "Unexpected guest load address");
            assert_eq!(
                gpa, GUEST_LOAD_ADDR,
                "Unexpected guest-physical load address"
            );
            emulate_read(ctx, uspace, LOAD_GUEST_VALUE, GUEST_LOAD_VALUE);
            false
        }
        _ => {
            panic!(
                "Unhandled trap: {:?}, sepc: {:#x}, stval: {:#x}",
                scause.cause(),
                ctx.guest_regs.sepc,
                stval::read()
            );
        }
    }
}

fn prepare_guest_context(ctx: &mut VmCpuRegisters) {
    // Set hstatus
    let mut hstatus =
        LocalRegisterCopy::<usize, hstatus::Register>::new(riscv::register::hstatus::read().bits());
    // Set Guest bit in order to return to guest mode.
    hstatus.modify(hstatus::spv::Guest);
    // Set SPVP bit in order to accessing VS-mode memory from HS-mode.
    hstatus.modify(hstatus::spvp::Supervisor);
    ctx.guest_regs.hstatus = hstatus.get();

    // Set sstatus in guest mode.
    let mut sstatus = sstatus::read();
    sstatus.set_spp(sstatus::SPP::Supervisor);
    // Keep HS interrupts off during the assembly transition into the guest.
    ctx.guest_regs.sstatus = sstatus.bits() & !(1 << 1);
    // Return to entry to start vm.
    ctx.guest_regs.sepc = VM_ENTRY;
}
