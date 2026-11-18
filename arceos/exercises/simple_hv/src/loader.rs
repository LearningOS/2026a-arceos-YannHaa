use crate::VM_ENTRY;
use axerrno::{AxError, AxResult};
use axhal::mem::PAGE_SIZE_4K;
use axhal::paging::MappingFlags;
use axmm::AddrSpace;
use std::fs::File;
use std::io::Read;

pub fn load_vm_image(fname: &str, uspace: &mut AddrSpace) -> AxResult<()> {
    let mut file = File::open(fname)?;
    let len = file.metadata()?.size();
    if len == 0 || len > PAGE_SIZE_4K as u64 {
        return Err(AxError::InvalidInput);
    }
    let mut image = [0; PAGE_SIZE_4K];
    let image = &mut image[..len as usize];
    file.read_exact(image)?;

    uspace.map_alloc(
        VM_ENTRY.into(),
        PAGE_SIZE_4K,
        MappingFlags::READ | MappingFlags::EXECUTE | MappingFlags::USER,
        true,
    )?;
    uspace.write(VM_ENTRY.into(), image)?;
    // Synchronize the instruction stream with the image just copied to RAM.
    unsafe { core::arch::asm!("fence.i") };
    Ok(())
}
