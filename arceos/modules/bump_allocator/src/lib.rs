#![no_std]

use allocator::{AllocError, AllocResult, BaseAllocator, ByteAllocator, PageAllocator};
use core::{alloc::Layout, ptr::NonNull};

/// Early memory allocator
/// Use it before formal bytes-allocator and pages-allocator can work!
/// This is a double-end memory range:
/// - Alloc bytes forward
/// - Alloc pages backward
///
/// [ bytes-used | avail-area | pages-used ]
/// |            | -->    <-- |            |
/// start       b_pos        p_pos       end
///
/// For bytes area, 'count' records number of allocations.
/// When it goes down to ZERO, free bytes-used area.
/// For pages area, it will never be freed!
///
pub struct EarlyAllocator<const SIZE: usize> {
    start: usize,
    end: usize,
    b_pos: usize,
    p_pos: usize,
    count: usize,
}

impl<const SIZE: usize> EarlyAllocator<SIZE> {
    pub const fn new() -> Self {
        Self {
            start: 0,
            end: 0,
            b_pos: 0,
            p_pos: 0,
            count: 0,
        }
    }
}

impl<const SIZE: usize> BaseAllocator for EarlyAllocator<SIZE> {
    fn init(&mut self, start: usize, size: usize) {
        assert!(SIZE.is_power_of_two());
        let end = start.checked_add(size).expect("invalid allocator region");
        self.start = start;
        self.end = end;
        self.b_pos = start;
        self.p_pos = end;
        self.count = 0;
    }

    fn add_memory(&mut self, start: usize, size: usize) -> AllocResult {
        let end = start.checked_add(size).ok_or(AllocError::InvalidParam)?;
        if start < self.end && end > self.start {
            return Err(AllocError::MemoryOverlap);
        }
        // This allocator can only extend its single contiguous free span.
        if start == self.end && self.p_pos == self.end {
            self.end = end;
            self.p_pos = end;
        } else if end == self.start && self.count == 0 {
            self.start = start;
            self.b_pos = start;
        } else {
            return Err(AllocError::InvalidParam);
        }
        Ok(())
    }
}

impl<const SIZE: usize> ByteAllocator for EarlyAllocator<SIZE> {
    fn alloc(&mut self, layout: Layout) -> AllocResult<NonNull<u8>> {
        let start = align_up(self.b_pos.max(1), layout.align()).ok_or(AllocError::NoMemory)?;
        let end = start
            .checked_add(layout.size().max(1))
            .filter(|&end| end <= self.p_pos)
            .ok_or(AllocError::NoMemory)?;
        let ptr = NonNull::new(start as *mut u8).ok_or(AllocError::NoMemory)?;
        self.b_pos = end;
        self.count += 1;
        Ok(ptr)
    }

    fn dealloc(&mut self, _pos: NonNull<u8>, _layout: Layout) {
        self.count -= 1;
        if self.count == 0 {
            self.b_pos = self.start;
        }
    }

    fn total_bytes(&self) -> usize {
        self.end - self.start
    }

    fn used_bytes(&self) -> usize {
        self.b_pos - self.start
    }

    fn available_bytes(&self) -> usize {
        self.p_pos - self.b_pos
    }
}

impl<const SIZE: usize> PageAllocator for EarlyAllocator<SIZE> {
    const PAGE_SIZE: usize = SIZE;

    fn alloc_pages(&mut self, num_pages: usize, align_pow2: usize) -> AllocResult<usize> {
        if num_pages == 0 || !align_pow2.is_power_of_two() {
            return Err(AllocError::InvalidParam);
        }
        let size = num_pages.checked_mul(SIZE).ok_or(AllocError::NoMemory)?;
        let align = align_pow2.max(SIZE);
        let start = self.p_pos.checked_sub(size).ok_or(AllocError::NoMemory)? & !(align - 1);
        if start < self.b_pos {
            return Err(AllocError::NoMemory);
        }
        self.p_pos = start;
        Ok(start)
    }

    fn dealloc_pages(&mut self, _pos: usize, _num_pages: usize) {}

    fn total_pages(&self) -> usize {
        whole_pages::<SIZE>(self.start, self.end)
    }

    fn used_pages(&self) -> usize {
        (self.end / SIZE) - (self.p_pos / SIZE)
    }

    fn available_pages(&self) -> usize {
        whole_pages::<SIZE>(self.b_pos, self.p_pos)
    }
}

fn align_up(addr: usize, align: usize) -> Option<usize> {
    addr.checked_add(align - 1).map(|addr| addr & !(align - 1))
}

fn whole_pages<const SIZE: usize>(start: usize, end: usize) -> usize {
    align_up(start, SIZE).map_or(0, |start| (end / SIZE).saturating_sub(start / SIZE))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_from_both_ends_and_recycles_bytes() {
        let mut allocator = EarlyAllocator::<0x1000>::new();
        allocator.init(0x1003, 0x4ffd);
        let layout = Layout::from_size_align(0x20, 0x40).unwrap();
        let first = allocator.alloc(layout).unwrap();
        let second = allocator.alloc(layout).unwrap();
        assert_eq!(first.as_ptr() as usize, 0x1040);
        assert_eq!(second.as_ptr() as usize, 0x1080);
        assert_eq!(allocator.alloc_pages(2, 0x2000).unwrap(), 0x4000);
        assert!(matches!(
            allocator.alloc_pages(3, 0x1000),
            Err(AllocError::NoMemory)
        ));
        assert_eq!(allocator.available_pages(), 2);
        allocator.dealloc(first, layout);
        assert_ne!(allocator.used_bytes(), 0);
        allocator.dealloc(second, layout);
        assert_eq!(allocator.used_bytes(), 0);
        assert_eq!(allocator.alloc(layout).unwrap(), first);
    }
}
