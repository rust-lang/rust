#![feature(alloc, allocator_api)]

extern crate alloc;

use alloc::heap::Heap;
use alloc::allocator::*;

// error-pattern: incorrect alloc info: expected size 2 and align 1, got size 1 and align 1

fn main() {
    unsafe {
        let x = Heap.alloc(Layout::from_size_align_unchecked(1, 1)).unwrap();
        let _y = Heap.realloc(x, Layout::from_size_align_unchecked(2, 1), Layout::from_size_align_unchecked(1, 1)).unwrap();
    }
}
