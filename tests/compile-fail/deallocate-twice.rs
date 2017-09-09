#![feature(alloc, allocator_api)]

extern crate alloc;

use alloc::heap::Heap;
use alloc::allocator::*;

// error-pattern: tried to deallocate dangling pointer

fn main() {
    unsafe {
        let x = Heap.alloc(Layout::from_size_align_unchecked(1, 1)).unwrap();
        Heap.dealloc(x, Layout::from_size_align_unchecked(1, 1));
        Heap.dealloc(x, Layout::from_size_align_unchecked(1, 1));
    }
}
