#![feature(alloc, allocator_api)]

extern crate alloc;

use alloc::heap::Heap;
use alloc::allocator::*;

// error-pattern: tried to deallocate or reallocate using incorrect alignment or size

use alloc::heap::*;
fn main() {
    unsafe {
        let x = Heap.alloc(Layout::from_size_align_unchecked(1, 1)).unwrap();
        Heap.dealloc(x, Layout::from_size_align_unchecked(2, 1));
    }
}
