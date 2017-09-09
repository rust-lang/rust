#![feature(alloc, allocator_api)]

extern crate alloc;

use alloc::heap::Heap;
use alloc::allocator::*;

// error-pattern: dangling pointer was dereferenced

fn main() {
    unsafe {
        let x = Heap.alloc(Layout::from_size_align_unchecked(1, 1)).unwrap();
        Heap.dealloc(x, Layout::from_size_align_unchecked(1, 1));
        Heap.realloc(x, Layout::from_size_align_unchecked(1, 1), Layout::from_size_align_unchecked(1, 1));
    }
}
