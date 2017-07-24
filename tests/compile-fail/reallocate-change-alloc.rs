#![feature(alloc, allocator_api)]

extern crate alloc;

use alloc::heap::Heap;
use alloc::allocator::*;

fn main() {
    unsafe {
        let x = Heap.alloc(Layout::from_size_align_unchecked(1, 1)).unwrap();
        let _y = Heap.realloc(x, Layout::from_size_align_unchecked(1, 1), Layout::from_size_align_unchecked(1, 1)).unwrap();
        let _z = *x; //~ ERROR: dangling pointer was dereferenced
    }
}
