#![feature(alloc, allocator_api)]

extern crate alloc;

use alloc::alloc::Global;
use std::alloc::*;

// error-pattern: tried to deallocate dangling pointer

fn main() {
    unsafe {
        let x = Global.alloc(Layout::from_size_align_unchecked(1, 1)).unwrap();
        Global.dealloc(x, Layout::from_size_align_unchecked(1, 1));
        Global.dealloc(x, Layout::from_size_align_unchecked(1, 1));
    }
}
