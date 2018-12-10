#![feature(alloc, allocator_api)]

extern crate alloc;

use alloc::alloc::Global;
use std::alloc::*;

// error-pattern: incorrect alloc info: expected size 2 and align 1, got size 1 and align 1

fn main() {
    unsafe {
        let x = Global.alloc(Layout::from_size_align_unchecked(1, 1)).unwrap();
        Global.realloc(x, Layout::from_size_align_unchecked(2, 1), 1).unwrap();
    }
}
