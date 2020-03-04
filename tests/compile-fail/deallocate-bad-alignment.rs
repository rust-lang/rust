#![feature(allocator_api)]

extern crate alloc;

use alloc::alloc::Global;
use std::alloc::{AllocRef, Layout};

// error-pattern: incorrect alloc info: expected size 1 and align 2, got size 1 and align 1

fn main() {
    unsafe {
        let x = Global.alloc(Layout::from_size_align_unchecked(1, 1)).unwrap().0;
        Global.dealloc(x, Layout::from_size_align_unchecked(1, 2));
    }
}
