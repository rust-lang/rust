#![feature(allocator_api)]

extern crate alloc;

use alloc::alloc::Global;
use std::alloc::{AllocRef, Layout};

// error-pattern: dangling pointer was dereferenced

fn main() {
    unsafe {
        let x = Global.alloc(Layout::from_size_align_unchecked(1, 1)).unwrap().0;
        Global.dealloc(x, Layout::from_size_align_unchecked(1, 1));
        Global.realloc(x, Layout::from_size_align_unchecked(1, 1), 1).unwrap();
    }
}
