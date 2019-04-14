#![feature(allocator_api)]

extern crate alloc;

use alloc::alloc::Global;
use std::alloc::*;

fn main() {
    unsafe {
        let x = Global.alloc(Layout::from_size_align_unchecked(1, 1)).unwrap();
        Global.realloc(x, Layout::from_size_align_unchecked(1, 1), 1).unwrap();
        let _z = *(x.as_ptr() as *mut u8); //~ ERROR dangling pointer was dereferenced
    }
}
