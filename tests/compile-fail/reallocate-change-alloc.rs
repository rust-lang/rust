#![feature(alloc, allocator_api)]

extern crate alloc;

use alloc::alloc::Global;
use std::alloc::*;

fn main() {
    unsafe {
        let x = Global.alloc(Layout::from_size_align_unchecked(1, 1));
        let _y = Global.realloc(x, Layout::from_size_align_unchecked(1, 1), 1);
        let _z = *(x as *mut u8); //~ ERROR constant evaluation error [E0080]
        //~^ NOTE dangling pointer was dereferenced
    }
}
