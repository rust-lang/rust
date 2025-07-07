#![feature(core_intrinsics)]
#![feature(const_heap)]

use std::intrinsics;

const Y: &u32 = unsafe {
    &*(intrinsics::const_make_global(std::ptr::null_mut()) as *const u32)
    //~^ error: pointer not dereferenceable
};

const Z: &u32 = unsafe {
    &*(intrinsics::const_make_global(std::ptr::dangling_mut()) as *const u32)
    //~^ error: pointer not dereferenceable
};

fn main() {}
