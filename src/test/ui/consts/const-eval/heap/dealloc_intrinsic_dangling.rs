#![feature(core_intrinsics)]
#![feature(const_heap)]
#![feature(const_mut_refs)]

use std::intrinsics;

const _X: &'static u8 = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    intrinsics::const_deallocate(ptr, 4, 4);
    &*ptr
    //~^ error: evaluation of constant value failed
};

const _Y: u8 = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    let reference = &*ptr;
    intrinsics::const_deallocate(ptr, 4, 4);
    *reference
    //~^ error: evaluation of constant value failed
};

fn main() {}
