#![feature(core_intrinsics)]
#![feature(const_heap)]

use std::intrinsics;

const _X: () = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    intrinsics::const_deallocate(ptr, 4, 2);
    //~^ error: evaluation of constant value failed
};
const _Y: () = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    intrinsics::const_deallocate(ptr, 2, 4);
    //~^ error: evaluation of constant value failed
};

const _Z: () = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    intrinsics::const_deallocate(ptr, 3, 4);
    //~^ error: evaluation of constant value failed
};

const _W: () = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    intrinsics::const_deallocate(ptr, 4, 3);
    //~^ error: evaluation of constant value failed
};

fn main() {}
