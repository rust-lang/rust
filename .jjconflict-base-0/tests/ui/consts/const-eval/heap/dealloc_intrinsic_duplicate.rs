#![feature(core_intrinsics)]
#![feature(const_heap)]

use std::intrinsics;

const _X: () = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    intrinsics::const_deallocate(ptr, 4, 4);
    intrinsics::const_deallocate(ptr, 4, 4);
    //~^ ERROR: dangling
};

fn main() {}
