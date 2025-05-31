#![feature(core_intrinsics)]
#![feature(const_heap)]

use std::intrinsics;

const _X: () = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    intrinsics::const_deallocate(ptr, 4, 2);
    //~^ error: incorrect layout on deallocation
};
const _Y: () = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    intrinsics::const_deallocate(ptr, 2, 4);
    //~^ error: incorrect layout on deallocation
};

const _Z: () = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    intrinsics::const_deallocate(ptr, 3, 4);
    //~^ error: incorrect layout on deallocation
};

const _W: () = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    intrinsics::const_deallocate(ptr, 4, 3);
    //~^ error: invalid align
};

fn main() {}
