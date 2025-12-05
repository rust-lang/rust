// Ensure that we can't call `const_make_global` twice.

#![feature(core_intrinsics)]
#![feature(const_heap)]

use std::intrinsics;

const Y: &i32 = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    let i = ptr as *mut i32;
    *i = 20;
    intrinsics::const_make_global(ptr);
    intrinsics::const_make_global(ptr);
    //~^ error: attempting to call `const_make_global` twice on the same allocation ALLOC0
    &*i
};

fn main() {}
