// Ensure that once an allocation is "made global", we can no longer mutate it.
#![feature(core_intrinsics)]
#![feature(const_heap)]
use std::intrinsics;

const A: &u8 = unsafe {
    let ptr = intrinsics::const_allocate(1, 1);
    *ptr = 1;
    let ptr: *const u8 = intrinsics::const_make_global(ptr);
    *(ptr as *mut u8) = 2;
    //~^ error: writing to ALLOC0 which is read-only
    &*ptr
};

fn main() {}
