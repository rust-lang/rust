// Ensure that we reject interning `const_allocate`d allocations in the final value of constants
// if they have not been made global through `const_make_global`. This covers the case where the
// pointer is even still mutable, which used to ICE.
#![feature(core_intrinsics)]
#![feature(const_heap)]
use std::intrinsics;

const BAR: *mut i32 = unsafe { intrinsics::const_allocate(4, 4) as *mut i32 };
//~^ error: encountered `const_allocate` pointer in final value that was not made global

fn main() {}
