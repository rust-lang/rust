#![feature(core_intrinsics)]
#![feature(const_heap)]
#![feature(const_mut_refs)]
use std::intrinsics;

const BAR: *mut i32 = unsafe { intrinsics::const_allocate(4, 4) as *mut i32};
//~^ error: untyped pointers are not allowed in constant

fn main() {}
