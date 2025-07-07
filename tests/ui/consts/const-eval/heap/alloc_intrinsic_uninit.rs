//@ stderr-per-bitwidth
// compile-test
#![feature(core_intrinsics)]
#![feature(const_heap)]
use std::intrinsics;

const BAR: &i32 = unsafe { //~ ERROR: uninitialized memory
    &*(intrinsics::const_make_global(intrinsics::const_allocate(4, 4)) as *mut i32)
};
fn main() {}
