//@ stderr-per-bitwidth
// compile-test
#![feature(core_intrinsics)]
#![feature(const_heap)]
use std::intrinsics;

const BAR: &i32 = unsafe { //~ ERROR: uninitialized memory
    // Make the pointer immutable to avoid errors related to mutable pointers in constants.
    &*(intrinsics::const_make_global(intrinsics::const_allocate(4, 4)) as *const i32)
};
fn main() {}
