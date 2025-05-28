//@ stderr-per-bitwidth
// compile-test
#![feature(core_intrinsics)]
#![feature(const_heap)]
use std::intrinsics;

const BAR: &i32 = unsafe { &*(intrinsics::const_allocate(4, 4) as *mut i32) };
//~^ ERROR: uninitialized memory
fn main() {}
