//@ stderr-per-bitwidth
// compile-test
#![feature(core_intrinsics)]
#![feature(const_heap)]
use std::intrinsics;

const BAR: &i32 = unsafe { &*(intrinsics::const_allocate(4, 4) as *mut i32) };
//~^ error: it is undefined behavior to use this value
fn main() {}
