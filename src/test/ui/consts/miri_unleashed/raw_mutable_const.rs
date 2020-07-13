// compile-flags: -Zunleash-the-miri-inside-of-you

#![allow(const_err)]

use std::cell::UnsafeCell;

const MUTABLE_BEHIND_RAW: *mut i32 = &UnsafeCell::new(42) as *const _ as *mut _;
//~^ ERROR: untyped pointers are not allowed in constant

fn main() {}
