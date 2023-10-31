// compile-flags: -Zunleash-the-miri-inside-of-you

use std::cell::UnsafeCell;

const MUTABLE_BEHIND_RAW: *mut i32 = &UnsafeCell::new(42) as *const _ as *mut _;
//~^ ERROR: unsupported untyped pointer in constant

fn main() {}
