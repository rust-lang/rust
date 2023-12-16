// compile-flags: -Zunleash-the-miri-inside-of-you
// Similar to `raw-ptr-const.rs`, but with *mutable* data. *Must* be rejected.

use std::cell::UnsafeCell;

const MUTABLE_BEHIND_RAW: *mut i32 = &UnsafeCell::new(42) as *const _ as *mut _;
//~^ ERROR: mutable pointer in final value

fn main() {}
