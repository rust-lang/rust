// compile-flags: -Zunleash-the-miri-inside-of-you

#![feature(const_raw_ptr_deref)]
#![deny(const_err)]

use std::cell::UnsafeCell;

// make sure we do not just intern this as mutable
const MUTABLE_BEHIND_RAW: *mut i32 = &UnsafeCell::new(42) as *const _ as *mut _;

const MUTATING_BEHIND_RAW: () = {
    // Test that `MUTABLE_BEHIND_RAW` is actually immutable, by doing this at const time.
    unsafe {
        *MUTABLE_BEHIND_RAW = 99 //~ WARN skipping const checks
        //~^ ERROR any use of this value will cause an error
        //~^^ tried to modify constant memory
    }
};

fn main() {}
