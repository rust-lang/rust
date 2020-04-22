// compile-flags: -Zunleash-the-miri-inside-of-you
// normalize-stderr-test "alloc[0-9]+" -> "allocN"

#![feature(const_raw_ptr_deref)]
#![feature(const_mut_refs)]
#![deny(const_err)] // The `allow` variant is tested by `mutable_const2`.
//~^ NOTE lint level
// Here we check that even though `MUTABLE_BEHIND_RAW` is created from a mutable
// allocation, we intern that allocation as *immutable* and reject writes to it.
// We avoid the `delay_span_bug` ICE by having compilation fail via the `deny` above.

use std::cell::UnsafeCell;

// make sure we do not just intern this as mutable
const MUTABLE_BEHIND_RAW: *mut i32 = &UnsafeCell::new(42) as *const _ as *mut _;
//~^ WARN: skipping const checks

const MUTATING_BEHIND_RAW: () = { //~ NOTE
    // Test that `MUTABLE_BEHIND_RAW` is actually immutable, by doing this at const time.
    unsafe {
        *MUTABLE_BEHIND_RAW = 99 //~ ERROR any use of this value will cause an error
        //~^ NOTE: which is read-only
        // FIXME would be good to match more of the error message here, but looks like we
        // normalize *after* checking the annoations here.
    }
};

fn main() {}
