// compile-flags: -Zunleash-the-miri-inside-of-you
// failure-status: 101
// rustc-env:RUST_BACKTRACE=0
// normalize-stderr-test "note: rustc 1.* running on .*" -> "note: rustc VERSION running on TARGET"
// normalize-stderr-test "note: compiler flags: .*" -> "note: compiler flags: FLAGS"
// normalize-stderr-test "interpret/intern.rs:[0-9]*:[0-9]*" -> "interpret/intern.rs:LL:CC"

#![feature(const_raw_ptr_deref)]
#![feature(const_mut_refs)]
#![deny(const_err)]

use std::cell::UnsafeCell;

// make sure we do not just intern this as mutable
const MUTABLE_BEHIND_RAW: *mut i32 = &UnsafeCell::new(42) as *const _ as *mut _;
//~^ WARN: skipping const checks
//~| ERROR: mutable allocation in constant

fn main() {}
