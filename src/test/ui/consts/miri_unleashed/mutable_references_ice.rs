// compile-flags: -Zunleash-the-miri-inside-of-you
// failure-status: 101
// rustc-env:RUST_BACKTRACE=0
// normalize-stderr-test "note: rustc 1.* running on .*" -> "note: rustc VERSION running on TARGET"
// normalize-stderr-test "note: compiler flags: .*" -> "note: compiler flags: FLAGS"
// normalize-stderr-test "interpret/intern.rs:[0-9]*:[0-9]*" -> "interpret/intern.rs:LL:CC"

#![allow(const_err)]

use std::cell::UnsafeCell;

// this test ICEs to ensure that our mutability story is sound

struct Meh {
    x: &'static UnsafeCell<i32>,
}

unsafe impl Sync for Meh {}

// the following will never be ok!
const MUH: Meh = Meh {
    x: &UnsafeCell::new(42),
};

fn main() {
    unsafe {
        *MUH.x.get() = 99; //~ WARN skipping const checks
    }
}
