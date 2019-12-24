// compile-flags: -Zunleash-the-miri-inside-of-you
// ignore-x86 FIXME: missing sysroot spans (#53081)
#![deny(const_err)]

use std::mem::ManuallyDrop;

fn main() {}

static TEST_OK: () = {
    let v: Vec<i32> = Vec::new();
    let _v = ManuallyDrop::new(v);
};

// Make sure we catch executing bad drop functions.
// The actual error is located in `real_drop_in_place` so we can't capture it with the
// error annotations here.
static TEST_BAD: () = {
    let _v: Vec<i32> = Vec::new();
    //~^ WARN skipping const check
};
