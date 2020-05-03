// compile-flags: -Zunleash-the-miri-inside-of-you -Zdeduplicate-diagnostics
#![feature(box_syntax)]
#![allow(const_err)]

use std::mem::ManuallyDrop;

fn main() {}

static TEST_BAD: &mut i32 = {
    &mut *(box 0)
    //~^ ERROR could not evaluate static initializer
    //~| NOTE heap allocations
};
