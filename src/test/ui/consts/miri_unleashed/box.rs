// compile-flags: -Zunleash-the-miri-inside-of-you
#![feature(box_syntax)]
#![allow(const_err)]

use std::mem::ManuallyDrop;

fn main() {}

static TEST_BAD: &mut i32 = {
    &mut *(box 0)
    //~^ ERROR could not evaluate static initializer
    //~| NOTE calling non-const function `alloc::alloc::exchange_malloc`
};
