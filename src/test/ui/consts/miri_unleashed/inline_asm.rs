// compile-flags: -Zunleash-the-miri-inside-of-you
// only-x86_64
#![allow(const_err)]

use std::arch::asm;

fn main() {}

// Make sure we catch executing inline assembly.
static TEST_BAD: () = {
    unsafe { asm!("nop"); }
    //~^ ERROR could not evaluate static initializer
    //~| NOTE inline assembly is not supported
};
