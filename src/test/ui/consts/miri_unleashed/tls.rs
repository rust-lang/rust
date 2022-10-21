// compile-flags: -Zunleash-the-miri-inside-of-you
#![feature(thread_local)]

use std::thread;

#[thread_local]
static A: u8 = 0;

// Make sure we catch accessing thread-local storage.
static TEST_BAD: () = {
    unsafe { let _val = A; }
    //~^ ERROR could not evaluate static initializer
    //~| NOTE cannot access thread local static
};

// Make sure we catch taking a reference to thread-local storage.
static TEST_BAD_REF: () = {
    unsafe { let _val = &A; }
    //~^ ERROR could not evaluate static initializer
    //~| NOTE cannot access thread local static
};

fn main() {}
