// compile-flags: -Zunleash-the-miri-inside-of-you
// stderr-per-bitwidth
#![allow(const_err)]

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

// These only fail during validation (they do not use but just create a reference to a static),
// so they cause an immediate error when *defining* the const.

const REF_INTERIOR_MUT: &usize = { //~ ERROR undefined behavior to use this value
//~| NOTE encountered a reference pointing to a static variable
//~| NOTE undefined behavior
//~| NOTE the raw bytes of the constant
    static FOO: AtomicUsize = AtomicUsize::new(0);
    unsafe { &*(&FOO as *const _ as *const usize) }
};

// ok some day perhaps
const READ_IMMUT: &usize = { //~ ERROR it is undefined behavior to use this value
//~| NOTE encountered a reference pointing to a static variable
//~| NOTE undefined behavior
//~| NOTE the raw bytes of the constant
    static FOO: usize = 0;
    &FOO
};

fn main() {}
