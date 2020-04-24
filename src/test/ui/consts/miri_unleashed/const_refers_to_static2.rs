// compile-flags: -Zunleash-the-miri-inside-of-you
#![allow(const_err)]

#![feature(const_raw_ptr_deref)]

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

// These tests cause immediate error when *defining* the const.

const REF_INTERIOR_MUT: &usize = { //~ ERROR undefined behavior to use this value
    static FOO: AtomicUsize = AtomicUsize::new(0);
    unsafe { &*(&FOO as *const _ as *const usize) }
    //~^ WARN skipping const checks
};

// ok some day perhaps
const READ_IMMUT: &usize = { //~ ERROR it is undefined behavior to use this value
    static FOO: usize = 0;
    &FOO
    //~^ WARN skipping const checks
};

fn main() {}
