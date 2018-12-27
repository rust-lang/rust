#![feature(const_transmute,const_let)]
#![allow(const_err)] // make sure we cannot allow away the errors tested here

use std::mem;

const BAD_UPVAR: &FnOnce() = &{ //~ ERROR it is undefined behavior to use this value
    let bad_ref: &'static u16 = unsafe { mem::transmute(0usize) };
    let another_var = 13;
    move || { let _ = bad_ref; let _ = another_var; }
};

fn main() {}
