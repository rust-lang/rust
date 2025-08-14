//@ stderr-per-bitwidth
#![allow(invalid_value)] // make sure we cannot allow away the errors tested here

use std::mem;

const BAD_UPVAR: &dyn FnOnce() = &{ //~ ERROR null reference
    let bad_ref: &'static u16 = unsafe { mem::transmute(0usize) };
    let another_var = 13;
    move || { let _ = bad_ref; let _ = another_var; }
};

fn main() {}
