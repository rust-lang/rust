//! Check what happens when the error occurs inside a std function that we can't print the span of.
//@ ignore-backends: gcc
//@ compile-flags: -Z ui-testing=no

use std::{
    mem::{self, MaybeUninit},
    ptr,
};

const X: () = {
    let mut x1 = 1;
    let mut x2 = 2;

    // Swap them, bytewise.
    unsafe {
        ptr::swap_nonoverlapping( //~ ERROR beyond the end of the allocation
            &mut x1 as *mut _ as *mut MaybeUninit<u8>,
            &mut x2 as *mut _ as *mut MaybeUninit<u8>,
            10,
        );
    }
};

fn main() {
    X
}
