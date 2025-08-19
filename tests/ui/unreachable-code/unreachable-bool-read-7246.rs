#![deny(unreachable_code)]
#![allow(dead_code)]

use std::ptr;
pub unsafe fn g() {
    return;
    if *ptr::null() {}; //~ ERROR unreachable
    //~| WARNING dereferencing a null pointer
}

pub fn main() {}

// https://github.com/rust-lang/rust/issues/7246
