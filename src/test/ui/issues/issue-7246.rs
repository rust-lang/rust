#![deny(unreachable_code)]
#![allow(dead_code)]

use std::ptr;
pub unsafe fn g() {
    return;
    if *ptr::null() {}; //~ ERROR unreachable
}

pub fn main() {}
