//@ check-pass
//@ add-minicore

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]
#![allow(dead_code)]

extern crate minicore;
use minicore::*;

pub unsafe fn should_lint(a: usize) -> *const u8 {
    unsafe { mem::transmute::<usize, *const u8>(a) }
    //~^ WARN transmuting an integer to a pointer
}
