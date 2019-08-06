// ignore-tidy-linelength
// This test checks that calling `mem::{uninitialized,zeroed}` with certain types results
// in a lint.

#![feature(never_type)]
#![allow(deprecated)]
#![deny(invalid_value)]

use std::mem;

fn main() {
    unsafe {
        let _val: ! = mem::zeroed(); //~ ERROR: does not permit zero-initialization
        let _val: ! = mem::uninitialized(); //~ ERROR: does not permit being left uninitialized

        let _val: &'static i32 = mem::zeroed(); //~ ERROR: does not permit zero-initialization
        let _val: &'static i32 = mem::uninitialized(); //~ ERROR: does not permit being left uninitialized

        let _val: fn() = mem::zeroed(); //~ ERROR: does not permit zero-initialization
        let _val: fn() = mem::uninitialized(); //~ ERROR: does not permit being left uninitialized

        // Some types that should work just fine.
        let _val: Option<&'static i32> = mem::zeroed();
        let _val: Option<fn()> = mem::zeroed();
        let _val: bool = mem::zeroed();
        let _val: i32 = mem::zeroed();
    }
}
