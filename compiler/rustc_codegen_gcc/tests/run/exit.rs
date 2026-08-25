// Compiler:
//
// Run-time:
//   status: 2

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use mini_core::*;

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    unsafe {
        libc::exit(2);
    }
    0
}
