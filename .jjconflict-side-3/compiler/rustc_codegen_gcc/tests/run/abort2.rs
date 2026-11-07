// Compiler:
//
// Run-time:
//   status: signal

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use mini_core::*;

fn fail() -> i32 {
    unsafe { intrinsics::abort() };
    0
}

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    fail();
    0
}
