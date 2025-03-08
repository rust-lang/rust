// Compiler:
//
// Run-time:
//   status: signal

#![feature(no_core, start)]
#![no_std]
#![no_core]

extern crate mini_core;
use mini_core::*;

fn test_fail() -> ! {
    unsafe { intrinsics::abort() };
}

#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    test_fail();
}
