// Compiler:
//
// Run-time:
//   status: signal

#![feature(no_core, start)]
#![no_std]
#![no_core]

extern crate mini_core;
use mini_core::*;

fn fail() -> i32 {
    unsafe { intrinsics::abort() };
    0
}

#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    fail();
    0
}
