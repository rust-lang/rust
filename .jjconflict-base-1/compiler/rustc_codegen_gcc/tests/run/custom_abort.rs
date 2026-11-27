// Compiler:
//
// Run-time:
//   status: 42

// Check that a program can define its own `abort`.

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use mini_core::*;

#[no_mangle]
extern "C" fn abort() {
    unsafe {
        libc::exit(42);
    }
}

#[no_mangle]
extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    abort();
    0
}
