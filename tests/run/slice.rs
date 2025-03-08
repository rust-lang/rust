// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 5

#![feature(no_core, start)]
#![no_std]
#![no_core]

extern crate mini_core;
use mini_core::*;

static mut TWO: usize = 2;

fn index_slice(s: &[u32]) -> u32 {
    unsafe { s[TWO] }
}

#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    let array = [42, 7, 5];
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, index_slice(&array));
    }
    0
}
