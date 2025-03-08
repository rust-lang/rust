// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 1
//     2

#![feature(no_core, start)]
#![no_std]
#![no_core]

extern crate mini_core;
use mini_core::*;

struct Test {
    field: isize,
}

struct Two {
    two: isize,
}

fn one() -> isize {
    1
}

#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    let test = Test { field: one() };
    let two = Two { two: 2 };
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, test.field);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, two.two);
    }
    0
}
