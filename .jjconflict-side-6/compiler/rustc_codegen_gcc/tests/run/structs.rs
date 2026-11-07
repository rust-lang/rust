// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 1
//     2

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

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

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let test = Test { field: one() };
    let two = Two { two: 2 };
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, test.field);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, two.two);
    }
    0
}
