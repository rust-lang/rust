// Compiler:
//
// Run-time:
//   stdout: 2
//     7 8
//     10

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use mini_core::*;

fn inc_ref(num: &mut isize) -> isize {
    *num = *num + 5;
    *num + 1
}

fn inc(num: isize) -> isize {
    num + 1
}

#[no_mangle]
extern "C" fn main(mut argc: isize, _argv: *const *const u8) -> i32 {
    argc = inc(argc);
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, argc);
    }

    let b = inc_ref(&mut argc);
    unsafe {
        libc::printf(b"%ld %ld\n\0" as *const u8 as *const i8, argc, b);
    }

    argc = 10;
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, argc);
    }
    0
}
