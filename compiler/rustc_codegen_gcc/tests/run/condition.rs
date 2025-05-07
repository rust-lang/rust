// Compiler:
//
// Run-time:
//   status: 0
//   stdout: true
//     1

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use mini_core::*;

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    unsafe {
        if argc == 1 {
            libc::printf(b"true\n\0" as *const u8 as *const i8);
        }

        let string = match argc {
            1 => b"1\n\0",
            2 => b"2\n\0",
            3 => b"3\n\0",
            4 => b"4\n\0",
            5 => b"5\n\0",
            _ => b"_\n\0",
        };
        libc::printf(string as *const u8 as *const i8);
    }
    0
}
