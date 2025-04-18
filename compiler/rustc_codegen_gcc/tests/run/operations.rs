// Compiler:
//
// Run-time:
//   stdout: 41
//     39
//     10

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use mini_core::*;

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, 40 + argc);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, 40 - argc);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, 10 * argc);
    }
    0
}
