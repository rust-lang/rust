// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 42
//     7
//     5
//     10

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use mini_core::*;

static mut ONE: usize = 1;

fn make_array() -> [u8; 3] {
    [42, 10, 5]
}

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let array = [42, 7, 5];
    let array2 = make_array();
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, array[ONE - 1]);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, array[ONE]);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, array[ONE + 1]);

        libc::printf(b"%d\n\0" as *const u8 as *const i8, array2[argc as usize] as u32);
    }
    0
}
