// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 42
//     7
//     5
//     10

#![feature(no_core, start)]
#![no_std]
#![no_core]

extern crate mini_core;
use mini_core::*;

static mut ONE: usize = 1;

fn make_array() -> [u8; 3] {
    [42, 10, 5]
}

#[start]
fn main(argc: isize, _argv: *const *const u8) -> isize {
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
