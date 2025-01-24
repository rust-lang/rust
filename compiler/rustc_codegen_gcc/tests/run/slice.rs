// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 5

#![feature(no_core)]

#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;

mod libc {
    #[link(name = "c")]
    extern "C" {
        pub fn printf(format: *const i8, ...) -> i32;
    }
}

static mut TWO: usize = 2;

fn index_slice(s: &[u32]) -> u32 {
    unsafe {
        s[TWO]
    }
}

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let array = [42, 7, 5];
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, index_slice(&array));
    }
    0
}
