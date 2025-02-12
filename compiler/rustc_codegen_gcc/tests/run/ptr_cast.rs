// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 1

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

static mut ONE: usize = 1;

fn make_array() -> [u8; 3] {
    [42, 10, 5]
}

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    unsafe {
        let ptr = ONE as *mut usize;
        let value = ptr as usize;
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, value);
    }
    0
}
