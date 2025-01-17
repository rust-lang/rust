// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 1

#![feature(no_core, start)]

#![no_std]
#![no_core]

extern crate mini_core;

mod libc {
    #[link(name = "c")]
    extern "C" {
        pub fn printf(format: *const i8, ...) -> i32;
    }
}

fn i16_as_i8(a: i16) -> i8 {
    a as i8
}

fn call_func(func: fn(i16) -> i8, param: i16) -> i8 {
    func(param)
}

#[start]
fn main(argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let result = call_func(i16_as_i8, argc as i16) as isize;
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, result);
    }
    0
}
