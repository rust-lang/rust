// Compiler:
//
// Run-time:
//   status: 0
//   stdout: Arg: 1
//     Argument: 1
//     String arg: 1
//     Int argument: 2
//     Both args: 11

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

#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    let string = "Arg: %d\n\0";
    let mut closure = || {
        unsafe {
            libc::printf(string as *const str as *const i8, argc);
        }
    };
    closure();

    let mut closure = || {
        unsafe {
            libc::printf("Argument: %d\n\0" as *const str as *const i8, argc);
        }
    };
    closure();

    let mut closure = |string| {
        unsafe {
            libc::printf(string as *const str as *const i8, argc);
        }
    };
    closure("String arg: %d\n\0");

    let mut closure = |arg: isize| {
        unsafe {
            libc::printf("Int argument: %d\n\0" as *const str as *const i8, arg);
        }
    };
    closure(argc + 1);

    let mut closure = |string, arg: isize| {
        unsafe {
            libc::printf(string as *const str as *const i8, arg);
        }
    };
    closure("Both args: %d\n\0", argc + 10);

    0
}
