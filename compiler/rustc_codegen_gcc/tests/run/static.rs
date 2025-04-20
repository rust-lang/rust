// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 10
//      14
//      1
//      12
//      12
//      1

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use mini_core::*;

struct Test {
    field: isize,
}

struct WithRef {
    refe: &'static Test,
}

static mut CONSTANT: isize = 10;

static mut TEST: Test = Test { field: 12 };

static mut TEST2: Test = Test { field: 14 };

static mut WITH_REF: WithRef = WithRef { refe: unsafe { &TEST } };

#[no_mangle]
extern "C" fn main(argc: isize, _argv: *const *const u8) -> i32 {
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, CONSTANT);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, TEST2.field);
        TEST2.field = argc;
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, TEST2.field);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, WITH_REF.refe.field);
        WITH_REF.refe = &TEST2;
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, TEST.field);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, WITH_REF.refe.field);
    }
    0
}
