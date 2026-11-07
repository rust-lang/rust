// Compiler:
//
// Run-time:
//   status: 0

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use mini_core::*;

#[repr(C)]
struct Big {
    a: i64,
    b: i64,
    c: i64,
}

#[repr(C, align(64))]
struct Aligned {
    x: i32,
}

#[inline(never)]
#[no_mangle]
extern "C" fn check(_b1: Big, a1: Aligned, _b2: Big, a2: Aligned) -> i32 {
    if (&a1 as *const Aligned as usize) % 64 != 0 {
        return 1;
    }
    if (&a2 as *const Aligned as usize) % 64 != 0 {
        return 2;
    }
    0
}

#[no_mangle]
extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    check(Big { a: 1, b: 2, c: 3 }, Aligned { x: 42 }, Big { a: 4, b: 5, c: 6 }, Aligned { x: 43 })
}
