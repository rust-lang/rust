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

#[inline(always)]
fn fib(n: u8) -> u8 {
    if n == 0 {
        return 1;
    }
    if n == 1 {
        return 1;
    }
    fib(n - 1) + fib(n - 2)
}

#[inline(always)]
fn fib_b(n: u8) -> u8 {
    if n == 0 {
        return 1;
    }
    if n == 1 {
        return 1;
    }
    fib_a(n - 1) + fib_a(n - 2)
}

#[inline(always)]
fn fib_a(n: u8) -> u8 {
    if n == 0 {
        return 1;
    }
    if n == 1 {
        return 1;
    }
    fib_b(n - 1) + fib_b(n - 2)
}

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    if fib(2) != fib_a(2) {
        intrinsics::abort();
    }
    0
}
