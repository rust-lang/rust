//@ needs-asm-support
//@ compile-flags: --test

#![allow(undefined_naked_function_abi)]
#![feature(naked_functions)]
#![feature(test)]
#![crate_type = "lib"]

use std::arch::asm;

#[test]
#[naked]
//~^ ERROR [E0736]
fn test_naked() {
    unsafe { asm!("", options(noreturn)) };
}

#[should_panic]
#[test]
#[naked]
//~^ ERROR [E0736]
fn test_naked_should_panic() {
    unsafe { asm!("", options(noreturn)) };
}

#[ignore]
#[test]
#[naked]
//~^ ERROR [E0736]
fn test_naked_ignore() {
    unsafe { asm!("", options(noreturn)) };
}

#[bench]
#[naked]
//~^ ERROR [E0736]
fn bench_naked() {
    unsafe { asm!("", options(noreturn)) };
}
