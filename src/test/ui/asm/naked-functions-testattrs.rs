// needs-asm-support
// compile-flags: --test

#![allow(undefined_naked_function_abi)]
#![feature(naked_functions)]
#![feature(test)]
#![crate_type = "lib"]

use std::arch::asm;

#[test]
#[naked]
//~^ ERROR cannot use testing attributes with `#[naked]`
fn test_naked() {
    unsafe { asm!("", options(noreturn)) };
}

#[should_panic]
#[test]
#[naked]
//~^ ERROR cannot use testing attributes with `#[naked]`
fn test_naked_should_panic() {
    unsafe { asm!("", options(noreturn)) };
}

#[ignore]
#[test]
#[naked]
//~^ ERROR cannot use testing attributes with `#[naked]`
fn test_naked_ignore() {
    unsafe { asm!("", options(noreturn)) };
}

#[bench]
#[naked]
//~^ ERROR cannot use testing attributes with `#[naked]`
fn bench_naked() {
    unsafe { asm!("", options(noreturn)) };
}
