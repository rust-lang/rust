//@ needs-asm-support
//@ compile-flags: --test

#![feature(naked_functions)]
#![feature(test)]
#![crate_type = "lib"]

use std::arch::naked_asm;

#[test]
#[naked]
//~^ ERROR [E0736]
extern "C" fn test_naked() {
    unsafe { naked_asm!("") };
}

#[should_panic]
#[test]
#[naked]
//~^ ERROR [E0736]
extern "C" fn test_naked_should_panic() {
    unsafe { naked_asm!("") };
}

#[ignore]
#[test]
#[naked]
//~^ ERROR [E0736]
extern "C" fn test_naked_ignore() {
    unsafe { naked_asm!("") };
}

#[bench]
#[naked]
//~^ ERROR [E0736]
extern "C" fn bench_naked() {
    unsafe { naked_asm!("") };
}
