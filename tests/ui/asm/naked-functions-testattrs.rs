//@ needs-asm-support
//@ compile-flags: --test

#![feature(test)]
#![crate_type = "lib"]

use std::arch::naked_asm;

#[test]
#[unsafe(naked)]
//~^ ERROR [E0736]
extern "C" fn test_naked() {
    naked_asm!("")
}

#[should_panic]
#[test]
#[unsafe(naked)]
//~^ ERROR [E0736]
extern "C" fn test_naked_should_panic() {
    naked_asm!("")
}

#[ignore]
#[test]
#[unsafe(naked)]
//~^ ERROR [E0736]
extern "C" fn test_naked_ignore() {
    naked_asm!("")
}

#[bench]
#[unsafe(naked)]
//~^ ERROR [E0736]
extern "C" fn bench_naked() {
    naked_asm!("")
}
