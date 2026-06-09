//! Regression test for #97724, recognising ttbr0_el2 as a valid armv8 system register
//@ add-minicore
//@ build-pass
//@ compile-flags: --target aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64
//@ ignore-backends: gcc
#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

static PT: [u64; 512] = [0; 512];
fn main() {
    unsafe {
        asm!("msr ttbr0_el2, {pt}", pt = in(reg) &PT as *const _ );
    }
}
