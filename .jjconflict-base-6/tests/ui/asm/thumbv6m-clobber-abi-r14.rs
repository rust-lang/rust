//@ add-minicore
//@ build-pass
//@ compile-flags: --target thumbv6m-none-eabi
//@ needs-llvm-components: arm
//@ ignore-backends: gcc

#![feature(no_core)]
#![no_core]
#![crate_type = "rlib"]

extern crate minicore;
use minicore::*;

pub fn clobber_c_abi() {
    unsafe {
        asm!("nop", clobber_abi("C"));
    }
}
