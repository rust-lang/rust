//@ add-minicore
//@ build-pass
//@ compile-flags: --target thumbv6m-none-eabi
//@ needs-llvm-components: arm

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
