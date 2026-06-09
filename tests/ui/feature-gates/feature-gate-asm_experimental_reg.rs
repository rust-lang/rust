//@ add-minicore
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86
//@ ignore-backends: gcc

#![feature(no_core, lang_items, rustc_attrs)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

unsafe fn main() {
    asm!("{:x}", in(xmm_reg) 0u128);
    //~^ ERROR type `u128` cannot be used with this register class in stable
}
