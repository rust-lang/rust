//@ add-minicore
//@ compile-flags: --target loongarch64-unknown-none
//@ needs-llvm-components: loongarch
//@ ignore-backends: gcc

#![feature(no_core, lang_items, rustc_attrs, repr_simd)]
#![crate_type = "rlib"]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct i8x16([i8; 16]);

impl Copy for i8x16 {}

unsafe fn main(x: i8x16) -> i8x16 {
    let y;
    asm!("xvadd.h {1:u}, {0:u}, {0:u}", out(vreg) y, in(vreg) x);
    //~^ ERROR register class `vreg` can only be used as a clobber in stable
    //~| ERROR register class `vreg` can only be used as a clobber in stable
    //~| ERROR type `i8x16` cannot be used with this register class in stable
    //~| ERROR type `i8x16` cannot be used with this register class in stable
    y
}
