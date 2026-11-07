//@ add-minicore
//@ revisions: aarch64 loongarch
//@ [aarch64] compile-flags: --target aarch64-unknown-linux-gnu -C target-feature=+sve
//@ [aarch64] needs-llvm-components: aarch64
//@ [loongarch] compile-flags: --target loongarch64-unknown-none
//@ [loongarch] needs-llvm-components: loongarch
//@ ignore-backends: gcc

#![feature(no_core, lang_items, rustc_attrs, repr_simd)]
#![crate_type = "rlib"]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::*;

#[cfg(aarch64)]
#[rustc_scalable_vector(4)]
pub struct svint32_t(i32);

#[cfg(aarch64)]
impl Copy for svint32_t {}

#[cfg(aarch64)]
#[rustc_scalable_vector(16)]
pub struct svbool_t(bool);

#[cfg(aarch64)]
impl Copy for svbool_t {}

#[cfg(aarch64)]
unsafe fn vector(x: svint32_t) {
    asm!("/* {0} */", in(vreg) x);
    //[aarch64]~^ ERROR type `svint32_t` cannot be used with this register class in stable

    asm!("/* {0} */", in(vreg_low16) x);
    //[aarch64]~^ ERROR type `svint32_t` cannot be used with this register class in stable

    asm!("", in("z0") x);
    //[aarch64]~^ ERROR type `svint32_t` cannot be used with this register class in stable
}

#[cfg(aarch64)]
unsafe fn predicate(p: svbool_t) {
    asm!("/* {0} */", in(preg) p);
    //[aarch64]~^ ERROR register class `preg` can only be used as a clobber in stable
    //[aarch64]~| ERROR type `svbool_t` cannot be used with this register class in stable
}

#[cfg(loongarch)]
#[repr(simd)]
pub struct i8x16([i8; 16]);

#[cfg(loongarch)]
impl Copy for i8x16 {}

#[cfg(loongarch)]
unsafe fn main(x: i8x16) -> i8x16 {
    let y;
    asm!("xvadd.h {1:u}, {0:u}, {0:u}", out(vreg) y, in(vreg) x);
    //[loongarch]~^ ERROR register class `vreg` can only be used as a clobber in stable
    //[loongarch]~| ERROR register class `vreg` can only be used as a clobber in stable
    //[loongarch]~| ERROR type `i8x16` cannot be used with this register class in stable
    //[loongarch]~| ERROR type `i8x16` cannot be used with this register class in stable
    y
}
