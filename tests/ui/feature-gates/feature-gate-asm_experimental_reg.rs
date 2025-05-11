//@ add-core-stubs
//@ needs-asm-support
//@ compile-flags: --target s390x-unknown-linux-gnu
//@ needs-llvm-components: systemz

#![feature(no_core, lang_items, rustc_attrs)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

unsafe fn main() {
    asm!("", in("v0") 0);
    //~^ ERROR register class `vreg` can only be used as a clobber in stable
    //~| ERROR type `i32` cannot be used with this register class
}
