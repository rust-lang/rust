//@ add-core-stubs
//@ compile-flags: --target aarch64-unknown-linux-gnu
//@ build-fail
//@ needs-llvm-components: arm

#![crate_type = "lib"]
#![feature(no_core, naked_functions)]
#![no_core]

extern crate minicore;
use minicore::*;

// check that a naked function using target features does not keep these features enabled
// for subsequent asm blocks.

#[no_mangle]
#[naked]
#[target_feature(enable = "i8mm")]
unsafe extern "C" fn a() {
    naked_asm!("usdot   v0.4s, v1.16b, v2.4b[3]")
}

#[no_mangle]
#[naked]
unsafe extern "C" fn b() {
    naked_asm!("usdot   v0.4s, v1.16b, v2.4b[3]")
}
