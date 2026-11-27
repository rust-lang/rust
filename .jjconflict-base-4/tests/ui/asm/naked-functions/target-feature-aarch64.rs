//@ add-minicore
//@ build-fail
//@ revisions: vanilla sha3
//@ compile-flags: --target aarch64-unknown-linux-gnu -Z deduplicate-diagnostics=yes
//@[sha3] compile-flags: -Ctarget-feature=+sha3
//@ needs-llvm-components: aarch64
//@ min-llvm-version: 23

#![crate_type = "lib"]
#![feature(no_core, naked_functions_target_feature)]
#![no_core]

extern crate minicore;
use minicore::*;

// check that a naked function using target features does not keep these features enabled
// for subsequent asm blocks.

#[no_mangle]
#[unsafe(naked)]
#[target_feature(enable = "i8mm")]
unsafe extern "C" fn a() {
    naked_asm!("usdot   v0.4s, v1.16b, v2.4b[3]")
}

//~? ERROR instruction requires: i8mm

#[no_mangle]
#[unsafe(naked)]
unsafe extern "C" fn c() {
    naked_asm!("usdot   v0.4s, v2.16b, v2.4b[3]")
}

#[no_mangle]
#[unsafe(naked)]
#[target_feature(enable = "sha3")]
unsafe extern "C" fn d() {
    naked_asm!("eor3 v0.16b, v1.16b, v2.16b, v3.16b")
}

//[vanilla]~? ERROR instruction requires: sha3

#[no_mangle]
#[unsafe(naked)]
unsafe extern "C" fn b() {
    naked_asm!("eor3 v0.16b, v1.16b, v2.16b, v3.16b")
}
