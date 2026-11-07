//@ add-minicore
//@ build-fail
//@ compile-flags: --target s390x-unknown-linux-gnu -Z deduplicate-diagnostics=yes
//@ needs-llvm-components: systemz
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
#[target_feature(enable = "vector-packed-decimal")]
unsafe extern "C" fn a() {
    naked_asm!("vlrlr   %v24, %r3, 0(%r2)")
}

//~? ERROR instruction requires: vector-packed-decimal

#[no_mangle]
#[unsafe(naked)]
unsafe extern "C" fn b() {
    naked_asm!("vlrlr   %v24, %r3, 0(%r3)")
}
