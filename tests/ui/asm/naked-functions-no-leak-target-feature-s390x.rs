//@ add-core-stubs
//@ compile-flags: --target s390x-unknown-linux-gnu
//@ build-fail
//@ needs-llvm-components: systemz

#![crate_type = "lib"]
#![feature(no_core, naked_functions, s390x_target_feature)]
#![no_core]

extern crate minicore;
use minicore::*;

// check that a naked function using target features does not keep these features enabled
// for subsequent asm blocks.

#[no_mangle]
#[naked]
#[target_feature(enable = "vector-packed-decimal")]
unsafe extern "C" fn a() {
    naked_asm!("vlrlr   %v24, %r3, 0(%r2)")
}

#[no_mangle]
#[naked]
unsafe extern "C" fn b() {
    naked_asm!("vlrlr   %v24, %r3, 0(%r2)")
}
