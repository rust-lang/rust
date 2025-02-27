//@ add-core-stubs
//@ assembly-output: emit-asm
//@ compile-flags: --target aarch64-unknown-linux-gnu -Ctarget-feature=+lse
//@ needs-llvm-components: aarch64

#![crate_type = "lib"]
#![feature(no_core, naked_functions)]
#![no_core]

extern crate minicore;
use minicore::*;

// check that a naked function using target features does not disable these features for subsequent
// asm blocks.

// CHECK-LABEL: a:
#[no_mangle]
#[naked]
#[target_feature(enable = "lse")]
unsafe extern "C" fn a() {
    naked_asm!("casp x2, x3, x2, x3, [x1]")
}

// CHECK-LABEL: b:
#[no_mangle]
#[naked]
unsafe extern "C" fn b() {
    naked_asm!("casp x2, x3, x2, x3, [x1]")
}
