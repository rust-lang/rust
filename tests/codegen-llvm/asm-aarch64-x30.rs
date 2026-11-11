//@ add-minicore
//@ compile-flags: --target aarch64-unknown-none
//@ needs-llvm-components: aarch64
//@ revisions: LLVM22 LLVM23
//@ [LLVM22] max-llvm-major-version: 22
//@ [LLVM23] min-llvm-version: 23

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub unsafe fn x30_output() -> u64 {
    // CHECK-LABEL: define{{.*}}@x30_output
    // LLVM22: asm sideeffect{{.*}}"=&{lr},~{cc},~{memory}"
    // LLVM23: asm sideeffect{{.*}}"=&{x30},~{cc},~{memory}"
    let output: u64;
    asm!("", out("x30") output);
    output
}
