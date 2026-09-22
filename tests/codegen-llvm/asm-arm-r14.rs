//@ add-minicore
//@ compile-flags: --target armv7-unknown-linux-gnueabihf
//@ needs-llvm-components: arm
//@ revisions: LLVM22 LLVM23
//@ [LLVM22] max-llvm-major-version: 22
//@ [LLVM23] min-llvm-version: 23

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub unsafe fn r14_output() -> u32 {
    // CHECK-LABEL: define{{.*}}@r14_output
    // LLVM22: asm sideeffect{{.*}}"=&{lr},~{cc},~{memory}"
    // LLVM23: asm sideeffect{{.*}}"=&{r14},~{cc},~{memory}"
    let output: u32;
    asm!("", out("r14") output);
    output
}
