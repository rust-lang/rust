//@ add-minicore
//@ compile-flags: --target armv7-unknown-linux-gnueabihf
//@ needs-llvm-components: arm

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub unsafe fn r14_output() -> u32 {
    // CHECK-LABEL: define{{.*}}@r14_output
    // CHECK: asm sideeffect{{.*}}"=&{lr},~{cc},~{memory}"
    let output: u32;
    asm!("", out("r14") output);
    output
}
