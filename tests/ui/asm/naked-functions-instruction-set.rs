//@ add-core-stubs
//@ compile-flags: --target armv5te-unknown-linux-gnueabi
//@ needs-llvm-components: arm
//@ needs-asm-support
//@ build-pass

#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
#[unsafe(naked)]
#[instruction_set(arm::t32)]
extern "C" fn test_thumb() {
    naked_asm!("bx lr");
}

#[no_mangle]
#[unsafe(naked)]
#[instruction_set(arm::a32)]
extern "C" fn test_arm() {
    naked_asm!("bx lr");
}
