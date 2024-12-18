//@ add-core-stubs
//@ compile-flags: --target armv5te-unknown-linux-gnueabi
//@ needs-llvm-components: arm
//@ needs-asm-support
//@ build-pass

#![feature(no_core)]
#![no_core]
#![crate_type = "rlib"]

extern crate minicore;
use minicore::*;

// ARM uses R11 for the frame pointer, make sure R7 is usable.
#[instruction_set(arm::a32)]
pub fn arm() {
    unsafe {
        asm!("", out("r7") _);
    }
}

// Thumb uses R7 for the frame pointer, make sure R11 is usable.
#[instruction_set(arm::t32)]
pub fn thumb() {
    unsafe {
        asm!("", out("r11") _);
    }
}
