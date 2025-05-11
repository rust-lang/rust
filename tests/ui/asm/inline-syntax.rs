//@ add-core-stubs
//@ revisions: x86_64 arm
//@[x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86_64] check-pass
//@[x86_64] needs-llvm-components: x86
// LLVM 19+ has full support for 64-bit cookies.
//@[arm] compile-flags: --target armv7-unknown-linux-gnueabihf
//@[arm] build-fail
//@[arm] needs-llvm-components: arm
//@ needs-asm-support

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

pub fn main() {
    unsafe {
        asm!(".intel_syntax noprefix", "nop");
        //[x86_64]~^ WARN avoid using `.intel_syntax`
        //[arm]~^^ ERROR unknown directive
        asm!(".intel_syntax aaa noprefix", "nop");
        //[x86_64]~^ WARN avoid using `.intel_syntax`
        //[arm]~^^ ERROR unknown directive
        asm!(".att_syntax noprefix", "nop");
        //[x86_64]~^ WARN avoid using `.att_syntax`
        //[arm]~^^ ERROR unknown directive
        asm!(".att_syntax bbb noprefix", "nop");
        //[x86_64]~^ WARN avoid using `.att_syntax`
        //[arm]~^^ ERROR unknown directive
        asm!(".intel_syntax noprefix; nop");
        //[x86_64]~^ WARN avoid using `.intel_syntax`
        //[arm]~^^ ERROR unknown directive

        asm!(
            r"
            .intel_syntax noprefix
            nop"
        );
        //[x86_64]~^^^ WARN avoid using `.intel_syntax`
        //[arm]~^^^^ ERROR unknown directive
    }
}

global_asm!(".intel_syntax noprefix", "nop");
//[x86_64]~^ WARN avoid using `.intel_syntax`
// Global assembly errors don't have line numbers, so no error on ARM.

//[arm]~? ERROR unknown directive
//[arm]~? ERROR unknown directive
