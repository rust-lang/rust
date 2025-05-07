//@ add-core-stubs
//@ compile-flags: --target armv7-unknown-linux-gnueabihf
//@ needs-llvm-components: arm

#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

fn main() {
    unsafe {
        asm!("", out("d0") _, out("d1") _);
        asm!("", out("d0") _, out("s1") _);
        //~^ ERROR register `s1` conflicts with register `d0`
    }
}
