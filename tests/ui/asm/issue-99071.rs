//@ add-core-stubs
//@ compile-flags: --target thumbv6m-none-eabi
//@ needs-llvm-components: arm
//@ needs-asm-support

#![feature(no_core)]
#![no_core]
#![crate_type = "rlib"]

extern crate minicore;
use minicore::*;

pub fn foo() {
    unsafe {
        asm!("", in("r8") 0);
        //~^ ERROR cannot use register `r8`: high registers (r8+) can only be used as clobbers in Thumb-1 code
    }
}
