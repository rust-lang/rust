//@ add-minicore
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86
//@ needs-asm-support
#![feature(no_core)]
#![no_core]
#![deny(unreachable_code)]

extern crate minicore;
use minicore::*;

unsafe fn such_unsafe() {}

fn goto_fallthough() {
    unsafe {
        asm!(
            "/* {} */",
            label {
                such_unsafe()
                //~^ ERROR [E0133]
            }
        )
    }
}

fn main() {
    goto_fallthough();
}
