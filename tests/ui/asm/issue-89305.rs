// Regression test for #89305, where a variable was erroneously reported
// as both unused and possibly-uninitialized.

//@ check-pass
//@ needs-asm-support

#![warn(unused)]

use std::arch::asm;

fn main() {
    unsafe {
        let x: () = asm!("nop");
        //~^ WARNING: unused variable: `x`
    }
}
