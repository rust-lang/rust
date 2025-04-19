//@ check-pass
//@ needs-asm-support
#![feature(naked_functions)]
#![crate_type = "lib"]

use std::arch::naked_asm;

#[unsafe(naked)]
pub extern "C" fn naked(p: char) -> u128 {
    //~^ WARN uses type `char`
    //~| WARN uses type `u128`
    naked_asm!("")
}
