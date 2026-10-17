//@ check-pass
//@ needs-asm-support
#![crate_type = "lib"]

use std::arch::naked_asm;

#[unsafe(naked)]
pub extern "C" fn naked(p: char) -> u128 {
    //~^ WARN uses type `char`
    naked_asm!("")
}
