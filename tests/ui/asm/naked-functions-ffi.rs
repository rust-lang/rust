// check-pass
// needs-asm-support
#![feature(naked_functions)]
#![crate_type = "lib"]

use std::arch::asm;

#[naked]
pub extern "C" fn naked(p: char) -> u128 {
    //~^ WARN uses type `char`
    unsafe {
        asm!("", options(noreturn));
    }
}
