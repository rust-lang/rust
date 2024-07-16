//@ only-x86_64
//@ needs-asm-support
#![feature(naked_functions)]
#![crate_type = "lib"]

use std::arch::asm;

#[target_feature(enable = "sse2")]
//~^ ERROR [E0736]
#[naked]
pub unsafe extern "C" fn naked_target_feature() {
    asm!("", options(noreturn));
}
