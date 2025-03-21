//@ needs-asm-support
//@ only-x86_64

#![feature(naked_functions)]

use std::arch::naked_asm;

#[naked]
#[target_feature(enable = "avx2")]
//~^ ERROR: `#[target_feature(/* ... */)]` is currently unstable on `#[naked]` functions
extern "C" fn naked() {
    unsafe { naked_asm!("") }
}

fn main() {}
