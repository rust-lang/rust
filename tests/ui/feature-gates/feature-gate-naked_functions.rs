//@ needs-asm-support

use std::arch::asm;

#[naked]
//~^ the `#[naked]` attribute is an experimental feature
extern "C" fn naked() {
    asm!("", options(noreturn))
    //~^ ERROR: requires unsafe
}

#[naked]
//~^ the `#[naked]` attribute is an experimental feature
extern "C" fn naked_2() -> isize {
    asm!("", options(noreturn))
    //~^ ERROR: requires unsafe
}

fn main() {}
