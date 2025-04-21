//@ needs-asm-support

use std::arch::naked_asm;
//~^ ERROR use of unstable library feature `naked_functions`

#[naked] //~ ERROR unsafe attribute used without unsafe
//~^ ERROR the `#[naked]` attribute is an experimental feature
extern "C" fn naked() {
    naked_asm!("")
    //~^ ERROR use of unstable library feature `naked_functions`
}

#[naked] //~ ERROR unsafe attribute used without unsafe
//~^ ERROR the `#[naked]` attribute is an experimental feature
extern "C" fn naked_2() -> isize {
    naked_asm!("")
    //~^ ERROR use of unstable library feature `naked_functions`
}

fn main() {}
