//@ needs-asm-support

use std::arch::asm;

const _: () = unsafe { asm!("nop") };
//~^ ERROR inline assembly

fn main() {}
