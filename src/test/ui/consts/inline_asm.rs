// needs-asm-support

#![feature(asm)]

const _: () = unsafe { asm!("nop") };
//~^ ERROR inline assembly

fn main() {}
