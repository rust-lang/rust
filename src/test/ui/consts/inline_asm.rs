#![feature(asm)]

const _: () = unsafe { asm!("nop") };
//~^ ERROR inline assembly

fn main() {}
