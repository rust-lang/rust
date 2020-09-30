#![feature(llvm_asm)]

const _: () = unsafe { llvm_asm!("nop") };
//~^ ERROR inline assembly

fn main() {}
