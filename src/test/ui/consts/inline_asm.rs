#![feature(llvm_asm)]

const _: () = unsafe { llvm_asm!("nop") };
//~^ ERROR contains unimplemented expression type

fn main() {}
