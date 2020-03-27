#![feature(llvm_asm)]

pub fn main() { unsafe { llvm_asm!("" : : : "hello", "world") }; }
