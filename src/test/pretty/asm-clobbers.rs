#![feature(asm)]

pub fn main() { unsafe { asm!("" : : : "hello", "world") }; }
