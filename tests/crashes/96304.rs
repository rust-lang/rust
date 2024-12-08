//@ known-bug: #96304

#![feature(asm_sym)]
core::arch::global_asm!("/* {} */", sym<&'static ()>::clone);

pub fn main() {}
