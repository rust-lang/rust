//@ only-x86_64
//@ needs-asm-support

#![feature(asm_const)]

// Test to make sure that we emit const errors eagerly for inline asm

use std::arch::asm;

fn test<T>() {
    unsafe { asm!("/* {} */", const 1 / 0); }
    //~^ ERROR evaluation of
}

fn main() {}
