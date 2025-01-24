//@ known-bug: #117877
//@ edition:2021
//@ needs-rustc-debug-assertions
//@ only-x86_64
#![feature(asm_const)]

use std::arch::asm;

async unsafe fn foo<'a>() {
    asm!("/* {0} */", const N);
}

fn main() {}
