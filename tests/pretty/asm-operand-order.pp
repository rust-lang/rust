#![feature(prelude_import)]
#![no_std]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;
//@ pretty-mode:expanded
//@ pp-exact:asm-operand-order.pp
//@ only-x86_64

use std::arch::asm;

pub fn main() {
    unsafe {
        asm!("{0}", in(reg) 4, out("rax") _);
        asm!("{0}", in(reg) 4, out("rax") _, options(nostack));
        asm!("{0} {1}", in(reg) 4, in(reg) 5, out("rax") _);
    }
}
