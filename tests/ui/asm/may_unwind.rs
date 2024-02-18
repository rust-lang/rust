//@ run-pass
//@ needs-asm-support

#![feature(asm_unwind)]

use std::arch::asm;

fn main() {
    unsafe { asm!("", options(may_unwind)) };
}
