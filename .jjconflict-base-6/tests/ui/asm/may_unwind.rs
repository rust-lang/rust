//@ run-pass
//@ needs-asm-support
//@ needs-unwind
//@ ignore-backends: gcc

#![feature(asm_unwind)]

use std::arch::asm;

fn main() {
    unsafe { asm!("", options(may_unwind)) };
}
