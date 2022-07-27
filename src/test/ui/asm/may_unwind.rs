// run-pass
// needs-asm-support
// ignore-uefi no unwind

#![feature(asm_unwind)]

use std::arch::asm;

fn main() {
    unsafe { asm!("", options(may_unwind)) };
}
