// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck
// needs-asm-support

#![feature(llvm_asm)]
#![allow(deprecated)] // llvm_asm!

use std::arch::asm;

fn main() {
    asm!("nop"); //~ ERROR use of inline assembly is unsafe and requires unsafe function or block
    llvm_asm!("nop"); //~ ERROR use of inline assembly is unsafe and requires unsafe function or block
}
