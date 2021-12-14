// compile-flags: -Zunleash-the-miri-inside-of-you
// only-x86_64
#![feature(llvm_asm)]
#![allow(const_err)]
#![allow(deprecated)] // llvm_asm!

use std::arch::asm;

fn main() {}

// Make sure we catch executing inline assembly.
static TEST_BAD1: () = {
    unsafe { llvm_asm!("xor %eax, %eax" ::: "eax"); }
    //~^ ERROR could not evaluate static initializer
    //~| NOTE inline assembly is not supported
    //~| NOTE in this expansion of llvm_asm!
    //~| NOTE in this expansion of llvm_asm!
};

// Make sure we catch executing inline assembly.
static TEST_BAD2: () = {
    unsafe { asm!("nop"); }
    //~^ ERROR could not evaluate static initializer
    //~| NOTE inline assembly is not supported
};
