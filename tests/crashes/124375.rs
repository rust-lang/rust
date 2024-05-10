//@ known-bug: #124375
//@ compile-flags: -Zmir-opt-level=0
//@ only-x86_64
#![crate_type = "lib"]
#![feature(naked_functions)]
use std::arch::asm;

#[naked]
pub unsafe extern "C" fn naked_with_args_and_return(a: isize, b: isize) -> isize {
    asm!("lea rax, [rdi + rsi]", "ret", options(noreturn));
}
