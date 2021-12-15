// needs-asm-support
#![feature(naked_functions)]

use std::arch::asm;

#[track_caller] //~ ERROR cannot use `#[track_caller]` with `#[naked]`
#[naked]
extern "C" fn f() {
    asm!("", options(noreturn));
}

struct S;

impl S {
    #[track_caller] //~ ERROR cannot use `#[track_caller]` with `#[naked]`
    #[naked]
    extern "C" fn g() {
        asm!("", options(noreturn));
    }
}

fn main() {}
