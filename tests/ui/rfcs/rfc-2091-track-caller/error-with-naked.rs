//@ needs-asm-support
#![feature(naked_functions)]

use std::arch::asm;

#[track_caller] //~ ERROR [E0736]
//~^ ERROR `#[track_caller]` requires Rust ABI
#[naked]
extern "C" fn f() {
    unsafe {
        asm!("", options(noreturn));
    }
}

struct S;

impl S {
    #[track_caller] //~ ERROR [E0736]
    //~^ ERROR `#[track_caller]` requires Rust ABI
    #[naked]
    extern "C" fn g() {
        unsafe {
            asm!("", options(noreturn));
        }
    }
}

fn main() {}
