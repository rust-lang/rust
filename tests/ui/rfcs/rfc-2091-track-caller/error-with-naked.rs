//@ needs-asm-support
#![feature(naked_functions)]

use std::arch::naked_asm;

#[track_caller] //~ ERROR [E0736]
//~^ ERROR `#[track_caller]` requires Rust ABI
#[naked]
extern "C" fn f() {
    unsafe {
        naked_asm!("");
    }
}

struct S;

impl S {
    #[track_caller] //~ ERROR [E0736]
    //~^ ERROR `#[track_caller]` requires Rust ABI
    #[naked]
    extern "C" fn g() {
        unsafe {
            naked_asm!("");
        }
    }
}

fn main() {}
