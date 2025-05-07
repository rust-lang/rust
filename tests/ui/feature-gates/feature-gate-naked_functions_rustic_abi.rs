//@ needs-asm-support
//@ only-x86_64

#![feature(rust_cold_cc)]

use std::arch::naked_asm;

#[unsafe(naked)]
pub unsafe fn rust_implicit() {
    //~^ ERROR `#[naked]` is currently unstable on `extern "Rust"` functions
    naked_asm!("ret");
}

#[unsafe(naked)]
pub unsafe extern "Rust" fn rust_explicit() {
    //~^ ERROR `#[naked]` is currently unstable on `extern "Rust"` functions
    naked_asm!("ret");
}

#[unsafe(naked)]
pub unsafe extern "rust-cold" fn rust_cold() {
    //~^ ERROR `#[naked]` is currently unstable on `extern "rust-cold"` functions
    naked_asm!("ret");
}

fn main() {}
