//@ revisions: x86_64 aarch64
//
//@[aarch64] only-aarch64
//@[x86_64] only-x86_64
//
//@ build-pass
//@ needs-asm-support

#![feature(naked_functions_rustic_abi, rust_cold_cc)]
#![crate_type = "lib"]

use std::arch::{asm, naked_asm};

#[unsafe(naked)]
pub fn rust_implicit() {
    naked_asm!("ret");
}

#[unsafe(naked)]
pub extern "Rust" fn rust_explicit() {
    naked_asm!("ret");
}

#[unsafe(naked)]
pub extern "rust-cold" fn rust_cold() {
    naked_asm!("ret");
}
