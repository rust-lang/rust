//@ revisions: x86_64 aarch64
//
//@[aarch64] only-aarch64
//@[x86_64] only-x86_64
//
//@ build-pass
//@ needs-asm-support

#![feature(naked_functions, naked_functions_rustic_abi, rust_cold_cc)]
#![crate_type = "lib"]

use std::arch::{asm, naked_asm};

#[naked]
pub unsafe fn rust_implicit() {
    naked_asm!("ret");
}

#[naked]
pub unsafe extern "Rust" fn rust_explicit() {
    naked_asm!("ret");
}

#[naked]
pub unsafe extern "rust-cold" fn rust_cold() {
    naked_asm!("ret");
}
