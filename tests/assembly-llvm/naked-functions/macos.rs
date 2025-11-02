//@ revisions: x86 aarch64
//@ add-core-stubs
//@ assembly-output: emit-asm
//
//@[aarch64] compile-flags: --target aarch64-apple-darwin
//@[aarch64] needs-llvm-components: aarch64
//
//@[x86] compile-flags: --target x86_64-apple-darwin
//@[x86] needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(no_core, asm_experimental_arch)]
#![no_core]

// tests that naked functions on macos emit `.private_extern {asm_name}`.
//
// Without this directive, LTO may fail because the symbol is not visible.
// See also https://github.com/rust-lang/rust/issues/148307.

extern crate minicore;
use minicore::*;

// CHECK: .p2align 2
// CHECK: .private_extern
// CHECK: ret
#[unsafe(naked)]
extern "C" fn ret() {
    naked_asm!("ret")
}

#[no_mangle]
pub fn entry() {
    ret()
}
