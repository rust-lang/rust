//@ revisions: macos-x86 macos-aarch64 linux-x86
//@ add-minicore
//@ assembly-output: emit-asm
//
//@[macos-aarch64] compile-flags: --target aarch64-apple-darwin
//@[macos-aarch64] needs-llvm-components: aarch64
//
//@[macos-x86] compile-flags: --target x86_64-apple-darwin
//@[macos-x86] needs-llvm-components: x86
//
//@[linux-x86] compile-flags: --target x86_64-unknown-linux-gnu
//@[linux-x86] needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(no_core, asm_experimental_arch)]
#![no_core]

// Tests that naked functions that are not externally linked (e.g. via `no_mangle`)
// are marked as `Visibility::Hidden` and emit `.private_extern` or `.hidden`.
//
// Without this directive, LTO may fail because the symbol is not visible.
// See also https://github.com/rust-lang/rust/issues/148307.

extern crate minicore;
use minicore::*;

// CHECK: .p2align 2
// macos-x86,macos-aarch64: .private_extern
// linux-x86: .globl
// linux-x86: .hidden
// CHECK: ret
#[unsafe(naked)]
extern "C" fn ret() {
    naked_asm!("ret")
}

// CHECK-LABEL: entry
#[no_mangle]
pub fn entry() {
    ret()
}
