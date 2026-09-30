//! Verify that the breakpoint operation emits the instructions we guarantee.

//@ add-minicore
//@ assembly-output: emit-asm
//@ revisions: AARCH64 I686 X86_64
//@ [AARCH64] compile-flags: --target aarch64-unknown-linux-gnu
//@ [AARCH64] needs-llvm-components: aarch64
//@ [I686] compile-flags: --target i686-unknown-linux-gnu
//@ [I686] needs-llvm-components: x86
//@ [X86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@ [X86_64] needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]

extern crate minicore;

#[rustc_intrinsic]
#[rustc_nounwind]
fn breakpoint();

// CHECK-LABEL: call_breakpoint
// AARCH64: brk #0xf000
// I686: int3
// X86_64: int3
#[unsafe(no_mangle)]
pub fn call_breakpoint() {
    breakpoint();
}
