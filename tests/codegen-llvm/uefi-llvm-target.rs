// Checks that the UEFI targets are lowered through LLVM's dedicated UEFI
// targets with LLVM 24 or later, and through the Windows targets they used
// before with older LLVMs.

//@ add-minicore
//@ revisions: x86_64 aarch64 x86_64_llvm23 aarch64_llvm23
//@[x86_64] compile-flags: --target x86_64-unknown-uefi
//@[x86_64] needs-llvm-components: x86
//@[x86_64] min-llvm-version: 24
//@[aarch64] compile-flags: --target aarch64-unknown-uefi
//@[aarch64] needs-llvm-components: aarch64
//@[aarch64] min-llvm-version: 24
//@[x86_64_llvm23] compile-flags: --target x86_64-unknown-uefi
//@[x86_64_llvm23] needs-llvm-components: x86
//@[x86_64_llvm23] max-llvm-major-version: 23
//@[aarch64_llvm23] compile-flags: --target aarch64-unknown-uefi
//@[aarch64_llvm23] needs-llvm-components: aarch64
//@[aarch64_llvm23] max-llvm-major-version: 23
//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

//x86_64: target triple = "x86_64-unknown-uefi"
//aarch64: target triple = "aarch64-unknown-uefi"
//x86_64_llvm23: target triple = "x86_64-unknown-windows-msvc"
//aarch64_llvm23: target triple = "aarch64-unknown-windows-msvc"
#[no_mangle]
pub extern "C" fn nop() {}
