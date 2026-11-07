//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: linux-x86-gnu-fs-true linux-x86-gnu-fs-false
//@[linux-x86-gnu-fs-true] compile-flags: --target x86_64-unknown-linux-gnu -Zfunction-sections=true
//@[linux-x86-gnu-fs-true] needs-llvm-components: x86
//@[linux-x86-gnu-fs-false] compile-flags: --target x86_64-unknown-linux-gnu -Zfunction-sections=false
//@[linux-x86-gnu-fs-false] needs-llvm-components: x86
//
//@ revisions: macos-aarch64-fs-true macos-aarch64-fs-false
//@[macos-aarch64-fs-true] compile-flags: --target aarch64-apple-darwin -Zfunction-sections=true
//@[macos-aarch64-fs-true] needs-llvm-components: aarch64
//@[macos-aarch64-fs-false] compile-flags: --target aarch64-apple-darwin -Zfunction-sections=false
//@[macos-aarch64-fs-false] needs-llvm-components: aarch64
//
//@ revisions: windows-x86-gnu-fs-true windows-x86-gnu-fs-false
//@[windows-x86-gnu-fs-true] compile-flags: --target x86_64-pc-windows-gnu -Zfunction-sections=true
//@[windows-x86-gnu-fs-true] needs-llvm-components: x86
//@[windows-x86-gnu-fs-false] compile-flags: --target x86_64-pc-windows-gnu -Zfunction-sections=false
//@[windows-x86-gnu-fs-false] needs-llvm-components: x86
//
//@ revisions: windows-x86-msvc-fs-true windows-x86-msvc-fs-false
//@[windows-x86-msvc-fs-true] compile-flags: --target x86_64-pc-windows-msvc -Zfunction-sections=true
//@[windows-x86-msvc-fs-true] needs-llvm-components: x86
//@[windows-x86-msvc-fs-false] compile-flags: --target x86_64-pc-windows-msvc -Zfunction-sections=false
//@[windows-x86-msvc-fs-false] needs-llvm-components: x86
//
//@ revisions: x86-uefi-fs-true x86-uefi-fs-false
//@[x86-uefi-fs-true] compile-flags: --target x86_64-unknown-uefi -Zfunction-sections=true
//@[x86-uefi-fs-true] needs-llvm-components: x86
//@[x86-uefi-fs-false] compile-flags: --target x86_64-unknown-uefi -Zfunction-sections=false
//@[x86-uefi-fs-false] needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

// Tests that naked and non-naked functions emit the same directives when (not) using
// -Zfunction-sections. This setting is ignored on macos, off by default on windows gnu,
// and on by default in the remaining revisions tested here.

extern crate minicore;
use minicore::*;

#[unsafe(naked)]
#[unsafe(no_mangle)]
extern "C" fn naked_ret() {
    // linux-x86-gnu-fs-true: .section .text.naked_ret,"ax",@progbits
    // linux-x86-gnu-fs-false: .text
    //
    // macos-aarch64-fs-true:  .section __TEXT,__text,regular,pure_instructions
    // macos-aarch64-fs-false: .section __TEXT,__text,regular,pure_instructions
    //
    // NOTE: the regular function below adds `unique,0` at the end, but we have no way of generating
    // the unique ID to use there, so don't emit that part.
    //
    // windows-x86-gnu-fs-true: .section .text$naked_ret,"xr",one_only,naked_ret
    // windows-x86-msvc-fs-true: .section .text,"xr",one_only,naked_ret
    // x86-uefi-fs-true: .section .text,"xr",one_only,naked_ret
    //
    // windows-x86-gnu-fs-false: .text
    // windows-x86-msvc-fs-false: .text
    // x86-uefi-fs-false: .text
    //
    // CHECK-LABEL: naked_ret:
    naked_asm!("ret")
}

// Use a different section here so that `regular_ret` has to explicitly specify the section.
#[link_section = cfg_select!(
    target_os = "macos" => "__FOO,bar",
    _ => ".bar",
)]
#[unsafe(no_mangle)]
extern "C" fn omarker() -> i32 {
    // CHECK-LABEL: omarker:
    32
}

#[unsafe(no_mangle)]
extern "C" fn regular_ret() {
    // linux-x86-gnu-fs-true: .section .text.regular_ret,"ax",@progbits
    // linux-x86-gnu-fs-false: .text
    //
    // macos-aarch64-fs-true:  .section __TEXT,__text,regular,pure_instructions
    // macos-aarch64-fs-false: .section __TEXT,__text,regular,pure_instructions
    //
    // windows-x86-gnu-fs-true: .section .text$regular_ret,"xr",one_only,regular_ret,unique,0
    // windows-x86-msvc-fs-true: .section .text,"xr",one_only,regular_ret,unique,0
    // x86-uefi-fs-true: .section .text,"xr",one_only,regular_ret,unique,0
    //
    // windows-x86-gnu-fs-false: .text
    // windows-x86-msvc-fs-false: .text
    // x86-uefi-fs-false: .text
    //
    // CHECK-LABEL: regular_ret:
}
