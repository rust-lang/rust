//@ revisions: windows-x86-gnu windows-x86-msvc x86-uefi
//@ add-minicore
//@ assembly-output: emit-asm
//
//@[windows-x86-gnu] compile-flags: --target x86_64-pc-windows-gnu
//@[windows-x86-gnu] needs-llvm-components: x86
//
//@[windows-x86-msvc] compile-flags: --target x86_64-pc-windows-msvc
//@[windows-x86-msvc] needs-llvm-components: x86
//
//@[x86-uefi] compile-flags: --target x86_64-unknown-uefi
//@[x86-uefi] needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

// Tests that naked and non-naked functions emit the same directives when the function uses
// `#[link_section = "..."]`.

extern crate minicore;
use minicore::*;

#[unsafe(naked)]
#[unsafe(no_mangle)]
#[unsafe(link_section = "naked")]
extern "C" fn naked_ret() {
    // CHECK: .def    naked_ret;
    // CHECK-NEXT: .scl    2;
    // CHECK-NEXT: .type   32;
    // CHECK-NEXT: .endef
    // CHECK-NEXT: .section    naked,"xr"
    // CHECK-NEXT: .globl  naked_ret
    // CHECK-NEXT: .p2align    4
    // CHECK-NEXT: naked_ret:
    // CHECK-NEXT: retq
    naked_asm!("ret")
}

#[unsafe(no_mangle)]
#[unsafe(link_section = "regular")]
extern "C" fn regular_ret() {
    // CHECK: .def    regular_ret;
    // CHECK-NEXT: .scl    2;
    // CHECK-NEXT: .type   32;
    // CHECK-NEXT: .endef
    // CHECK-NEXT: .section    regular,"xr"
    // CHECK-NEXT: .globl  regular_ret
    // CHECK-NEXT: .p2align    4
    // CHECK-NEXT: regular_ret:
    // CHECK-NEXT: retq
}
