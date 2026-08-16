//@ add-minicore
//@ revisions: el eb
//@ assembly-output: emit-asm
//@ [el] compile-flags: --target bpfel-unknown-none -Copt-level=0
//@ [eb] compile-flags: --target bpfeb-unknown-none -Copt-level=0
//@ needs-llvm-components: bpf
//@ min-llvm-version: 23

// Test that on LLVM 23 and higher BPF functions can accept more than 5 arguments.
// Earlier versions had a hard limit of at most 5 arguments.
#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: callee:
// CHECK: r0 = *(u64 *)(r11 + 40)
#[no_mangle]
#[inline(never)]
fn callee(
    _a0: u64,
    _a1: u64,
    _a2: u64,
    _a3: u64,
    _a4: u64,
    _a5: u64,
    _a6: u64,
    _a7: u64,
    _a8: u64,
    a9: u64,
) -> u64 {
    a9
}

// CHECK-LABEL: caller:
// CHECK: [[A5:r[0-9]+]] = *(u64 *)(r11 + 8)
// CHECK: [[A6:r[0-9]+]] = *(u64 *)(r11 + 16)
// CHECK: [[A7:r[0-9]+]] = *(u64 *)(r11 + 24)
// CHECK: [[A8:r[0-9]+]] = *(u64 *)(r11 + 32)
// CHECK: [[A9:r[0-9]+]] = *(u64 *)(r11 + 40)
// CHECK: *(u64 *)(r11 - 8) = [[A5]]
// CHECK: *(u64 *)(r11 - 16) = [[A6]]
// CHECK: *(u64 *)(r11 - 24) = [[A7]]
// CHECK: *(u64 *)(r11 - 32) = [[A8]]
// CHECK: *(u64 *)(r11 - 40) = [[A9]]
// CHECK: call callee
#[no_mangle]
fn caller(
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    a7: u64,
    a8: u64,
    a9: u64,
) -> u64 {
    callee(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9)
}
