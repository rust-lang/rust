//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: AARCH64_LINUX AARCH64_BE AARCH64_DARWIN AARCH64_MSVC ARM64EC_MSVC
//@[AARCH64_LINUX] compile-flags: -Copt-level=3 --target aarch64-unknown-linux-gnu
//@[AARCH64_LINUX] needs-llvm-components: aarch64
//@[AARCH64_LINUX] filecheck-flags: --check-prefixes AARCH64,AARCH64-NOTAPPLE
//@[AARCH64_BE] compile-flags: -Copt-level=3 --target aarch64_be-unknown-linux-gnu
//@[AARCH64_BE] needs-llvm-components: aarch64
//@[AARCH64_BE] filecheck-flags: --check-prefixes AARCH64,AARCH64-NOTAPPLE
//@[AARCH64_DARWIN] compile-flags: -Copt-level=3 --target aarch64-apple-darwin
//@[AARCH64_DARWIN] needs-llvm-components: aarch64
//@[AARCH64_DARWIN] filecheck-flags: --check-prefixes AARCH64,AARCH64-APPLE
//@[AARCH64_MSVC] compile-flags: -Copt-level=3 --target aarch64-pc-windows-msvc
//@[AARCH64_MSVC] needs-llvm-components: aarch64
//@[AARCH64_MSVC] filecheck-flags: --check-prefixes AARCH64,AARCH64-NOTAPPLE
//@[ARM64EC_MSVC] compile-flags: -Copt-level=3 --target arm64ec-pc-windows-msvc
//@[ARM64EC_MSVC] needs-llvm-components: aarch64
//@[ARM64EC_MSVC] min-llvm-version: 23
//@[ARM64EC_MSVC] filecheck-flags: --check-prefixes AARCH64,AARCH64-NOTAPPLE

#![feature(f16b, no_core)]
#![no_core]
#![crate_type = "lib"]
#![allow(improper_ctypes_definitions)]

// Check that the assembly that rustc generates matches what clang emits.

extern crate minicore;

use minicore::From;
use minicore::num::f16b;

// CHECK-LABEL: identity_f16b
// AARCH64: ret
// X64: ret
#[unsafe(no_mangle)]
pub extern "C" fn identity_f16b(value: f16b) -> f16b {
    value
}

// CHECK-LABEL: f16b_to_bits
// AARCH64-NOTAPPLE: fmov w0, s0
// AARCH64-APPLE: fmov w8, s0
// AARCH64-APPLE-NEXT: and w0, w8, #0xffff
// AARCH64-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn f16b_to_bits(value: f16b) -> u16 {
    value.to_bits()
}

// CHECK-LABEL: f16b_from_bits
// AARCH64: fmov s0, w0
// AARCH64-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn f16b_from_bits(bits: u16) -> f16b {
    f16b::from_bits(bits)
}

// CHECK-LABEL: widen_f16b
// AARCH64: fmov w8, s0
// AARCH64-NEXT: lsl w8, w8, #16
// AARCH64-NEXT: fmov s0, w8
// AARCH64-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn widen_f16b(value: f16b) -> f32 {
    f32::from(value)
}
