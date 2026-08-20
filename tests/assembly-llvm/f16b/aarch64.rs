//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: AARCH64_LINUX AARCH64_DARWIN AARCH64_BE AARCH64_MSVC ARM64EC_MSVC
//@ [AARCH64_LINUX] compile-flags: -Copt-level=3 --target aarch64-unknown-linux-gnu
//@ [AARCH64_LINUX] needs-llvm-components: aarch64
//@ [AARCH64_BE] compile-flags: -Copt-level=3 --target aarch64_be-unknown-linux-gnu
//@ [AARCH64_BE] needs-llvm-components: aarch64
//@ [AARCH64_DARWIN] compile-flags: -Copt-level=3 --target aarch64-apple-darwin
//@ [AARCH64_DARWIN] needs-llvm-components: aarch64
//@ [AARCH64_MSVC] compile-flags: -Copt-level=3 --target aarch64-pc-windows-msvc
//@ [AARCH64_MSVC] needs-llvm-components: aarch64
//@ [ARM64EC_MSVC] compile-flags: -Copt-level=3 --target arm64ec-pc-windows-msvc
//@ [ARM64EC_MSVC] needs-llvm-components: aarch64
//@ [ARM64EC_MSVC] min-llvm-version: 23

#![feature(f16b, no_core)]
#![no_core]
#![crate_type = "lib"]
#![allow(improper_ctypes_definitions)]

// Check that the assembly that rustc generates matches what clang emits.

extern crate minicore;

use minicore::From;
use minicore::num::f16b;

// CHECK-LABEL: {{^"?[#_]?identity_f16b"?:}}
// CHECK: ret
#[unsafe(no_mangle)]
pub extern "C" fn identity_f16b(value: f16b) -> f16b {
    value
}

// CHECK-LABEL: {{^"?[#_]?f16b_to_bits"?:}}
// AARCH64_LINUX: fmov w0, s0
// AARCH64_BE: fmov w0, s0
// AARCH64_DARWIN: fmov w8, s0
// AARCH64_DARWIN-NEXT: and w0, w8, #0xffff
// AARCH64_MSVC: fmov w0, s0
// ARM64EC_MSVC: fmov w0, s0
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn f16b_to_bits(value: f16b) -> u16 {
    value.to_bits()
}

// CHECK-LABEL: {{^"?[#_]?f16b_from_bits"?:}}
// CHECK: fmov s0, w0
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn f16b_from_bits(bits: u16) -> f16b {
    f16b::from_bits(bits)
}

// CHECK-LABEL: {{^"?[#_]?widen_f16b"?:}}
// CHECK: fmov w8, s0
// CHECK-NEXT: lsl w8, w8, #16
// CHECK-NEXT: fmov s0, w8
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn widen_f16b(value: f16b) -> f32 {
    f32::from(value)
}
