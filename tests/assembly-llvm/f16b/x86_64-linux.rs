//@ add-minicore
//@ assembly-output: emit-asm
//
//@ compile-flags: -Copt-level=3 -Cllvm-args=-x86-asm-syntax=intel
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86

#![feature(f16b, no_core)]
#![no_core]
#![crate_type = "lib"]
#![allow(improper_ctypes_definitions)]

// Check that the assembly that rustc generates matches what clang emits.

extern crate minicore;

use minicore::From;
use minicore::num::f16b;

// CHECK-LABEL: identity_f16b:
// CHECK: ret
#[unsafe(no_mangle)]
pub extern "C" fn identity_f16b(value: f16b) -> f16b {
    value
}

// CHECK-LABEL: f16b_to_bits:
// CHECK: pextrw eax, xmm0, 0
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn f16b_to_bits(value: f16b) -> u16 {
    value.to_bits()
}

// CHECK-LABEL: f16b_from_bits:
// CHECK: pinsrw xmm0, edi, 0
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn f16b_from_bits(bits: u16) -> f16b {
    f16b::from_bits(bits)
}

// CHECK-LABEL: widen_f16b:
// CHECK: pextrw eax, xmm0, 0
// CHECK-NEXT: shl eax, 16
// CHECK-NEXT: movd xmm0, eax
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn widen_f16b(value: f16b) -> f32 {
    f32::from(value)
}
