//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 --target riscv64gc-unknown-linux-gnu
//@ needs-llvm-components: riscv

#![feature(f16b, no_core)]
#![no_core]
#![crate_type = "lib"]
#![allow(improper_ctypes_definitions)]

// Check that the assembly that rustc generates matches what clang emits.

extern crate minicore;

use minicore::From;
use minicore::num::f16b;

// CHECK-LABEL: identity_f16b:
// CHECK: fmv.x.w a0, fa0
// CHECK-NEXT: lui a1, 1048560
// CHECK-NEXT: or a0, a0, a1
// CHECK-NEXT: fmv.w.x fa0, a0
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn identity_f16b(value: f16b) -> f16b {
    value
}

// CHECK-LABEL: f16b_to_bits:
// CHECK: fmv.x.w a0, fa0
// CHECK-NEXT: slli a0, a0, 48
// CHECK-NEXT: srli a0, a0, 48
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn f16b_to_bits(value: f16b) -> u16 {
    value.to_bits()
}

// CHECK-LABEL: f16b_from_bits:
// CHECK: lui a1, 1048560
// CHECK-NEXT: or a0, a0, a1
// CHECK-NEXT: fmv.w.x fa0, a0
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn f16b_from_bits(bits: u16) -> f16b {
    f16b::from_bits(bits)
}

// CHECK-LABEL: widen_f16b:
// CHECK: fmv.x.w a0, fa0
// CHECK-NEXT: slli a0, a0, 16
// CHECK-NEXT: fmv.w.x fa0, a0
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn widen_f16b(value: f16b) -> f32 {
    f32::from(value)
}
