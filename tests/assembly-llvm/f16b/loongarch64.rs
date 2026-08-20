//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 --target loongarch64-unknown-linux-gnu
//@ needs-llvm-components: loongarch

#![feature(f16b, no_core)]
#![no_core]
#![crate_type = "lib"]
#![allow(improper_ctypes_definitions)]

// Check that the assembly that rustc generates matches what clang emits.

extern crate minicore;

use minicore::From;
use minicore::num::f16b;

// CHECK-LABEL: identity_f16b:
// CHECK: movfr2gr.s $a0, $fa0
// CHECK-NEXT: lu12i.w $a1, -16
// CHECK-NEXT: or $a0, $a0, $a1
// CHECK-NEXT: movgr2fr.w $fa0, $a0
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn identity_f16b(value: f16b) -> f16b {
    value
}

// CHECK-LABEL: f16b_to_bits:
// CHECK: movfr2gr.s $a0, $fa0
// CHECK-NEXT: bstrpick.d $a0, $a0, 15, 0
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn f16b_to_bits(value: f16b) -> u16 {
    value.to_bits()
}

// CHECK-LABEL: f16b_from_bits:
// CHECK: lu12i.w $a1, -16
// CHECK-NEXT: or $a0, $a0, $a1
// CHECK-NEXT: movgr2fr.w $fa0, $a0
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn f16b_from_bits(bits: u16) -> f16b {
    f16b::from_bits(bits)
}

// CHECK-LABEL: widen_f16b:
// CHECK: movfr2gr.s $a0, $fa0
// CHECK-NEXT: slli.d $a0, $a0, 16
// CHECK-NEXT: movgr2fr.w $fa0, $a0
// CHECK-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn widen_f16b(value: f16b) -> f32 {
    f32::from(value)
}
