//@ compile-flags: -Copt-level=3 --crate-type=rlib
#![feature(core_intrinsics, repr_simd)]

use std::intrinsics::simd::{simd_eq, simd_fabs};

#[repr(simd)]
pub struct V([f32; 4]);

#[repr(simd)]
pub struct M([i32; 4]);

#[no_mangle]
// CHECK-LABEL: @is_infinite
pub fn is_infinite(v: V) -> M {
    // CHECK: fabs
    // CHECK: cmp oeq
    unsafe { simd_eq(simd_fabs(v), V([f32::INFINITY; 4])) }
}
