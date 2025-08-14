//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

use std::intrinsics::simd::{simd_fmax, simd_fmin};

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f32x4(pub [f32; 4]);

// CHECK-LABEL: @fmin
#[no_mangle]
pub unsafe fn fmin(a: f32x4, b: f32x4) -> f32x4 {
    // CHECK: call <4 x float> @llvm.minnum.v4f32
    simd_fmin(a, b)
}

// CHECK-LABEL: @fmax
#[no_mangle]
pub unsafe fn fmax(a: f32x4, b: f32x4) -> f32x4 {
    // CHECK: call <4 x float> @llvm.maxnum.v4f32
    simd_fmax(a, b)
}
