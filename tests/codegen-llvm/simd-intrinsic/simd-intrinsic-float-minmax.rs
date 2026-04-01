//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::*;

// CHECK-LABEL: @fmin
#[no_mangle]
pub unsafe fn fmin(a: f32x4, b: f32x4) -> f32x4 {
    // CHECK: call nsz <4 x float> @llvm.minimumnum.v4f32
    simd_minimum_number_nsz(a, b)
}

// CHECK-LABEL: @fmax
#[no_mangle]
pub unsafe fn fmax(a: f32x4, b: f32x4) -> f32x4 {
    // CHECK: call nsz <4 x float> @llvm.maximumnum.v4f32
    simd_maximum_number_nsz(a, b)
}
