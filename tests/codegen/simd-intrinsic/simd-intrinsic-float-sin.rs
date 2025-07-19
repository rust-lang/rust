//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_fsin;

// CHECK-LABEL: @fsin_32x2
#[no_mangle]
pub unsafe fn fsin_32x2(a: f32x2) -> f32x2 {
    // CHECK: call <2 x float> @llvm.sin.v2f32
    simd_fsin(a)
}

// CHECK-LABEL: @fsin_32x4
#[no_mangle]
pub unsafe fn fsin_32x4(a: f32x4) -> f32x4 {
    // CHECK: call <4 x float> @llvm.sin.v4f32
    simd_fsin(a)
}

// CHECK-LABEL: @fsin_32x8
#[no_mangle]
pub unsafe fn fsin_32x8(a: f32x8) -> f32x8 {
    // CHECK: call <8 x float> @llvm.sin.v8f32
    simd_fsin(a)
}

// CHECK-LABEL: @fsin_32x16
#[no_mangle]
pub unsafe fn fsin_32x16(a: f32x16) -> f32x16 {
    // CHECK: call <16 x float> @llvm.sin.v16f32
    simd_fsin(a)
}

// CHECK-LABEL: @fsin_64x4
#[no_mangle]
pub unsafe fn fsin_64x4(a: f64x4) -> f64x4 {
    // CHECK: call <4 x double> @llvm.sin.v4f64
    simd_fsin(a)
}

// CHECK-LABEL: @fsin_64x2
#[no_mangle]
pub unsafe fn fsin_64x2(a: f64x2) -> f64x2 {
    // CHECK: call <2 x double> @llvm.sin.v2f64
    simd_fsin(a)
}

// CHECK-LABEL: @fsin_64x8
#[no_mangle]
pub unsafe fn fsin_64x8(a: f64x8) -> f64x8 {
    // CHECK: call <8 x double> @llvm.sin.v8f64
    simd_fsin(a)
}
