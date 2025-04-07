//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

use std::intrinsics::simd::simd_floor;

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f32x2(pub [f32; 2]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f32x4(pub [f32; 4]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f32x8(pub [f32; 8]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f32x16(pub [f32; 16]);

// CHECK-LABEL: @floor_32x2
#[no_mangle]
pub unsafe fn floor_32x2(a: f32x2) -> f32x2 {
    // CHECK: call <2 x float> @llvm.floor.v2f32
    simd_floor(a)
}

// CHECK-LABEL: @floor_32x4
#[no_mangle]
pub unsafe fn floor_32x4(a: f32x4) -> f32x4 {
    // CHECK: call <4 x float> @llvm.floor.v4f32
    simd_floor(a)
}

// CHECK-LABEL: @floor_32x8
#[no_mangle]
pub unsafe fn floor_32x8(a: f32x8) -> f32x8 {
    // CHECK: call <8 x float> @llvm.floor.v8f32
    simd_floor(a)
}

// CHECK-LABEL: @floor_32x16
#[no_mangle]
pub unsafe fn floor_32x16(a: f32x16) -> f32x16 {
    // CHECK: call <16 x float> @llvm.floor.v16f32
    simd_floor(a)
}

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f64x2(pub [f64; 2]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f64x4(pub [f64; 4]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f64x8(pub [f64; 8]);

// CHECK-LABEL: @floor_64x4
#[no_mangle]
pub unsafe fn floor_64x4(a: f64x4) -> f64x4 {
    // CHECK: call <4 x double> @llvm.floor.v4f64
    simd_floor(a)
}

// CHECK-LABEL: @floor_64x2
#[no_mangle]
pub unsafe fn floor_64x2(a: f64x2) -> f64x2 {
    // CHECK: call <2 x double> @llvm.floor.v2f64
    simd_floor(a)
}

// CHECK-LABEL: @floor_64x8
#[no_mangle]
pub unsafe fn floor_64x8(a: f64x8) -> f64x8 {
    // CHECK: call <8 x double> @llvm.floor.v8f64
    simd_floor(a)
}
