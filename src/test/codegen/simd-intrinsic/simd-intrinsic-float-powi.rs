// ignore-emscripten

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f32x2(pub f32, pub f32);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f32x4(pub f32, pub f32, pub f32, pub f32);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f32x8(pub f32, pub f32, pub f32, pub f32,
                 pub f32, pub f32, pub f32, pub f32);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f32x16(pub f32, pub f32, pub f32, pub f32,
                  pub f32, pub f32, pub f32, pub f32,
                  pub f32, pub f32, pub f32, pub f32,
                  pub f32, pub f32, pub f32, pub f32);

extern "platform-intrinsic" {
    fn simd_fpowi<T>(x: T, b: i32) -> T;
}

// CHECK-LABEL: @fpowi_32x2
#[no_mangle]
pub unsafe fn fpowi_32x2(a: f32x2, b: i32) -> f32x2 {
    // CHECK: call fast <2 x float> @llvm.powi.v2f32
    simd_fpowi(a, b)
}

// CHECK-LABEL: @fpowi_32x4
#[no_mangle]
pub unsafe fn fpowi_32x4(a: f32x4, b: i32) -> f32x4 {
    // CHECK: call fast <4 x float> @llvm.powi.v4f32
    simd_fpowi(a, b)
}

// CHECK-LABEL: @fpowi_32x8
#[no_mangle]
pub unsafe fn fpowi_32x8(a: f32x8, b: i32) -> f32x8 {
    // CHECK: call fast <8 x float> @llvm.powi.v8f32
    simd_fpowi(a, b)
}

// CHECK-LABEL: @fpowi_32x16
#[no_mangle]
pub unsafe fn fpowi_32x16(a: f32x16, b: i32) -> f32x16 {
    // CHECK: call fast <16 x float> @llvm.powi.v16f32
    simd_fpowi(a, b)
}

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f64x2(pub f64, pub f64);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f64x4(pub f64, pub f64, pub f64, pub f64);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f64x8(pub f64, pub f64, pub f64, pub f64,
                 pub f64, pub f64, pub f64, pub f64);

// CHECK-LABEL: @fpowi_64x4
#[no_mangle]
pub unsafe fn fpowi_64x4(a: f64x4, b: i32) -> f64x4 {
    // CHECK: call fast <4 x double> @llvm.powi.v4f64
    simd_fpowi(a, b)
}

// CHECK-LABEL: @fpowi_64x2
#[no_mangle]
pub unsafe fn fpowi_64x2(a: f64x2, b: i32) -> f64x2 {
    // CHECK: call fast <2 x double> @llvm.powi.v2f64
    simd_fpowi(a, b)
}

// CHECK-LABEL: @fpowi_64x8
#[no_mangle]
pub unsafe fn fpowi_64x8(a: f64x8, b: i32) -> f64x8 {
    // CHECK: call fast <8 x double> @llvm.powi.v8f64
    simd_fpowi(a, b)
}
