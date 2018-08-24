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
    fn simd_fsqrt<T>(x: T) -> T;
}

// CHECK-LABEL: @fsqrt_32x2
#[no_mangle]
pub unsafe fn fsqrt_32x2(a: f32x2) -> f32x2 {
    // CHECK: call fast <2 x float> @llvm.sqrt.v2f32
    simd_fsqrt(a)
}

// CHECK-LABEL: @fsqrt_32x4
#[no_mangle]
pub unsafe fn fsqrt_32x4(a: f32x4) -> f32x4 {
    // CHECK: call fast <4 x float> @llvm.sqrt.v4f32
    simd_fsqrt(a)
}

// CHECK-LABEL: @fsqrt_32x8
#[no_mangle]
pub unsafe fn fsqrt_32x8(a: f32x8) -> f32x8 {
    // CHECK: call fast <8 x float> @llvm.sqrt.v8f32
    simd_fsqrt(a)
}

// CHECK-LABEL: @fsqrt_32x16
#[no_mangle]
pub unsafe fn fsqrt_32x16(a: f32x16) -> f32x16 {
    // CHECK: call fast <16 x float> @llvm.sqrt.v16f32
    simd_fsqrt(a)
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

// CHECK-LABEL: @fsqrt_64x4
#[no_mangle]
pub unsafe fn fsqrt_64x4(a: f64x4) -> f64x4 {
    // CHECK: call fast <4 x double> @llvm.sqrt.v4f64
    simd_fsqrt(a)
}

// CHECK-LABEL: @fsqrt_64x2
#[no_mangle]
pub unsafe fn fsqrt_64x2(a: f64x2) -> f64x2 {
    // CHECK: call fast <2 x double> @llvm.sqrt.v2f64
    simd_fsqrt(a)
}

// CHECK-LABEL: @fsqrt_64x8
#[no_mangle]
pub unsafe fn fsqrt_64x8(a: f64x8) -> f64x8 {
    // CHECK: call fast <8 x double> @llvm.sqrt.v8f64
    simd_fsqrt(a)
}
