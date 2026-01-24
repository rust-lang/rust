#![crate_type = "lib"]
#![no_std]
#![feature(repr_simd, core_intrinsics)]
use core::intrinsics::simd::simd_splat;

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

// Test that `simd_splat` produces the canonical LLVM splat sequence.

#[no_mangle]
unsafe fn int(x: u16) -> u16x2 {
    // CHECK-LABEL: int
    // CHECK: start:
    // CHECK-NEXT: %0 = insertelement <2 x i16> poison, i16 %x, i64 0
    // CHECK-NEXT: %1 = shufflevector <2 x i16> %0, <2 x i16> poison, <2 x i32> zeroinitializer
    // CHECK-NEXT: store
    // CHECK-NEXT: ret
    simd_splat(x)
}

#[no_mangle]
unsafe fn float(x: f32) -> f32x4 {
    // CHECK-LABEL: float
    // CHECK: start:
    // CHECK-NEXT: %0 = insertelement <4 x float> poison, float %x, i64 0
    // CHECK-NEXT: %1 = shufflevector <4 x float> %0, <4 x float> poison, <4 x i32> zeroinitializer
    // CHECK-NEXT: store
    // CHECK-NEXT: ret
    simd_splat(x)
}
