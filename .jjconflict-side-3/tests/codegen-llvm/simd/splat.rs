//@ compile-flags: -Copt-level=3

// LLVM IR isn't very portable and the one tested here depends on the ABI which is different between
// x86 (where we use SSE registers) and others. `x86-64` and `x86-32-sse2` are identical, but
// compiletest does not support taking the union of multiple `only` annotations.
//@ revisions: x86-64 x86-32-sse2 by-ref
//@[x86-64] only-x86_64
//@[x86-64] filecheck-flags: --check-prefix=by-val
//@[x86-32-sse2] only-rustc_abi-x86-sse2
//@[x86-32-sse2] filecheck-flags: --check-prefix=by-val
//@[by-ref] ignore-rustc_abi-x86-sse2
//@[by-ref] ignore-x86_64

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
    // CHECK-NEXT: [[VREG:%[a-z0-9_]+]] = shufflevector <2 x i16> %0, <2 x i16> poison, <2 x i32> zeroinitializer
    // by-ref-NEXT: store <2 x i16> [[VREG]]
    // by-ref-NEXT: ret
    // by-val-NEXT: ret <2 x i16> [[VREG]]
    simd_splat(x)
}

#[no_mangle]
unsafe fn float(x: f32) -> f32x4 {
    // CHECK-LABEL: float
    // CHECK: start:
    // CHECK-NEXT: %0 = insertelement <4 x float> poison, float %x, i64 0
    // CHECK-NEXT: [[VREG:%[a-z0-9_]+]] = shufflevector <4 x float> %0, <4 x float> poison, <4 x i32> zeroinitializer
    // by-ref-NEXT: store <4 x float> [[VREG]]
    // by-ref-NEXT: ret
    // by-val-NEXT: ret <4 x float> [[VREG]]
    simd_splat(x)
}
