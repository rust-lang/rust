//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::{simd_reduce_all, simd_reduce_any};

pub type mask32x2 = Simd<i32, 2>;
pub type mask8x16 = Simd<i8, 16>;

// NOTE(eddyb) `%{{x|1}}` is used because on some targets (e.g. WASM)
// SIMD vectors are passed directly, resulting in `%x` being a vector,
// while on others they're passed indirectly, resulting in `%x` being
// a pointer to a vector, and `%1` a vector loaded from that pointer.
// This is controlled by the target spec option `simd_types_indirect`.

// CHECK-LABEL: @reduce_any_32x2
#[no_mangle]
pub unsafe fn reduce_any_32x2(x: mask32x2) -> bool {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> %{{x|1}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // CHECK: [[C:%[0-9]+]] = call i1 @llvm.vector.reduce.or.v2i1(<2 x i1> [[B]])
    // CHECK: %{{[0-9]+}} = zext i1 [[C]] to i8
    simd_reduce_any(x)
}

// CHECK-LABEL: @reduce_all_32x2
#[no_mangle]
pub unsafe fn reduce_all_32x2(x: mask32x2) -> bool {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> %{{x|1}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // CHECK: [[C:%[0-9]+]] = call i1 @llvm.vector.reduce.and.v2i1(<2 x i1> [[B]])
    // CHECK: %{{[0-9]+}} = zext i1 [[C]] to i8
    simd_reduce_all(x)
}

// CHECK-LABEL: @reduce_any_8x16
#[no_mangle]
pub unsafe fn reduce_any_8x16(x: mask8x16) -> bool {
    // CHECK: [[A:%[0-9]+]] = lshr <16 x i8> %{{x|1}}, {{<i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>|splat \(i8 7\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <16 x i8> [[A]] to <16 x i1>
    // CHECK: [[C:%[0-9]+]] = call i1 @llvm.vector.reduce.or.v16i1(<16 x i1> [[B]])
    // CHECK: %{{[0-9]+}} = zext i1 [[C]] to i8
    simd_reduce_any(x)
}

// CHECK-LABEL: @reduce_all_8x16
#[no_mangle]
pub unsafe fn reduce_all_8x16(x: mask8x16) -> bool {
    // CHECK: [[A:%[0-9]+]] = lshr <16 x i8> %{{x|1}}, {{<i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>|splat \(i8 7\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <16 x i8> [[A]] to <16 x i1>
    // CHECK: [[C:%[0-9]+]] = call i1 @llvm.vector.reduce.and.v16i1(<16 x i1> [[B]])
    // CHECK: %{{[0-9]+}} = zext i1 [[C]] to i8
    simd_reduce_all(x)
}
