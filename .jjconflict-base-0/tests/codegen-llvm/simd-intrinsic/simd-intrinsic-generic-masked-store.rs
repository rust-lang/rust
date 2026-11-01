//@ compile-flags: -C no-prepopulate-passes
//@ revisions: LLVM21 LLVM22
//@ [LLVM22] min-llvm-version: 22
//@ [LLVM21] max-llvm-major-version: 21
// ignore-tidy-linelength

#![crate_type = "lib"]
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::{SimdAlign, simd_masked_store};

pub type Vec2<T> = Simd<T, 2>;
pub type Vec4<T> = Simd<T, 4>;

// CHECK-LABEL: @store_f32x2
#[no_mangle]
pub unsafe fn store_f32x2(mask: Vec2<i32>, pointer: *mut f32, values: Vec2<f32>) {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> {{.*}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // LLVM21: call void @llvm.masked.store.v2f32.p0(<2 x float> {{.*}}, ptr {{.*}}, i32 4, <2 x i1> [[B]])
    //                                                                               ^^^^^
    // LLVM22: call void @llvm.masked.store.v2f32.p0(<2 x float> {{.*}}, ptr align 4 {{.*}}, <2 x i1> [[B]])
    //                                                                       ^^^^^^^
    // the align parameter should be equal to the alignment of the element type (assumed to be 4)
    simd_masked_store::<_, _, _, { SimdAlign::Element }>(mask, pointer, values)
}

// CHECK-LABEL: @store_f32x2_aligned
#[no_mangle]
pub unsafe fn store_f32x2_aligned(mask: Vec2<i32>, pointer: *mut f32, values: Vec2<f32>) {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> {{.*}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // LLVM21: call void @llvm.masked.store.v2f32.p0(<2 x float> {{.*}}, ptr {{.*}}, i32 8, <2 x i1> [[B]])
    //                                                                               ^^^^^
    // LLVM22: call void @llvm.masked.store.v2f32.p0(<2 x float> {{.*}}, ptr align 8 {{.*}}, <2 x i1> [[B]])
    //                                                                       ^^^^^^^
    // the align parameter should be equal to the size of the vector
    simd_masked_store::<_, _, _, { SimdAlign::Vector }>(mask, pointer, values)
}

// CHECK-LABEL: @store_f32x2_unaligned
#[no_mangle]
pub unsafe fn store_f32x2_unaligned(mask: Vec2<i32>, pointer: *mut f32, values: Vec2<f32>) {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> {{.*}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // LLVM21: call void @llvm.masked.store.v2f32.p0(<2 x float> {{.*}}, ptr {{.*}}, i32 1, <2 x i1> [[B]])
    //                                                                               ^^^^^
    // LLVM22: call void @llvm.masked.store.v2f32.p0(<2 x float> {{.*}}, ptr align 1 {{.*}}, <2 x i1> [[B]])
    //                                                                       ^^^^^^^
    // the align parameter should be 1
    simd_masked_store::<_, _, _, { SimdAlign::Unaligned }>(mask, pointer, values)
}

// CHECK-LABEL: @store_f32x2_unsigned
#[no_mangle]
pub unsafe fn store_f32x2_unsigned(mask: Vec2<u32>, pointer: *mut f32, values: Vec2<f32>) {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> {{.*}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // LLVM21: call void @llvm.masked.store.v2f32.p0(<2 x float> {{.*}}, ptr {{.*}}, i32 4, <2 x i1> [[B]])
    // LLVM22: call void @llvm.masked.store.v2f32.p0(<2 x float> {{.*}}, ptr align 4 {{.*}}, <2 x i1> [[B]])
    simd_masked_store::<_, _, _, { SimdAlign::Element }>(mask, pointer, values)
}

// CHECK-LABEL: @store_pf32x4
#[no_mangle]
pub unsafe fn store_pf32x4(mask: Vec4<i32>, pointer: *mut *const f32, values: Vec4<*const f32>) {
    // CHECK: [[A:%[0-9]+]] = lshr <4 x i32> {{.*}}, {{<i32 31, i32 31, i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <4 x i32> [[A]] to <4 x i1>
    // LLVM21: call void @llvm.masked.store.v4p0.p0(<4 x ptr> {{.*}}, ptr {{.*}}, i32 {{.*}}, <4 x i1> [[B]])
    // LLVM22: call void @llvm.masked.store.v4p0.p0(<4 x ptr> {{.*}}, ptr align {{.*}} {{.*}}, <4 x i1> [[B]])
    simd_masked_store::<_, _, _, { SimdAlign::Element }>(mask, pointer, values)
}
