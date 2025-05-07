//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

use std::intrinsics::simd::simd_masked_store;

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vec2<T>(pub [T; 2]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vec4<T>(pub [T; 4]);

// CHECK-LABEL: @store_f32x2
#[no_mangle]
pub unsafe fn store_f32x2(mask: Vec2<i32>, pointer: *mut f32, values: Vec2<f32>) {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> {{.*}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // CHECK: call void @llvm.masked.store.v2f32.p0(<2 x float> {{.*}}, ptr {{.*}}, i32 4, <2 x i1> [[B]])
    simd_masked_store(mask, pointer, values)
}

// CHECK-LABEL: @store_f32x2_unsigned
#[no_mangle]
pub unsafe fn store_f32x2_unsigned(mask: Vec2<u32>, pointer: *mut f32, values: Vec2<f32>) {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> {{.*}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // CHECK: call void @llvm.masked.store.v2f32.p0(<2 x float> {{.*}}, ptr {{.*}}, i32 4, <2 x i1> [[B]])
    simd_masked_store(mask, pointer, values)
}

// CHECK-LABEL: @store_pf32x4
#[no_mangle]
pub unsafe fn store_pf32x4(mask: Vec4<i32>, pointer: *mut *const f32, values: Vec4<*const f32>) {
    // CHECK: [[A:%[0-9]+]] = lshr <4 x i32> {{.*}}, {{<i32 31, i32 31, i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <4 x i32> [[A]] to <4 x i1>
    // CHECK: call void @llvm.masked.store.v4p0.p0(<4 x ptr> {{.*}}, ptr {{.*}}, i32 {{.*}}, <4 x i1> [[B]])
    simd_masked_store(mask, pointer, values)
}
