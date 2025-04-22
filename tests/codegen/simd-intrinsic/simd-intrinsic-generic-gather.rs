//

//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

use std::intrinsics::simd::simd_gather;

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vec2<T>(pub [T; 2]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vec4<T>(pub [T; 4]);

// CHECK-LABEL: @gather_f32x2
#[no_mangle]
pub unsafe fn gather_f32x2(
    pointers: Vec2<*const f32>,
    mask: Vec2<i32>,
    values: Vec2<f32>,
) -> Vec2<f32> {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> {{.*}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // CHECK: call <2 x float> @llvm.masked.gather.v2f32.v2p0(<2 x ptr> {{.*}}, i32 {{.*}}, <2 x i1> [[B]], <2 x float> {{.*}})
    simd_gather(values, pointers, mask)
}

// CHECK-LABEL: @gather_f32x2_unsigned
#[no_mangle]
pub unsafe fn gather_f32x2_unsigned(
    pointers: Vec2<*const f32>,
    mask: Vec2<u32>,
    values: Vec2<f32>,
) -> Vec2<f32> {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> {{.*}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // CHECK: call <2 x float> @llvm.masked.gather.v2f32.v2p0(<2 x ptr> {{.*}}, i32 {{.*}}, <2 x i1> [[B]], <2 x float> {{.*}})
    simd_gather(values, pointers, mask)
}

// CHECK-LABEL: @gather_pf32x2
#[no_mangle]
pub unsafe fn gather_pf32x2(
    pointers: Vec2<*const *const f32>,
    mask: Vec2<i32>,
    values: Vec2<*const f32>,
) -> Vec2<*const f32> {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> {{.*}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // CHECK: call <2 x ptr> @llvm.masked.gather.v2p0.v2p0(<2 x ptr> {{.*}}, i32 {{.*}}, <2 x i1> [[B]], <2 x ptr> {{.*}})
    simd_gather(values, pointers, mask)
}
