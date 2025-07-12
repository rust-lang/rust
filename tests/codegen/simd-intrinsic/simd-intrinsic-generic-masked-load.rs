//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_masked_load;

pub type Vec2<T> = Simd<T, 2>;
pub type Vec4<T> = Simd<T, 4>;

// CHECK-LABEL: @load_f32x2
#[no_mangle]
pub unsafe fn load_f32x2(mask: Vec2<i32>, pointer: *const f32, values: Vec2<f32>) -> Vec2<f32> {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> {{.*}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // CHECK: call <2 x float> @llvm.masked.load.v2f32.p0(ptr {{.*}}, i32 4, <2 x i1> [[B]], <2 x float> {{.*}})
    simd_masked_load(mask, pointer, values)
}

// CHECK-LABEL: @load_f32x2_unsigned
#[no_mangle]
pub unsafe fn load_f32x2_unsigned(
    mask: Vec2<u32>,
    pointer: *const f32,
    values: Vec2<f32>,
) -> Vec2<f32> {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> {{.*}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // CHECK: call <2 x float> @llvm.masked.load.v2f32.p0(ptr {{.*}}, i32 4, <2 x i1> [[B]], <2 x float> {{.*}})
    simd_masked_load(mask, pointer, values)
}

// CHECK-LABEL: @load_pf32x4
#[no_mangle]
pub unsafe fn load_pf32x4(
    mask: Vec4<i32>,
    pointer: *const *const f32,
    values: Vec4<*const f32>,
) -> Vec4<*const f32> {
    // CHECK: [[A:%[0-9]+]] = lshr <4 x i32> {{.*}}, {{<i32 31, i32 31, i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <4 x i32> [[A]] to <4 x i1>
    // CHECK: call <4 x ptr> @llvm.masked.load.v4p0.p0(ptr {{.*}}, i32 {{.*}}, <4 x i1> [[B]], <4 x ptr> {{.*}})
    simd_masked_load(mask, pointer, values)
}
