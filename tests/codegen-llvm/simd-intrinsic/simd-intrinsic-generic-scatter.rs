//

//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_scatter;

pub type Vec2<T> = Simd<T, 2>;
pub type Vec4<T> = Simd<T, 4>;

// CHECK-LABEL: @scatter_f32x2
#[no_mangle]
pub unsafe fn scatter_f32x2(pointers: Vec2<*mut f32>, mask: Vec2<i32>, values: Vec2<f32>) {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> {{.*}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // CHECK: call void @llvm.masked.scatter.v2f32.v2p0(<2 x float> {{.*}}, <2 x ptr> {{.*}}, i32 {{.*}}, <2 x i1> [[B]]
    simd_scatter(values, pointers, mask)
}

// CHECK-LABEL: @scatter_f32x2_unsigned
#[no_mangle]
pub unsafe fn scatter_f32x2_unsigned(pointers: Vec2<*mut f32>, mask: Vec2<u32>, values: Vec2<f32>) {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> {{.*}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // CHECK: call void @llvm.masked.scatter.v2f32.v2p0(<2 x float> {{.*}}, <2 x ptr> {{.*}}, i32 {{.*}}, <2 x i1> [[B]]
    simd_scatter(values, pointers, mask)
}

// CHECK-LABEL: @scatter_pf32x2
#[no_mangle]
pub unsafe fn scatter_pf32x2(
    pointers: Vec2<*mut *const f32>,
    mask: Vec2<i32>,
    values: Vec2<*const f32>,
) {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> {{.*}}, {{<i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // CHECK: call void @llvm.masked.scatter.v2p0.v2p0(<2 x ptr> {{.*}}, <2 x ptr> {{.*}}, i32 {{.*}}, <2 x i1> [[B]]
    simd_scatter(values, pointers, mask)
}
