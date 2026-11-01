//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::{simd_select, simd_select_bitmask};

pub type b8x4 = i8x4;

// CHECK-LABEL: @select_m8
#[no_mangle]
pub unsafe fn select_m8(m: b8x4, a: f32x4, b: f32x4) -> f32x4 {
    // CHECK: [[A:%[0-9]+]] = lshr <4 x i8> %{{.*}}, {{<i8 7, i8 7, i8 7, i8 7>|splat \(i8 7\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <4 x i8> [[A]] to <4 x i1>
    // CHECK: select <4 x i1> [[B]]
    simd_select(m, a, b)
}

// CHECK-LABEL: @select_m32
#[no_mangle]
pub unsafe fn select_m32(m: i32x4, a: f32x4, b: f32x4) -> f32x4 {
    // CHECK: [[A:%[0-9]+]] = lshr <4 x i32> %{{.*}}, {{<i32 31, i32 31, i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <4 x i32> [[A]] to <4 x i1>
    // CHECK: select <4 x i1> [[B]]
    simd_select(m, a, b)
}

// CHECK-LABEL: @select_m32_unsigned
#[no_mangle]
pub unsafe fn select_m32_unsigned(m: u32x4, a: f32x4, b: f32x4) -> f32x4 {
    // CHECK: [[A:%[0-9]+]] = lshr <4 x i32> %{{.*}}, {{<i32 31, i32 31, i32 31, i32 31>|splat \(i32 31\)}}
    // CHECK: [[B:%[0-9]+]] = trunc <4 x i32> [[A]] to <4 x i1>
    // CHECK: select <4 x i1> [[B]]
    simd_select(m, a, b)
}

// CHECK-LABEL: @select_bitmask
#[no_mangle]
pub unsafe fn select_bitmask(m: i8, a: f32x8, b: f32x8) -> f32x8 {
    // CHECK: [[A:%[0-9]+]] = bitcast i8 {{.*}} to <8 x i1>
    // CHECK: select <8 x i1> [[A]]
    simd_select_bitmask(m, a, b)
}
