//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

#![feature(repr_simd, intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f32x4(pub [f32; 4]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct f32x8([f32; 8]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct b8x4(pub [i8; 4]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct i32x4([i32; 4]);

extern "rust-intrinsic" {
    fn simd_select<T, U>(x: T, a: U, b: U) -> U;
    fn simd_select_bitmask<T, U>(x: T, a: U, b: U) -> U;
}

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

// CHECK-LABEL: @select_bitmask
#[no_mangle]
pub unsafe fn select_bitmask(m: i8, a: f32x8, b: f32x8) -> f32x8 {
    // CHECK: [[A:%[0-9]+]] = bitcast i8 {{.*}} to <8 x i1>
    // CHECK: select <8 x i1> [[A]]
    simd_select_bitmask(m, a, b)
}
