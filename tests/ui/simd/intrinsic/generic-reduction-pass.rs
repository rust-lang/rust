//@ run-pass
#![allow(non_camel_case_types)]
//@ ignore-emscripten

// Test that the simd_reduce_{op} intrinsics produce the correct results.
#![feature(repr_simd, core_intrinsics)]

use std::intrinsics::simd::*;

#[repr(simd)]
#[derive(Copy, Clone)]
struct i32x4(pub [i32; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u32x4(pub [u32; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct f32x4(pub [f32; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct b8x4(pub [i8; 4]);

fn main() {
    unsafe {
        let x = i32x4([1, -2, 3, 4]);
        let r: i32 = simd_reduce_add_unordered(x);
        assert_eq!(r, 6_i32);
        let r: i32 = simd_reduce_mul_unordered(x);
        assert_eq!(r, -24_i32);
        let r: i32 = simd_reduce_add_ordered(x, -1);
        assert_eq!(r, 5_i32);
        let r: i32 = simd_reduce_mul_ordered(x, -1);
        assert_eq!(r, 24_i32);

        let r: i32 = simd_reduce_min(x);
        assert_eq!(r, -2_i32);
        let r: i32 = simd_reduce_max(x);
        assert_eq!(r, 4_i32);

        let x = i32x4([-1, -1, -1, -1]);
        let r: i32 = simd_reduce_and(x);
        assert_eq!(r, -1_i32);
        let r: i32 = simd_reduce_or(x);
        assert_eq!(r, -1_i32);
        let r: i32 = simd_reduce_xor(x);
        assert_eq!(r, 0_i32);

        let x = i32x4([-1, -1, 0, -1]);
        let r: i32 = simd_reduce_and(x);
        assert_eq!(r, 0_i32);
        let r: i32 = simd_reduce_or(x);
        assert_eq!(r, -1_i32);
        let r: i32 = simd_reduce_xor(x);
        assert_eq!(r, -1_i32);
    }

    unsafe {
        let x = u32x4([1, 2, 3, 4]);
        let r: u32 = simd_reduce_add_unordered(x);
        assert_eq!(r, 10_u32);
        let r: u32 = simd_reduce_mul_unordered(x);
        assert_eq!(r, 24_u32);
        let r: u32 = simd_reduce_add_ordered(x, 1);
        assert_eq!(r, 11_u32);
        let r: u32 = simd_reduce_mul_ordered(x, 2);
        assert_eq!(r, 48_u32);

        let r: u32 = simd_reduce_min(x);
        assert_eq!(r, 1_u32);
        let r: u32 = simd_reduce_max(x);
        assert_eq!(r, 4_u32);

        let t = u32::MAX;
        let x = u32x4([t, t, t, t]);
        let r: u32 = simd_reduce_and(x);
        assert_eq!(r, t);
        let r: u32 = simd_reduce_or(x);
        assert_eq!(r, t);
        let r: u32 = simd_reduce_xor(x);
        assert_eq!(r, 0_u32);

        let x = u32x4([t, t, 0, t]);
        let r: u32 = simd_reduce_and(x);
        assert_eq!(r, 0_u32);
        let r: u32 = simd_reduce_or(x);
        assert_eq!(r, t);
        let r: u32 = simd_reduce_xor(x);
        assert_eq!(r, t);
    }

    unsafe {
        let x = f32x4([1., -2., 3., 4.]);
        let r: f32 = simd_reduce_add_unordered(x);
        assert_eq!(r, 6_f32);
        let r: f32 = simd_reduce_mul_unordered(x);
        assert_eq!(r, -24_f32);
        let r: f32 = simd_reduce_add_ordered(x, 0.);
        assert_eq!(r, 6_f32);
        let r: f32 = simd_reduce_mul_ordered(x, 1.);
        assert_eq!(r, -24_f32);
        let r: f32 = simd_reduce_add_ordered(x, 1.);
        assert_eq!(r, 7_f32);
        let r: f32 = simd_reduce_mul_ordered(x, 2.);
        assert_eq!(r, -48_f32);

        let r: f32 = simd_reduce_min(x);
        assert_eq!(r, -2_f32);
        let r: f32 = simd_reduce_max(x);
        assert_eq!(r, 4_f32);
    }

    unsafe {
        let x = b8x4([!0, !0, !0, !0]);
        let r: bool = simd_reduce_all(x);
        assert_eq!(r, true);
        let r: bool = simd_reduce_any(x);
        assert_eq!(r, true);

        let x = b8x4([!0, !0, 0, !0]);
        let r: bool = simd_reduce_all(x);
        assert_eq!(r, false);
        let r: bool = simd_reduce_any(x);
        assert_eq!(r, true);

        let x = b8x4([0, 0, 0, 0]);
        let r: bool = simd_reduce_all(x);
        assert_eq!(r, false);
        let r: bool = simd_reduce_any(x);
        assert_eq!(r, false);
    }
}
