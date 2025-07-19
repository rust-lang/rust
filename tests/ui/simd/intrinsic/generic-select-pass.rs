//@ run-pass
#![allow(non_camel_case_types)]
//@ ignore-emscripten
//@ ignore-endian-big behavior of simd_select_bitmask is endian-specific

// Test that the simd_select intrinsics produces correct results.
#![feature(repr_simd, core_intrinsics)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::{simd_select, simd_select_bitmask};

type b8x4 = i8x4;

fn main() {
    let m0 = b8x4::from_array([!0, !0, !0, !0]);
    let m1 = b8x4::from_array([0, 0, 0, 0]);
    let m2 = b8x4::from_array([!0, !0, 0, 0]);
    let m3 = b8x4::from_array([0, 0, !0, !0]);
    let m4 = b8x4::from_array([!0, 0, !0, 0]);

    unsafe {
        let a = i32x4::from_array([1, -2, 3, 4]);
        let b = i32x4::from_array([5, 6, -7, 8]);

        let r: i32x4 = simd_select(m0, a, b);
        let e = a;
        assert_eq!(r, e);

        let r: i32x4 = simd_select(m1, a, b);
        let e = b;
        assert_eq!(r, e);

        let r: i32x4 = simd_select(m2, a, b);
        let e = i32x4::from_array([1, -2, -7, 8]);
        assert_eq!(r, e);

        let r: i32x4 = simd_select(m3, a, b);
        let e = i32x4::from_array([5, 6, 3, 4]);
        assert_eq!(r, e);

        let r: i32x4 = simd_select(m4, a, b);
        let e = i32x4::from_array([1, 6, 3, 8]);
        assert_eq!(r, e);
    }

    unsafe {
        let a = u32x4::from_array([1, 2, 3, 4]);
        let b = u32x4::from_array([5, 6, 7, 8]);

        let r: u32x4 = simd_select(m0, a, b);
        let e = a;
        assert_eq!(r, e);

        let r: u32x4 = simd_select(m1, a, b);
        let e = b;
        assert_eq!(r, e);

        let r: u32x4 = simd_select(m2, a, b);
        let e = u32x4::from_array([1, 2, 7, 8]);
        assert_eq!(r, e);

        let r: u32x4 = simd_select(m3, a, b);
        let e = u32x4::from_array([5, 6, 3, 4]);
        assert_eq!(r, e);

        let r: u32x4 = simd_select(m4, a, b);
        let e = u32x4::from_array([1, 6, 3, 8]);
        assert_eq!(r, e);
    }

    unsafe {
        let a = f32x4::from_array([1., 2., 3., 4.]);
        let b = f32x4::from_array([5., 6., 7., 8.]);

        let r: f32x4 = simd_select(m0, a, b);
        let e = a;
        assert_eq!(r, e);

        let r: f32x4 = simd_select(m1, a, b);
        let e = b;
        assert_eq!(r, e);

        let r: f32x4 = simd_select(m2, a, b);
        let e = f32x4::from_array([1., 2., 7., 8.]);
        assert_eq!(r, e);

        let r: f32x4 = simd_select(m3, a, b);
        let e = f32x4::from_array([5., 6., 3., 4.]);
        assert_eq!(r, e);

        let r: f32x4 = simd_select(m4, a, b);
        let e = f32x4::from_array([1., 6., 3., 8.]);
        assert_eq!(r, e);
    }

    unsafe {
        let t = !0 as i8;
        let f = 0 as i8;
        let a = b8x4::from_array([t, f, t, f]);
        let b = b8x4::from_array([f, f, f, t]);

        let r: b8x4 = simd_select(m0, a, b);
        let e = a;
        assert_eq!(r, e);

        let r: b8x4 = simd_select(m1, a, b);
        let e = b;
        assert_eq!(r, e);

        let r: b8x4 = simd_select(m2, a, b);
        let e = b8x4::from_array([t, f, f, t]);
        assert_eq!(r, e);

        let r: b8x4 = simd_select(m3, a, b);
        let e = b8x4::from_array([f, f, t, f]);
        assert_eq!(r, e);

        let r: b8x4 = simd_select(m4, a, b);
        let e = b8x4::from_array([t, f, t, t]);
        assert_eq!(r, e);
    }

    unsafe {
        let a = u32x8::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
        let b = u32x8::from_array([8, 9, 10, 11, 12, 13, 14, 15]);

        let r: u32x8 = simd_select_bitmask(0u8, a, b);
        let e = b;
        assert_eq!(r, e);

        let r: u32x8 = simd_select_bitmask(0xffu8, a, b);
        let e = a;
        assert_eq!(r, e);

        let r: u32x8 = simd_select_bitmask(0b01010101u8, a, b);
        let e = u32x8::from_array([0, 9, 2, 11, 4, 13, 6, 15]);
        assert_eq!(r, e);

        let r: u32x8 = simd_select_bitmask(0b10101010u8, a, b);
        let e = u32x8::from_array([8, 1, 10, 3, 12, 5, 14, 7]);
        assert_eq!(r, e);

        let r: u32x8 = simd_select_bitmask(0b11110000u8, a, b);
        let e = u32x8::from_array([8, 9, 10, 11, 4, 5, 6, 7]);
        assert_eq!(r, e);
    }

    unsafe {
        let a = u32x4::from_array([0, 1, 2, 3]);
        let b = u32x4::from_array([4, 5, 6, 7]);

        let r: u32x4 = simd_select_bitmask(0u8, a, b);
        let e = b;
        assert_eq!(r, e);

        let r: u32x4 = simd_select_bitmask(0xfu8, a, b);
        let e = a;
        assert_eq!(r, e);

        let r: u32x4 = simd_select_bitmask(0b0101u8, a, b);
        let e = u32x4::from_array([0, 5, 2, 7]);
        assert_eq!(r, e);

        let r: u32x4 = simd_select_bitmask(0b1010u8, a, b);
        let e = u32x4::from_array([4, 1, 6, 3]);
        assert_eq!(r, e);

        let r: u32x4 = simd_select_bitmask(0b1100u8, a, b);
        let e = u32x4::from_array([4, 5, 2, 3]);
        assert_eq!(r, e);
    }
}
