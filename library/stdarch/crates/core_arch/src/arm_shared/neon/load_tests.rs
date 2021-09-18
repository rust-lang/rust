//! Tests for ARM+v7+neon load (vld1) intrinsics.
//!
//! These are included in `{arm, aarch64}::neon`.

use super::*;

#[cfg(target_arch = "arm")]
use crate::core_arch::arm::*;

#[cfg(target_arch = "aarch64")]
use crate::core_arch::aarch64::*;

use crate::core_arch::simd::*;
use std::mem;
use stdarch_test::simd_test;
#[simd_test(enable = "neon")]
unsafe fn test_vld1_s8() {
    let a: [i8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let e = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
    let r: i8x8 = transmute(vld1_s8(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1q_s8() {
    let a: [i8; 17] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let e = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    let r: i8x16 = transmute(vld1q_s8(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1_s16() {
    let a: [i16; 5] = [0, 1, 2, 3, 4];
    let e = i16x4::new(1, 2, 3, 4);
    let r: i16x4 = transmute(vld1_s16(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1q_s16() {
    let a: [i16; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let e = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
    let r: i16x8 = transmute(vld1q_s16(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1_s32() {
    let a: [i32; 3] = [0, 1, 2];
    let e = i32x2::new(1, 2);
    let r: i32x2 = transmute(vld1_s32(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1q_s32() {
    let a: [i32; 5] = [0, 1, 2, 3, 4];
    let e = i32x4::new(1, 2, 3, 4);
    let r: i32x4 = transmute(vld1q_s32(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1_s64() {
    let a: [i64; 2] = [0, 1];
    let e = i64x1::new(1);
    let r: i64x1 = transmute(vld1_s64(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1q_s64() {
    let a: [i64; 3] = [0, 1, 2];
    let e = i64x2::new(1, 2);
    let r: i64x2 = transmute(vld1q_s64(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1_u8() {
    let a: [u8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let e = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
    let r: u8x8 = transmute(vld1_u8(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1q_u8() {
    let a: [u8; 17] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let e = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    let r: u8x16 = transmute(vld1q_u8(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1_u16() {
    let a: [u16; 5] = [0, 1, 2, 3, 4];
    let e = u16x4::new(1, 2, 3, 4);
    let r: u16x4 = transmute(vld1_u16(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1q_u16() {
    let a: [u16; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let e = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
    let r: u16x8 = transmute(vld1q_u16(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1_u32() {
    let a: [u32; 3] = [0, 1, 2];
    let e = u32x2::new(1, 2);
    let r: u32x2 = transmute(vld1_u32(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1q_u32() {
    let a: [u32; 5] = [0, 1, 2, 3, 4];
    let e = u32x4::new(1, 2, 3, 4);
    let r: u32x4 = transmute(vld1q_u32(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1_u64() {
    let a: [u64; 2] = [0, 1];
    let e = u64x1::new(1);
    let r: u64x1 = transmute(vld1_u64(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1q_u64() {
    let a: [u64; 3] = [0, 1, 2];
    let e = u64x2::new(1, 2);
    let r: u64x2 = transmute(vld1q_u64(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1_p8() {
    let a: [p8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let e = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
    let r: u8x8 = transmute(vld1_p8(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1q_p8() {
    let a: [p8; 17] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let e = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    let r: u8x16 = transmute(vld1q_p8(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1_p16() {
    let a: [p16; 5] = [0, 1, 2, 3, 4];
    let e = u16x4::new(1, 2, 3, 4);
    let r: u16x4 = transmute(vld1_p16(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1q_p16() {
    let a: [p16; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let e = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
    let r: u16x8 = transmute(vld1q_p16(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon,aes")]
unsafe fn test_vld1_p64() {
    let a: [p64; 2] = [0, 1];
    let e = u64x1::new(1);
    let r: u64x1 = transmute(vld1_p64(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon,aes")]
unsafe fn test_vld1q_p64() {
    let a: [p64; 3] = [0, 1, 2];
    let e = u64x2::new(1, 2);
    let r: u64x2 = transmute(vld1q_p64(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1_f32() {
    let a: [f32; 3] = [0., 1., 2.];
    let e = f32x2::new(1., 2.);
    let r: f32x2 = transmute(vld1_f32(a[1..].as_ptr()));
    assert_eq!(r, e)
}

#[simd_test(enable = "neon")]
unsafe fn test_vld1q_f32() {
    let a: [f32; 5] = [0., 1., 2., 3., 4.];
    let e = f32x4::new(1., 2., 3., 4.);
    let r: f32x4 = transmute(vld1q_f32(a[1..].as_ptr()));
    assert_eq!(r, e)
}
