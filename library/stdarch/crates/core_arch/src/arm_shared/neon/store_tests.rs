//! Tests for ARM+v7+neon store (vst1) intrinsics.
//!
//! These are included in `{arm, aarch64}::neon`.

use super::*;

#[cfg(target_arch = "arm")]
use crate::core_arch::arm::*;

#[cfg(target_arch = "aarch64")]
use crate::core_arch::aarch64::*;

use crate::core_arch::simd::*;
use stdarch_test::simd_test;

#[simd_test(enable = "neon")]
unsafe fn test_vst1_s8() {
    let mut vals = [0_i8; 9];
    let a = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);

    vst1_s8(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
    assert_eq!(vals[5], 5);
    assert_eq!(vals[6], 6);
    assert_eq!(vals[7], 7);
    assert_eq!(vals[8], 8);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_s8() {
    let mut vals = [0_i8; 17];
    let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    vst1q_s8(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
    assert_eq!(vals[5], 5);
    assert_eq!(vals[6], 6);
    assert_eq!(vals[7], 7);
    assert_eq!(vals[8], 8);
    assert_eq!(vals[9], 9);
    assert_eq!(vals[10], 10);
    assert_eq!(vals[11], 11);
    assert_eq!(vals[12], 12);
    assert_eq!(vals[13], 13);
    assert_eq!(vals[14], 14);
    assert_eq!(vals[15], 15);
    assert_eq!(vals[16], 16);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_s16() {
    let mut vals = [0_i16; 5];
    let a = i16x4::new(1, 2, 3, 4);

    vst1_s16(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_s16() {
    let mut vals = [0_i16; 9];
    let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);

    vst1q_s16(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
    assert_eq!(vals[5], 5);
    assert_eq!(vals[6], 6);
    assert_eq!(vals[7], 7);
    assert_eq!(vals[8], 8);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_s32() {
    let mut vals = [0_i32; 3];
    let a = i32x2::new(1, 2);

    vst1_s32(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_s32() {
    let mut vals = [0_i32; 5];
    let a = i32x4::new(1, 2, 3, 4);

    vst1q_s32(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_s64() {
    let mut vals = [0_i64; 2];
    let a = i64x1::new(1);

    vst1_s64(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_s64() {
    let mut vals = [0_i64; 3];
    let a = i64x2::new(1, 2);

    vst1q_s64(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_u8() {
    let mut vals = [0_u8; 9];
    let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);

    vst1_u8(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
    assert_eq!(vals[5], 5);
    assert_eq!(vals[6], 6);
    assert_eq!(vals[7], 7);
    assert_eq!(vals[8], 8);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_u8() {
    let mut vals = [0_u8; 17];
    let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    vst1q_u8(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
    assert_eq!(vals[5], 5);
    assert_eq!(vals[6], 6);
    assert_eq!(vals[7], 7);
    assert_eq!(vals[8], 8);
    assert_eq!(vals[9], 9);
    assert_eq!(vals[10], 10);
    assert_eq!(vals[11], 11);
    assert_eq!(vals[12], 12);
    assert_eq!(vals[13], 13);
    assert_eq!(vals[14], 14);
    assert_eq!(vals[15], 15);
    assert_eq!(vals[16], 16);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_u16() {
    let mut vals = [0_u16; 5];
    let a = u16x4::new(1, 2, 3, 4);

    vst1_u16(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_u16() {
    let mut vals = [0_u16; 9];
    let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);

    vst1q_u16(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
    assert_eq!(vals[5], 5);
    assert_eq!(vals[6], 6);
    assert_eq!(vals[7], 7);
    assert_eq!(vals[8], 8);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_u32() {
    let mut vals = [0_u32; 3];
    let a = u32x2::new(1, 2);

    vst1_u32(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_u32() {
    let mut vals = [0_u32; 5];
    let a = u32x4::new(1, 2, 3, 4);

    vst1q_u32(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_u64() {
    let mut vals = [0_u64; 2];
    let a = u64x1::new(1);

    vst1_u64(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_u64() {
    let mut vals = [0_u64; 3];
    let a = u64x2::new(1, 2);

    vst1q_u64(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_p8() {
    let mut vals = [0_u8; 9];
    let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);

    vst1_p8(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
    assert_eq!(vals[5], 5);
    assert_eq!(vals[6], 6);
    assert_eq!(vals[7], 7);
    assert_eq!(vals[8], 8);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_p8() {
    let mut vals = [0_u8; 17];
    let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    vst1q_p8(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
    assert_eq!(vals[5], 5);
    assert_eq!(vals[6], 6);
    assert_eq!(vals[7], 7);
    assert_eq!(vals[8], 8);
    assert_eq!(vals[9], 9);
    assert_eq!(vals[10], 10);
    assert_eq!(vals[11], 11);
    assert_eq!(vals[12], 12);
    assert_eq!(vals[13], 13);
    assert_eq!(vals[14], 14);
    assert_eq!(vals[15], 15);
    assert_eq!(vals[16], 16);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_p16() {
    let mut vals = [0_u16; 5];
    let a = u16x4::new(1, 2, 3, 4);

    vst1_p16(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_p16() {
    let mut vals = [0_u16; 9];
    let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);

    vst1q_p16(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
    assert_eq!(vals[5], 5);
    assert_eq!(vals[6], 6);
    assert_eq!(vals[7], 7);
    assert_eq!(vals[8], 8);
}

#[simd_test(enable = "neon,aes")]
unsafe fn test_vst1_p64() {
    let mut vals = [0_u64; 2];
    let a = u64x1::new(1);

    vst1_p64(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
}

#[simd_test(enable = "neon,aes")]
unsafe fn test_vst1q_p64() {
    let mut vals = [0_u64; 3];
    let a = u64x2::new(1, 2);

    vst1q_p64(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0);
    assert_eq!(vals[1], 1);
    assert_eq!(vals[2], 2);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_f32() {
    let mut vals = [0_f32; 3];
    let a = f32x2::new(1., 2.);

    vst1_f32(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0.);
    assert_eq!(vals[1], 1.);
    assert_eq!(vals[2], 2.);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_f32() {
    let mut vals = [0_f32; 5];
    let a = f32x4::new(1., 2., 3., 4.);

    vst1q_f32(vals[1..].as_mut_ptr(), transmute(a));

    assert_eq!(vals[0], 0.);
    assert_eq!(vals[1], 1.);
    assert_eq!(vals[2], 2.);
    assert_eq!(vals[3], 3.);
    assert_eq!(vals[4], 4.);
}
