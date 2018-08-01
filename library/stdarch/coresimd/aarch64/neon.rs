//! ARMv8 ASIMD intrinsics

#![allow(non_camel_case_types)]

// FIXME: replace neon with asimd

use coresimd::arm::*;
use coresimd::simd_llvm::*;
#[cfg(test)]
use stdsimd_test::assert_instr;

types! {
    /// ARM-specific 64-bit wide vector of one packed `f64`.
    pub struct float64x1_t(f64); // FIXME: check this!
    /// ARM-specific 128-bit wide vector of two packed `f64`.
    pub struct float64x2_t(f64, f64);
    /// ARM-specific 64-bit wide vector of one packed `p64`.
    pub struct poly64x1_t(i64); // FIXME: check this!
    /// ARM-specific 64-bit wide vector of two packed `p64`.
    pub struct poly64x2_t(i64, i64); // FIXME: check this!
}

/// ARM-specific type containing two `int8x16_t` vectors.
#[derive(Copy, Clone)]
pub struct int8x16x2_t(pub int8x16_t, pub int8x16_t);
/// ARM-specific type containing three `int8x16_t` vectors.
#[derive(Copy, Clone)]
pub struct int8x16x3_t(pub int8x16_t, pub int8x16_t, pub int8x16_t);
/// ARM-specific type containing four `int8x16_t` vectors.
#[derive(Copy, Clone)]
pub struct int8x16x4_t(
    pub int8x16_t,
    pub int8x16_t,
    pub int8x16_t,
    pub int8x16_t,
);

/// ARM-specific type containing two `uint8x16_t` vectors.
#[derive(Copy, Clone)]
pub struct uint8x16x2_t(pub uint8x16_t, pub uint8x16_t);
/// ARM-specific type containing three `uint8x16_t` vectors.
#[derive(Copy, Clone)]
pub struct uint8x16x3_t(pub uint8x16_t, pub uint8x16_t, pub uint8x16_t);
/// ARM-specific type containing four `uint8x16_t` vectors.
#[derive(Copy, Clone)]
pub struct uint8x16x4_t(
    pub uint8x16_t,
    pub uint8x16_t,
    pub uint8x16_t,
    pub uint8x16_t,
);

/// ARM-specific type containing two `poly8x16_t` vectors.
#[derive(Copy, Clone)]
pub struct poly8x16x2_t(pub poly8x16_t, pub poly8x16_t);
/// ARM-specific type containing three `poly8x16_t` vectors.
#[derive(Copy, Clone)]
pub struct poly8x16x3_t(pub poly8x16_t, pub poly8x16_t, pub poly8x16_t);
/// ARM-specific type containing four `poly8x16_t` vectors.
#[derive(Copy, Clone)]
pub struct poly8x16x4_t(
    pub poly8x16_t,
    pub poly8x16_t,
    pub poly8x16_t,
    pub poly8x16_t,
);

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.aarch64.neon.smaxv.i8.v8i8"]
    fn vmaxv_s8_(a: int8x8_t) -> i8;
    #[link_name = "llvm.aarch64.neon.smaxv.i8.6i8"]
    fn vmaxvq_s8_(a: int8x16_t) -> i8;
    #[link_name = "llvm.aarch64.neon.smaxv.i16.v4i16"]
    fn vmaxv_s16_(a: int16x4_t) -> i16;
    #[link_name = "llvm.aarch64.neon.smaxv.i16.v8i16"]
    fn vmaxvq_s16_(a: int16x8_t) -> i16;
    #[link_name = "llvm.aarch64.neon.smaxv.i32.v2i32"]
    fn vmaxv_s32_(a: int32x2_t) -> i32;
    #[link_name = "llvm.aarch64.neon.smaxv.i32.v4i32"]
    fn vmaxvq_s32_(a: int32x4_t) -> i32;

    #[link_name = "llvm.aarch64.neon.umaxv.i8.v8i8"]
    fn vmaxv_u8_(a: uint8x8_t) -> u8;
    #[link_name = "llvm.aarch64.neon.umaxv.i8.6i8"]
    fn vmaxvq_u8_(a: uint8x16_t) -> u8;
    #[link_name = "llvm.aarch64.neon.umaxv.i16.v4i16"]
    fn vmaxv_u16_(a: uint16x4_t) -> u16;
    #[link_name = "llvm.aarch64.neon.umaxv.i16.v8i16"]
    fn vmaxvq_u16_(a: uint16x8_t) -> u16;
    #[link_name = "llvm.aarch64.neon.umaxv.i32.v2i32"]
    fn vmaxv_u32_(a: uint32x2_t) -> u32;
    #[link_name = "llvm.aarch64.neon.umaxv.i32.v4i32"]
    fn vmaxvq_u32_(a: uint32x4_t) -> u32;

    #[link_name = "llvm.aarch64.neon.fmaxv.f32.v2f32"]
    fn vmaxv_f32_(a: float32x2_t) -> f32;
    #[link_name = "llvm.aarch64.neon.fmaxv.f32.v4f32"]
    fn vmaxvq_f32_(a: float32x4_t) -> f32;
    #[link_name = "llvm.aarch64.neon.fmaxv.f64.v2f64"]
    fn vmaxvq_f64_(a: float64x2_t) -> f64;

    #[link_name = "llvm.aarch64.neon.sminv.i8.v8i8"]
    fn vminv_s8_(a: int8x8_t) -> i8;
    #[link_name = "llvm.aarch64.neon.sminv.i8.6i8"]
    fn vminvq_s8_(a: int8x16_t) -> i8;
    #[link_name = "llvm.aarch64.neon.sminv.i16.v4i16"]
    fn vminv_s16_(a: int16x4_t) -> i16;
    #[link_name = "llvm.aarch64.neon.sminv.i16.v8i16"]
    fn vminvq_s16_(a: int16x8_t) -> i16;
    #[link_name = "llvm.aarch64.neon.sminv.i32.v2i32"]
    fn vminv_s32_(a: int32x2_t) -> i32;
    #[link_name = "llvm.aarch64.neon.sminv.i32.v4i32"]
    fn vminvq_s32_(a: int32x4_t) -> i32;

    #[link_name = "llvm.aarch64.neon.uminv.i8.v8i8"]
    fn vminv_u8_(a: uint8x8_t) -> u8;
    #[link_name = "llvm.aarch64.neon.uminv.i8.6i8"]
    fn vminvq_u8_(a: uint8x16_t) -> u8;
    #[link_name = "llvm.aarch64.neon.uminv.i16.v4i16"]
    fn vminv_u16_(a: uint16x4_t) -> u16;
    #[link_name = "llvm.aarch64.neon.uminv.i16.v8i16"]
    fn vminvq_u16_(a: uint16x8_t) -> u16;
    #[link_name = "llvm.aarch64.neon.uminv.i32.v2i32"]
    fn vminv_u32_(a: uint32x2_t) -> u32;
    #[link_name = "llvm.aarch64.neon.uminv.i32.v4i32"]
    fn vminvq_u32_(a: uint32x4_t) -> u32;

    #[link_name = "llvm.aarch64.neon.fminv.f32.v2f32"]
    fn vminv_f32_(a: float32x2_t) -> f32;
    #[link_name = "llvm.aarch64.neon.fminv.f32.v4f32"]
    fn vminvq_f32_(a: float32x4_t) -> f32;
    #[link_name = "llvm.aarch64.neon.fminv.f64.v2f64"]
    fn vminvq_f64_(a: float64x2_t) -> f64;

    #[link_name = "llvm.aarch64.neon.sminp.v16i8"]
    fn vpminq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    #[link_name = "llvm.aarch64.neon.sminp.v8i16"]
    fn vpminq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    #[link_name = "llvm.aarch64.neon.sminp.v4i32"]
    fn vpminq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    #[link_name = "llvm.aarch64.neon.uminp.v16i8"]
    fn vpminq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    #[link_name = "llvm.aarch64.neon.uminp.v8i16"]
    fn vpminq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    #[link_name = "llvm.aarch64.neon.uminp.v4i32"]
    fn vpminq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    #[link_name = "llvm.aarch64.neon.fminp.4f32"]
    fn vpminq_f32_(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    #[link_name = "llvm.aarch64.neon.fminp.v2f64"]
    fn vpminq_f64_(a: float64x2_t, b: float64x2_t) -> float64x2_t;

    #[link_name = "llvm.aarch64.neon.smaxp.v16i8"]
    fn vpmaxq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    #[link_name = "llvm.aarch64.neon.smaxp.v8i16"]
    fn vpmaxq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    #[link_name = "llvm.aarch64.neon.smaxp.v4i32"]
    fn vpmaxq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    #[link_name = "llvm.aarch64.neon.umaxp.v16i8"]
    fn vpmaxq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    #[link_name = "llvm.aarch64.neon.umaxp.v8i16"]
    fn vpmaxq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    #[link_name = "llvm.aarch64.neon.umaxp.v4i32"]
    fn vpmaxq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    #[link_name = "llvm.aarch64.neon.fmaxp.4f32"]
    fn vpmaxq_f32_(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    #[link_name = "llvm.aarch64.neon.fmaxp.v2f64"]
    fn vpmaxq_f64_(a: float64x2_t, b: float64x2_t) -> float64x2_t;

    #[link_name = "llvm.aarch64.neon.tbl1.v8i8"]
    fn vqtbl1(a: int8x16_t, b: uint8x8_t) -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.tbl1.v16i8"]
    fn vqtbl1q(a: int8x16_t, b: uint8x16_t) -> int8x16_t;

    #[link_name = "llvm.aarch64.neon.tbx1.v8i8"]
    fn vqtbx1(a: int8x8_t, b: int8x16_t, c: uint8x8_t) -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.tbx1.v16i8"]
    fn vqtbx1q(a: int8x16_t, b: int8x16_t, c: uint8x16_t) -> int8x16_t;

    #[link_name = "llvm.aarch64.neon.tbl2.v8i8"]
    fn vqtbl2(a0: int8x16_t, a1: int8x16_t, b: uint8x8_t) -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.tbl2.v16i8"]
    fn vqtbl2q(a0: int8x16_t, a1: int8x16_t, b: uint8x16_t) -> int8x16_t;

    #[link_name = "llvm.aarch64.neon.tbx2.v8i8"]
    fn vqtbx2(
        a: int8x8_t, b0: int8x16_t, b1: int8x16_t, c: uint8x8_t,
    ) -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.tbx2.v16i8"]
    fn vqtbx2q(
        a: int8x16_t, b0: int8x16_t, b1: int8x16_t, c: uint8x16_t,
    ) -> int8x16_t;

    #[link_name = "llvm.aarch64.neon.tbl3.v8i8"]
    fn vqtbl3(
        a0: int8x16_t, a1: int8x16_t, a2: int8x16_t, b: uint8x8_t,
    ) -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.tbl3.v16i8"]
    fn vqtbl3q(
        a0: int8x16_t, a1: int8x16_t, a2: int8x16_t, b: uint8x16_t,
    ) -> int8x16_t;

    #[link_name = "llvm.aarch64.neon.tbx3.v8i8"]
    fn vqtbx3(
        a: int8x8_t, b0: int8x16_t, b1: int8x16_t, b2: int8x16_t, c: uint8x8_t,
    ) -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.tbx3.v16i8"]
    fn vqtbx3q(
        a: int8x16_t, b0: int8x16_t, b1: int8x16_t, b2: int8x16_t,
        c: uint8x16_t,
    ) -> int8x16_t;

    #[link_name = "llvm.aarch64.neon.tbl4.v8i8"]
    fn vqtbl4(
        a0: int8x16_t, a1: int8x16_t, a2: int8x16_t, a3: int8x16_t,
        b: uint8x8_t,
    ) -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.tbl4.v16i8"]
    fn vqtbl4q(
        a0: int8x16_t, a1: int8x16_t, a2: int8x16_t, a3: int8x16_t,
        b: uint8x16_t,
    ) -> int8x16_t;

    #[link_name = "llvm.aarch64.neon.tbx4.v8i8"]
    fn vqtbx4(
        a: int8x8_t, b0: int8x16_t, b1: int8x16_t, b2: int8x16_t,
        b3: int8x16_t, c: uint8x8_t,
    ) -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.tbx4.v16i8"]
    fn vqtbx4q(
        a: int8x16_t, b0: int8x16_t, b1: int8x16_t, b2: int8x16_t,
        b3: int8x16_t, c: uint8x16_t,
    ) -> int8x16_t;
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fadd))]
pub unsafe fn vadd_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fadd))]
pub unsafe fn vaddq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddd_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddd_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_add(a, b)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxv))]
pub unsafe fn vmaxv_s8(a: int8x8_t) -> i8 {
    vmaxv_s8_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxv))]
pub unsafe fn vmaxvq_s8(a: int8x16_t) -> i8 {
    vmaxvq_s8_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxv))]
pub unsafe fn vmaxv_s16(a: int16x4_t) -> i16 {
    vmaxv_s16_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxv))]
pub unsafe fn vmaxvq_s16(a: int16x8_t) -> i16 {
    vmaxvq_s16_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxp))]
pub unsafe fn vmaxv_s32(a: int32x2_t) -> i32 {
    vmaxv_s32_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxv))]
pub unsafe fn vmaxvq_s32(a: int32x4_t) -> i32 {
    vmaxvq_s32_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxv))]
pub unsafe fn vmaxv_u8(a: uint8x8_t) -> u8 {
    vmaxv_u8_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxv))]
pub unsafe fn vmaxvq_u8(a: uint8x16_t) -> u8 {
    vmaxvq_u8_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxv))]
pub unsafe fn vmaxv_u16(a: uint16x4_t) -> u16 {
    vmaxv_u16_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxv))]
pub unsafe fn vmaxvq_u16(a: uint16x8_t) -> u16 {
    vmaxvq_u16_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxp))]
pub unsafe fn vmaxv_u32(a: uint32x2_t) -> u32 {
    vmaxv_u32_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxv))]
pub unsafe fn vmaxvq_u32(a: uint32x4_t) -> u32 {
    vmaxvq_u32_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxp))]
pub unsafe fn vmaxv_f32(a: float32x2_t) -> f32 {
    vmaxv_f32_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxv))]
pub unsafe fn vmaxvq_f32(a: float32x4_t) -> f32 {
    vmaxvq_f32_(a)
}

/// Horizontal vector max.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxp))]
pub unsafe fn vmaxvq_f64(a: float64x2_t) -> f64 {
    vmaxvq_f64_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminv))]
pub unsafe fn vminv_s8(a: int8x8_t) -> i8 {
    vminv_s8_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminv))]
pub unsafe fn vminvq_s8(a: int8x16_t) -> i8 {
    vminvq_s8_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminv))]
pub unsafe fn vminv_s16(a: int16x4_t) -> i16 {
    vminv_s16_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminv))]
pub unsafe fn vminvq_s16(a: int16x8_t) -> i16 {
    vminvq_s16_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminp))]
pub unsafe fn vminv_s32(a: int32x2_t) -> i32 {
    vminv_s32_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminv))]
pub unsafe fn vminvq_s32(a: int32x4_t) -> i32 {
    vminvq_s32_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminv))]
pub unsafe fn vminv_u8(a: uint8x8_t) -> u8 {
    vminv_u8_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminv))]
pub unsafe fn vminvq_u8(a: uint8x16_t) -> u8 {
    vminvq_u8_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminv))]
pub unsafe fn vminv_u16(a: uint16x4_t) -> u16 {
    vminv_u16_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminv))]
pub unsafe fn vminvq_u16(a: uint16x8_t) -> u16 {
    vminvq_u16_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminp))]
pub unsafe fn vminv_u32(a: uint32x2_t) -> u32 {
    vminv_u32_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminv))]
pub unsafe fn vminvq_u32(a: uint32x4_t) -> u32 {
    vminvq_u32_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminp))]
pub unsafe fn vminv_f32(a: float32x2_t) -> f32 {
    vminv_f32_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminv))]
pub unsafe fn vminvq_f32(a: float32x4_t) -> f32 {
    vminvq_f32_(a)
}

/// Horizontal vector min.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminp))]
pub unsafe fn vminvq_f64(a: float64x2_t) -> f64 {
    vminvq_f64_(a)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminp))]
pub unsafe fn vpminq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vpminq_s8_(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminp))]
pub unsafe fn vpminq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vpminq_s16_(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sminp))]
pub unsafe fn vpminq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vpminq_s32_(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminp))]
pub unsafe fn vpminq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vpminq_u8_(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminp))]
pub unsafe fn vpminq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vpminq_u16_(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uminp))]
pub unsafe fn vpminq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vpminq_u32_(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminp))]
pub unsafe fn vpminq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vpminq_f32_(a, b)
}

/// Folding minimum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminp))]
pub unsafe fn vpminq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    vpminq_f64_(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxp))]
pub unsafe fn vpmaxq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vpmaxq_s8_(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxp))]
pub unsafe fn vpmaxq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vpmaxq_s16_(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smaxp))]
pub unsafe fn vpmaxq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vpmaxq_s32_(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxp))]
pub unsafe fn vpmaxq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vpmaxq_u8_(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxp))]
pub unsafe fn vpmaxq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vpmaxq_u16_(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umaxp))]
pub unsafe fn vpmaxq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vpmaxq_u32_(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxp))]
pub unsafe fn vpmaxq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vpmaxq_f32_(a, b)
}

/// Folding maximum of adjacent pairs
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxp))]
pub unsafe fn vpmaxq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    vpmaxq_f64_(a, b)
}

/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
pub unsafe fn vcombine_s8(low: int8x8_t, high: int8x8_t) -> int8x16_t {
    simd_shuffle16(
        low,
        high,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    )
}

/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
pub unsafe fn vcombine_s16(low: int16x4_t, high: int16x4_t) -> int16x8_t {
    simd_shuffle8(low, high, [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
pub unsafe fn vcombine_s32(low: int32x2_t, high: int32x2_t) -> int32x4_t {
    simd_shuffle4(low, high, [0, 1, 2, 3])
}

/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
pub unsafe fn vcombine_s64(low: int64x1_t, high: int64x1_t) -> int64x2_t {
    simd_shuffle2(low, high, [0, 1])
}

/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
pub unsafe fn vcombine_u8(low: uint8x8_t, high: uint8x8_t) -> uint8x16_t {
    simd_shuffle16(
        low,
        high,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    )
}

/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
pub unsafe fn vcombine_u16(low: uint16x4_t, high: uint16x4_t) -> uint16x8_t {
    simd_shuffle8(low, high, [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
pub unsafe fn vcombine_u32(low: uint32x2_t, high: uint32x2_t) -> uint32x4_t {
    simd_shuffle4(low, high, [0, 1, 2, 3])
}

/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
pub unsafe fn vcombine_u64(low: uint64x1_t, high: uint64x1_t) -> uint64x2_t {
    simd_shuffle2(low, high, [0, 1])
}

/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
pub unsafe fn vcombine_p64(low: poly64x1_t, high: poly64x1_t) -> poly64x2_t {
    simd_shuffle2(low, high, [0, 1])
}

/* FIXME: 16-bit float
/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
pub unsafe fn vcombine_f16 ( low: float16x4_t,  high: float16x4_t) -> float16x8_t {
    simd_shuffle8(low, high, [0, 1, 2, 3, 4, 5, 6, 7])
}
*/

/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
pub unsafe fn vcombine_f32(
    low: float32x2_t, high: float32x2_t,
) -> float32x4_t {
    simd_shuffle4(low, high, [0, 1, 2, 3])
}

/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
pub unsafe fn vcombine_p8(low: poly8x8_t, high: poly8x8_t) -> poly8x16_t {
    simd_shuffle16(
        low,
        high,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    )
}

/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
pub unsafe fn vcombine_p16(low: poly16x4_t, high: poly16x4_t) -> poly16x8_t {
    simd_shuffle8(low, high, [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
pub unsafe fn vcombine_f64(
    low: float64x1_t, high: float64x1_t,
) -> float64x2_t {
    simd_shuffle2(low, high, [0, 1])
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl1_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    vqtbl1_s8(vcombine_s8(a, ::mem::zeroed()), ::mem::transmute(b))
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl1_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    vqtbl1_u8(vcombine_u8(a, ::mem::zeroed()), b)
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl1_p8(a: poly8x8_t, b: uint8x8_t) -> poly8x8_t {
    vqtbl1_p8(vcombine_p8(a, ::mem::zeroed()), b)
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl2_s8(a: int8x8x2_t, b: int8x8_t) -> int8x8_t {
    vqtbl1_s8(vcombine_s8(a.0, a.1), ::mem::transmute(b))
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl2_u8(a: uint8x8x2_t, b: uint8x8_t) -> uint8x8_t {
    vqtbl1_u8(vcombine_u8(a.0, a.1), b)
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl2_p8(a: poly8x8x2_t, b: uint8x8_t) -> poly8x8_t {
    vqtbl1_p8(vcombine_p8(a.0, a.1), b)
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl3_s8(a: int8x8x3_t, b: int8x8_t) -> int8x8_t {
    vqtbl2_s8(
        int8x16x2_t(vcombine_s8(a.0, a.1), vcombine_s8(a.2, ::mem::zeroed())),
        ::mem::transmute(b),
    )
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl3_u8(a: uint8x8x3_t, b: uint8x8_t) -> uint8x8_t {
    vqtbl2_u8(
        uint8x16x2_t(vcombine_u8(a.0, a.1), vcombine_u8(a.2, ::mem::zeroed())),
        b,
    )
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl3_p8(a: poly8x8x3_t, b: uint8x8_t) -> poly8x8_t {
    vqtbl2_p8(
        poly8x16x2_t(vcombine_p8(a.0, a.1), vcombine_p8(a.2, ::mem::zeroed())),
        b,
    )
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl4_s8(a: int8x8x4_t, b: int8x8_t) -> int8x8_t {
    vqtbl2_s8(
        int8x16x2_t(vcombine_s8(a.0, a.1), vcombine_s8(a.2, a.3)),
        ::mem::transmute(b),
    )
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl4_u8(a: uint8x8x4_t, b: uint8x8_t) -> uint8x8_t {
    vqtbl2_u8(
        uint8x16x2_t(vcombine_u8(a.0, a.1), vcombine_u8(a.2, a.3)),
        b,
    )
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl4_p8(a: poly8x8x4_t, b: uint8x8_t) -> poly8x8_t {
    vqtbl2_p8(
        poly8x16x2_t(vcombine_p8(a.0, a.1), vcombine_p8(a.2, a.3)),
        b,
    )
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx1_s8(a: int8x8_t, b: int8x8_t, c: int8x8_t) -> int8x8_t {
    use coresimd::simd::i8x8;
    let r = vqtbx1_s8(a, vcombine_s8(b, ::mem::zeroed()), ::mem::transmute(c));
    let m: int8x8_t = simd_lt(c, ::mem::transmute(i8x8::splat(8)));
    simd_select(m, r, a)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx1_u8(a: uint8x8_t, b: uint8x8_t, c: uint8x8_t) -> uint8x8_t {
    use coresimd::simd::u8x8;
    let r = vqtbx1_u8(a, vcombine_u8(b, ::mem::zeroed()), c);
    let m: int8x8_t = simd_lt(c, ::mem::transmute(u8x8::splat(8)));
    simd_select(m, r, a)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx1_p8(a: poly8x8_t, b: poly8x8_t, c: uint8x8_t) -> poly8x8_t {
    use coresimd::simd::u8x8;
    let r = vqtbx1_p8(a, vcombine_p8(b, ::mem::zeroed()), c);
    let m: int8x8_t = simd_lt(c, ::mem::transmute(u8x8::splat(8)));
    simd_select(m, r, a)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx2_s8(a: int8x8_t, b: int8x8x2_t, c: int8x8_t) -> int8x8_t {
    vqtbx1_s8(a, vcombine_s8(b.0, b.1), ::mem::transmute(c))
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx2_u8(
    a: uint8x8_t, b: uint8x8x2_t, c: uint8x8_t,
) -> uint8x8_t {
    vqtbx1_u8(a, vcombine_u8(b.0, b.1), c)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx2_p8(
    a: poly8x8_t, b: poly8x8x2_t, c: uint8x8_t,
) -> poly8x8_t {
    vqtbx1_p8(a, vcombine_p8(b.0, b.1), c)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx3_s8(a: int8x8_t, b: int8x8x3_t, c: int8x8_t) -> int8x8_t {
    use coresimd::simd::i8x8;
    let r = vqtbx2_s8(
        a,
        int8x16x2_t(vcombine_s8(b.0, b.1), vcombine_s8(b.2, ::mem::zeroed())),
        ::mem::transmute(c),
    );
    let m: int8x8_t = simd_lt(c, ::mem::transmute(i8x8::splat(24)));
    simd_select(m, r, a)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx3_u8(
    a: uint8x8_t, b: uint8x8x3_t, c: uint8x8_t,
) -> uint8x8_t {
    use coresimd::simd::u8x8;
    let r = vqtbx2_u8(
        a,
        uint8x16x2_t(vcombine_u8(b.0, b.1), vcombine_u8(b.2, ::mem::zeroed())),
        c,
    );
    let m: int8x8_t = simd_lt(c, ::mem::transmute(u8x8::splat(24)));
    simd_select(m, r, a)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx3_p8(
    a: poly8x8_t, b: poly8x8x3_t, c: uint8x8_t,
) -> poly8x8_t {
    use coresimd::simd::u8x8;
    let r = vqtbx2_p8(
        a,
        poly8x16x2_t(vcombine_p8(b.0, b.1), vcombine_p8(b.2, ::mem::zeroed())),
        c,
    );
    let m: int8x8_t = simd_lt(c, ::mem::transmute(u8x8::splat(24)));
    simd_select(m, r, a)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx4_s8(a: int8x8_t, b: int8x8x4_t, c: int8x8_t) -> int8x8_t {
    vqtbx2_s8(
        a,
        int8x16x2_t(vcombine_s8(b.0, b.1), vcombine_s8(b.2, b.3)),
        ::mem::transmute(c),
    )
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx4_u8(
    a: uint8x8_t, b: uint8x8x4_t, c: uint8x8_t,
) -> uint8x8_t {
    vqtbx2_u8(
        a,
        uint8x16x2_t(vcombine_u8(b.0, b.1), vcombine_u8(b.2, b.3)),
        c,
    )
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx4_p8(
    a: poly8x8_t, b: poly8x8x4_t, c: uint8x8_t,
) -> poly8x8_t {
    vqtbx2_p8(
        a,
        poly8x16x2_t(vcombine_p8(b.0, b.1), vcombine_p8(b.2, b.3)),
        c,
    )
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl1_s8(t: int8x16_t, idx: uint8x8_t) -> int8x8_t {
    vqtbl1(t, idx)
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl1q_s8(t: int8x16_t, idx: uint8x16_t) -> int8x16_t {
    vqtbl1q(t, idx)
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl1_u8(t: uint8x16_t, idx: uint8x8_t) -> uint8x8_t {
    ::mem::transmute(vqtbl1(::mem::transmute(t), ::mem::transmute(idx)))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl1q_u8(t: uint8x16_t, idx: uint8x16_t) -> uint8x16_t {
    ::mem::transmute(vqtbl1q(::mem::transmute(t), ::mem::transmute(idx)))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl1_p8(t: poly8x16_t, idx: uint8x8_t) -> poly8x8_t {
    ::mem::transmute(vqtbl1(::mem::transmute(t), ::mem::transmute(idx)))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl1q_p8(t: poly8x16_t, idx: uint8x16_t) -> poly8x16_t {
    ::mem::transmute(vqtbl1q(::mem::transmute(t), ::mem::transmute(idx)))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx1_s8(
    a: int8x8_t, t: int8x16_t, idx: uint8x8_t,
) -> int8x8_t {
    vqtbx1(a, t, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx1q_s8(
    a: int8x16_t, t: int8x16_t, idx: uint8x16_t,
) -> int8x16_t {
    vqtbx1q(a, t, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx1_u8(
    a: uint8x8_t, t: uint8x16_t, idx: uint8x8_t,
) -> uint8x8_t {
    ::mem::transmute(vqtbx1(
        ::mem::transmute(a),
        ::mem::transmute(t),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx1q_u8(
    a: uint8x16_t, t: uint8x16_t, idx: uint8x16_t,
) -> uint8x16_t {
    ::mem::transmute(vqtbx1q(
        ::mem::transmute(a),
        ::mem::transmute(t),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx1_p8(
    a: poly8x8_t, t: poly8x16_t, idx: uint8x8_t,
) -> poly8x8_t {
    ::mem::transmute(vqtbx1(
        ::mem::transmute(a),
        ::mem::transmute(t),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx1q_p8(
    a: poly8x16_t, t: poly8x16_t, idx: uint8x16_t,
) -> poly8x16_t {
    ::mem::transmute(vqtbx1q(
        ::mem::transmute(a),
        ::mem::transmute(t),
        ::mem::transmute(idx),
    ))
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl2_s8(t: int8x16x2_t, idx: uint8x8_t) -> int8x8_t {
    vqtbl2(t.0, t.1, idx)
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl2q_s8(t: int8x16x2_t, idx: uint8x16_t) -> int8x16_t {
    vqtbl2q(t.0, t.1, idx)
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl2_u8(t: uint8x16x2_t, idx: uint8x8_t) -> uint8x8_t {
    ::mem::transmute(vqtbl2(
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl2q_u8(t: uint8x16x2_t, idx: uint8x16_t) -> uint8x16_t {
    ::mem::transmute(vqtbl2q(
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl2_p8(t: poly8x16x2_t, idx: uint8x8_t) -> poly8x8_t {
    ::mem::transmute(vqtbl2(
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl2q_p8(t: poly8x16x2_t, idx: uint8x16_t) -> poly8x16_t {
    ::mem::transmute(vqtbl2q(
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx2_s8(
    a: int8x8_t, t: int8x16x2_t, idx: uint8x8_t,
) -> int8x8_t {
    vqtbx2(a, t.0, t.1, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx2q_s8(
    a: int8x16_t, t: int8x16x2_t, idx: uint8x16_t,
) -> int8x16_t {
    vqtbx2q(a, t.0, t.1, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx2_u8(
    a: uint8x8_t, t: uint8x16x2_t, idx: uint8x8_t,
) -> uint8x8_t {
    ::mem::transmute(vqtbx2(
        ::mem::transmute(a),
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx2q_u8(
    a: uint8x16_t, t: uint8x16x2_t, idx: uint8x16_t,
) -> uint8x16_t {
    ::mem::transmute(vqtbx2q(
        ::mem::transmute(a),
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx2_p8(
    a: poly8x8_t, t: poly8x16x2_t, idx: uint8x8_t,
) -> poly8x8_t {
    ::mem::transmute(vqtbx2(
        ::mem::transmute(a),
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx2q_p8(
    a: poly8x16_t, t: poly8x16x2_t, idx: uint8x16_t,
) -> poly8x16_t {
    ::mem::transmute(vqtbx2q(
        ::mem::transmute(a),
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(idx),
    ))
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl3_s8(t: int8x16x3_t, idx: uint8x8_t) -> int8x8_t {
    vqtbl3(t.0, t.1, t.2, idx)
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl3q_s8(t: int8x16x3_t, idx: uint8x16_t) -> int8x16_t {
    vqtbl3q(t.0, t.1, t.2, idx)
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl3_u8(t: uint8x16x3_t, idx: uint8x8_t) -> uint8x8_t {
    ::mem::transmute(vqtbl3(
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl3q_u8(t: uint8x16x3_t, idx: uint8x16_t) -> uint8x16_t {
    ::mem::transmute(vqtbl3q(
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl3_p8(t: poly8x16x3_t, idx: uint8x8_t) -> poly8x8_t {
    ::mem::transmute(vqtbl3(
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl3q_p8(t: poly8x16x3_t, idx: uint8x16_t) -> poly8x16_t {
    ::mem::transmute(vqtbl3q(
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx3_s8(
    a: int8x8_t, t: int8x16x3_t, idx: uint8x8_t,
) -> int8x8_t {
    vqtbx3(a, t.0, t.1, t.2, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx3q_s8(
    a: int8x16_t, t: int8x16x3_t, idx: uint8x16_t,
) -> int8x16_t {
    vqtbx3q(a, t.0, t.1, t.2, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx3_u8(
    a: uint8x8_t, t: uint8x16x3_t, idx: uint8x8_t,
) -> uint8x8_t {
    ::mem::transmute(vqtbx3(
        ::mem::transmute(a),
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx3q_u8(
    a: uint8x16_t, t: uint8x16x3_t, idx: uint8x16_t,
) -> uint8x16_t {
    ::mem::transmute(vqtbx3q(
        ::mem::transmute(a),
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx3_p8(
    a: poly8x8_t, t: poly8x16x3_t, idx: uint8x8_t,
) -> poly8x8_t {
    ::mem::transmute(vqtbx3(
        ::mem::transmute(a),
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx3q_p8(
    a: poly8x16_t, t: poly8x16x3_t, idx: uint8x16_t,
) -> poly8x16_t {
    ::mem::transmute(vqtbx3q(
        ::mem::transmute(a),
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(idx),
    ))
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl4_s8(t: int8x16x4_t, idx: uint8x8_t) -> int8x8_t {
    vqtbl4(t.0, t.1, t.2, t.3, idx)
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl4q_s8(t: int8x16x4_t, idx: uint8x16_t) -> int8x16_t {
    vqtbl4q(t.0, t.1, t.2, t.3, idx)
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl4_u8(t: uint8x16x4_t, idx: uint8x8_t) -> uint8x8_t {
    ::mem::transmute(vqtbl4(
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(t.3),
        ::mem::transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl4q_u8(t: uint8x16x4_t, idx: uint8x16_t) -> uint8x16_t {
    ::mem::transmute(vqtbl4q(
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(t.3),
        ::mem::transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl4_p8(t: poly8x16x4_t, idx: uint8x8_t) -> poly8x8_t {
    ::mem::transmute(vqtbl4(
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(t.3),
        ::mem::transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl4q_p8(t: poly8x16x4_t, idx: uint8x16_t) -> poly8x16_t {
    ::mem::transmute(vqtbl4q(
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(t.3),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx4_s8(
    a: int8x8_t, t: int8x16x4_t, idx: uint8x8_t,
) -> int8x8_t {
    vqtbx4(a, t.0, t.1, t.2, t.3, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx4q_s8(
    a: int8x16_t, t: int8x16x4_t, idx: uint8x16_t,
) -> int8x16_t {
    vqtbx4q(a, t.0, t.1, t.2, t.3, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx4_u8(
    a: uint8x8_t, t: uint8x16x4_t, idx: uint8x8_t,
) -> uint8x8_t {
    ::mem::transmute(vqtbx4(
        ::mem::transmute(a),
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(t.3),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx4q_u8(
    a: uint8x16_t, t: uint8x16x4_t, idx: uint8x16_t,
) -> uint8x16_t {
    ::mem::transmute(vqtbx4q(
        ::mem::transmute(a),
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(t.3),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx4_p8(
    a: poly8x8_t, t: poly8x16x4_t, idx: uint8x8_t,
) -> poly8x8_t {
    ::mem::transmute(vqtbx4(
        ::mem::transmute(a),
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(t.3),
        ::mem::transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx4q_p8(
    a: poly8x16_t, t: poly8x16x4_t, idx: uint8x16_t,
) -> poly8x16_t {
    ::mem::transmute(vqtbx4q(
        ::mem::transmute(a),
        ::mem::transmute(t.0),
        ::mem::transmute(t.1),
        ::mem::transmute(t.2),
        ::mem::transmute(t.3),
        ::mem::transmute(idx),
    ))
}

#[cfg(test)]
mod tests {
    use coresimd::aarch64::*;
    use coresimd::simd::*;
    use std::mem;
    use stdsimd_test::simd_test;

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_f64() {
        let a = 1.;
        let b = 8.;
        let e = 9.;
        let r: f64 =
            mem::transmute(vadd_f64(mem::transmute(a), mem::transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_f64() {
        let a = f64x2::new(1., 2.);
        let b = f64x2::new(8., 7.);
        let e = f64x2::new(9., 9.);
        let r: f64x2 = ::mem::transmute(vaddq_f64(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddd_s64() {
        let a = 1_i64;
        let b = 8_i64;
        let e = 9_i64;
        let r: i64 =
            mem::transmute(vaddd_s64(mem::transmute(a), mem::transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddd_u64() {
        let a = 1_u64;
        let b = 8_u64;
        let e = 9_u64;
        let r: u64 =
            mem::transmute(vaddd_u64(mem::transmute(a), mem::transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxv_s8() {
        let r = vmaxv_s8(::mem::transmute(i8x8::new(1, 2, 3, 4, -8, 6, 7, 5)));
        assert_eq!(r, 7_i8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_s8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let r = vmaxvq_s8(::mem::transmute(i8x16::new(
            1, 2, 3, 4,
            -16, 6, 7, 5,
            8, 1, 1, 1,
            1, 1, 1, 1,
        )));
        assert_eq!(r, 8_i8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxv_s16() {
        let r = vmaxv_s16(::mem::transmute(i16x4::new(1, 2, -4, 3)));
        assert_eq!(r, 3_i16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_s16() {
        let r =
            vmaxvq_s16(::mem::transmute(i16x8::new(1, 2, 7, 4, -16, 6, 7, 5)));
        assert_eq!(r, 7_i16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxv_s32() {
        let r = vmaxv_s32(::mem::transmute(i32x2::new(1, -4)));
        assert_eq!(r, 1_i32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_s32() {
        let r = vmaxvq_s32(::mem::transmute(i32x4::new(1, 2, -32, 4)));
        assert_eq!(r, 4_i32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxv_u8() {
        let r = vmaxv_u8(::mem::transmute(u8x8::new(1, 2, 3, 4, 8, 6, 7, 5)));
        assert_eq!(r, 8_u8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_u8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let r = vmaxvq_u8(::mem::transmute(u8x16::new(
            1, 2, 3, 4,
            16, 6, 7, 5,
            8, 1, 1, 1,
            1, 1, 1, 1,
        )));
        assert_eq!(r, 16_u8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxv_u16() {
        let r = vmaxv_u16(::mem::transmute(u16x4::new(1, 2, 4, 3)));
        assert_eq!(r, 4_u16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_u16() {
        let r =
            vmaxvq_u16(::mem::transmute(u16x8::new(1, 2, 7, 4, 16, 6, 7, 5)));
        assert_eq!(r, 16_u16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxv_u32() {
        let r = vmaxv_u32(::mem::transmute(u32x2::new(1, 4)));
        assert_eq!(r, 4_u32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_u32() {
        let r = vmaxvq_u32(::mem::transmute(u32x4::new(1, 2, 32, 4)));
        assert_eq!(r, 32_u32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxv_f32() {
        let r = vmaxv_f32(::mem::transmute(f32x2::new(1., 4.)));
        assert_eq!(r, 4_f32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_f32() {
        let r = vmaxvq_f32(::mem::transmute(f32x4::new(1., 2., 32., 4.)));
        assert_eq!(r, 32_f32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_f64() {
        let r = vmaxvq_f64(::mem::transmute(f64x2::new(1., 4.)));
        assert_eq!(r, 4_f64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminv_s8() {
        let r = vminv_s8(::mem::transmute(i8x8::new(1, 2, 3, 4, -8, 6, 7, 5)));
        assert_eq!(r, -8_i8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_s8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let r = vminvq_s8(::mem::transmute(i8x16::new(
            1, 2, 3, 4,
            -16, 6, 7, 5,
            8, 1, 1, 1,
            1, 1, 1, 1,
        )));
        assert_eq!(r, -16_i8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminv_s16() {
        let r = vminv_s16(::mem::transmute(i16x4::new(1, 2, -4, 3)));
        assert_eq!(r, -4_i16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_s16() {
        let r =
            vminvq_s16(::mem::transmute(i16x8::new(1, 2, 7, 4, -16, 6, 7, 5)));
        assert_eq!(r, -16_i16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminv_s32() {
        let r = vminv_s32(::mem::transmute(i32x2::new(1, -4)));
        assert_eq!(r, -4_i32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_s32() {
        let r = vminvq_s32(::mem::transmute(i32x4::new(1, 2, -32, 4)));
        assert_eq!(r, -32_i32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminv_u8() {
        let r = vminv_u8(::mem::transmute(u8x8::new(1, 2, 3, 4, 8, 6, 7, 5)));
        assert_eq!(r, 1_u8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_u8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let r = vminvq_u8(::mem::transmute(u8x16::new(
            1, 2, 3, 4,
            16, 6, 7, 5,
            8, 1, 1, 1,
            1, 1, 1, 1,
        )));
        assert_eq!(r, 1_u8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminv_u16() {
        let r = vminv_u16(::mem::transmute(u16x4::new(1, 2, 4, 3)));
        assert_eq!(r, 1_u16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_u16() {
        let r =
            vminvq_u16(::mem::transmute(u16x8::new(1, 2, 7, 4, 16, 6, 7, 5)));
        assert_eq!(r, 1_u16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminv_u32() {
        let r = vminv_u32(::mem::transmute(u32x2::new(1, 4)));
        assert_eq!(r, 1_u32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_u32() {
        let r = vminvq_u32(::mem::transmute(u32x4::new(1, 2, 32, 4)));
        assert_eq!(r, 1_u32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminv_f32() {
        let r = vminv_f32(::mem::transmute(f32x2::new(1., 4.)));
        assert_eq!(r, 1_f32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_f32() {
        let r = vminvq_f32(::mem::transmute(f32x4::new(1., 2., 32., 4.)));
        assert_eq!(r, 1_f32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_f64() {
        let r = vminvq_f64(::mem::transmute(f64x2::new(1., 4.)));
        assert_eq!(r, 1_f64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpminq_s8() {
        #[cfg_attr(rustfmt, skip)]
        let a = i8x16::new(1, -2, 3, -4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8);
        #[cfg_attr(rustfmt, skip)]
        let b = i8x16::new(0, 3, 2, 5, 4, 7, 6, 9, 0, 3, 2, 5, 4, 7, 6, 9);
        #[cfg_attr(rustfmt, skip)]
        let e = i8x16::new(-2, -4, 5, 7, 1, 3, 5, 7, 0, 2, 4, 6, 0, 2, 4, 6);
        let r: i8x16 = ::mem::transmute(vpminq_s8(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpminq_s16() {
        let a = i16x8::new(1, -2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = i16x8::new(-2, 3, 5, 7, 0, 2, 4, 6);
        let r: i16x8 = ::mem::transmute(vpminq_s16(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpminq_s32() {
        let a = i32x4::new(1, -2, 3, 4);
        let b = i32x4::new(0, 3, 2, 5);
        let e = i32x4::new(-2, 3, 0, 2);
        let r: i32x4 = ::mem::transmute(vpminq_s32(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpminq_u8() {
        #[cfg_attr(rustfmt, skip)]
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8);
        #[cfg_attr(rustfmt, skip)]
        let b = u8x16::new(0, 3, 2, 5, 4, 7, 6, 9, 0, 3, 2, 5, 4, 7, 6, 9);
        #[cfg_attr(rustfmt, skip)]
        let e = u8x16::new(1, 3, 5, 7, 1, 3, 5, 7, 0, 2, 4, 6, 0, 2, 4, 6);
        let r: u8x16 = ::mem::transmute(vpminq_u8(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpminq_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u16x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = u16x8::new(1, 3, 5, 7, 0, 2, 4, 6);
        let r: u16x8 = ::mem::transmute(vpminq_u16(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpminq_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let b = u32x4::new(0, 3, 2, 5);
        let e = u32x4::new(1, 3, 0, 2);
        let r: u32x4 = ::mem::transmute(vpminq_u32(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmin_f32() {
        let a = f32x4::new(1., -2., 3., 4.);
        let b = f32x4::new(0., 3., 2., 5.);
        let e = f32x4::new(-2., 3., 0., 2.);
        let r: f32x4 = ::mem::transmute(vpminq_f32(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmin_f64() {
        let a = f64x2::new(1., -2.);
        let b = f64x2::new(0., 3.);
        let e = f64x2::new(-2., 0.);
        let r: f64x2 = ::mem::transmute(vpminq_f64(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmaxq_s8() {
        #[cfg_attr(rustfmt, skip)]
        let a = i8x16::new(1, -2, 3, -4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8);
        #[cfg_attr(rustfmt, skip)]
        let b = i8x16::new(0, 3, 2, 5, 4, 7, 6, 9, 0, 3, 2, 5, 4, 7, 6, 9);
        #[cfg_attr(rustfmt, skip)]
        let e = i8x16::new(1, 3, 6, 8, 2, 4, 6, 8, 3, 5, 7, 9, 3, 5, 7, 9);
        let r: i8x16 = ::mem::transmute(vpmaxq_s8(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmaxq_s16() {
        let a = i16x8::new(1, -2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = i16x8::new(1, 4, 6, 8, 3, 5, 7, 9);
        let r: i16x8 = ::mem::transmute(vpmaxq_s16(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmaxq_s32() {
        let a = i32x4::new(1, -2, 3, 4);
        let b = i32x4::new(0, 3, 2, 5);
        let e = i32x4::new(1, 4, 3, 5);
        let r: i32x4 = ::mem::transmute(vpmaxq_s32(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmaxq_u8() {
        #[cfg_attr(rustfmt, skip)]
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8);
        #[cfg_attr(rustfmt, skip)]
        let b = u8x16::new(0, 3, 2, 5, 4, 7, 6, 9, 0, 3, 2, 5, 4, 7, 6, 9);
        #[cfg_attr(rustfmt, skip)]
        let e = u8x16::new(2, 4, 6, 8, 2, 4, 6, 8, 3, 5, 7, 9, 3, 5, 7, 9);
        let r: u8x16 = ::mem::transmute(vpmaxq_u8(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmaxq_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u16x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = u16x8::new(2, 4, 6, 8, 3, 5, 7, 9);
        let r: u16x8 = ::mem::transmute(vpmaxq_u16(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmaxq_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let b = u32x4::new(0, 3, 2, 5);
        let e = u32x4::new(2, 4, 3, 5);
        let r: u32x4 = ::mem::transmute(vpmaxq_u32(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmax_f32() {
        let a = f32x4::new(1., -2., 3., 4.);
        let b = f32x4::new(0., 3., 2., 5.);
        let e = f32x4::new(1., 4., 3., 5.);
        let r: f32x4 = ::mem::transmute(vpmaxq_f32(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmax_f64() {
        let a = f64x2::new(1., -2.);
        let b = f64x2::new(0., 3.);
        let e = f64x2::new(1., 3.);
        let r: f64x2 = ::mem::transmute(vpmaxq_f64(
            ::mem::transmute(a),
            ::mem::transmute(b),
        ));
        assert_eq!(r, e);
    }

    macro_rules! test_vcombine {
        ($test_id:ident => $fn_id:ident ([$($a:expr),*], [$($b:expr),*])) => {
            #[allow(unused_assignments)]
            #[simd_test(enable = "neon")]
            unsafe fn $test_id() {
                let a = [$($a),*];
                let b = [$($b),*];
                let e = [$($a),* $(, $b)*];
                let c = $fn_id(::mem::transmute(a), ::mem::transmute(b));
                let mut d = e;
                d = ::mem::transmute(c);
                assert_eq!(d, e);
            }
        }
    }

    test_vcombine!(test_vcombine_s8 => vcombine_s8([3_i8, -4, 5, -6, 7, 8, 9, 10], [13_i8, -14, 15, -16, 17, 18, 19, 110]));
    test_vcombine!(test_vcombine_u8 => vcombine_u8([3_u8, 4, 5, 6, 7, 8, 9, 10], [13_u8, 14, 15, 16, 17, 18, 19, 110]));
    test_vcombine!(test_vcombine_p8 => vcombine_p8([3_u8, 4, 5, 6, 7, 8, 9, 10], [13_u8, 14, 15, 16, 17, 18, 19, 110]));

    test_vcombine!(test_vcombine_s16 => vcombine_s16([3_i16, -4, 5, -6], [13_i16, -14, 15, -16]));
    test_vcombine!(test_vcombine_u16 => vcombine_u16([3_u16, 4, 5, 6], [13_u16, 14, 15, 16]));
    test_vcombine!(test_vcombine_p16 => vcombine_p16([3_u16, 4, 5, 6], [13_u16, 14, 15, 16]));
    // FIXME: 16-bit floats
    // test_vcombine!(test_vcombine_f16 => vcombine_f16([3_f16, 4., 5., 6.],
    // [13_f16, 14., 15., 16.]));

    test_vcombine!(test_vcombine_s32 => vcombine_s32([3_i32, -4], [13_i32, -14]));
    test_vcombine!(test_vcombine_u32 => vcombine_u32([3_u32, 4], [13_u32, 14]));
    // note: poly32x4 does not exist, and neither does vcombine_p32
    test_vcombine!(test_vcombine_f32 => vcombine_f32([3_f32, -4.], [13_f32, -14.]));

    test_vcombine!(test_vcombine_s64 => vcombine_s64([-3_i64], [13_i64]));
    test_vcombine!(test_vcombine_u64 => vcombine_u64([3_u64], [13_u64]));
    test_vcombine!(test_vcombine_p64 => vcombine_p64([3_u64], [13_u64]));
    test_vcombine!(test_vcombine_f64 => vcombine_f64([-3_f64], [13_f64]));

}

#[cfg(test)]
#[cfg(target_endian = "little")]
#[path = "../arm/table_lookup_tests.rs"]
mod table_lookup_tests;
