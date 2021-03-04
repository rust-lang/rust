//! ARMv8 ASIMD intrinsics

#![allow(non_camel_case_types)]

#[rustfmt::skip]
mod generated;
#[rustfmt::skip]
pub use self::generated::*;

// FIXME: replace neon with asimd

use crate::{
    core_arch::{arm::*, simd::*, simd_llvm::*},
    mem::{transmute, zeroed},
};
#[cfg(test)]
use stdarch_test::assert_instr;

types! {
    /// ARM-specific 64-bit wide vector of one packed `f64`.
    pub struct float64x1_t(f64); // FIXME: check this!
    /// ARM-specific 128-bit wide vector of two packed `f64`.
    pub struct float64x2_t(f64, f64);
}

/// ARM-specific type containing two `int8x16_t` vectors.
#[derive(Copy, Clone)]
pub struct int8x16x2_t(pub int8x16_t, pub int8x16_t);
/// ARM-specific type containing three `int8x16_t` vectors.
#[derive(Copy, Clone)]
pub struct int8x16x3_t(pub int8x16_t, pub int8x16_t, pub int8x16_t);
/// ARM-specific type containing four `int8x16_t` vectors.
#[derive(Copy, Clone)]
pub struct int8x16x4_t(pub int8x16_t, pub int8x16_t, pub int8x16_t, pub int8x16_t);

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
    // absolute value
    #[link_name = "llvm.aarch64.neon.abs.i64"]
    fn vabsd_s64_(a: i64) -> i64;
    #[link_name = "llvm.aarch64.neon.abs.v1i64"]
    fn vabs_s64_(a: int64x1_t) -> int64x1_t;
    #[link_name = "llvm.aarch64.neon.abs.v2i64"]
    fn vabsq_s64_(a: int64x2_t) -> int64x2_t;
    #[link_name = "llvm.fabs.v1f64"]
    fn vabs_f64_(a: float64x1_t) -> float64x1_t;
    #[link_name = "llvm.fabs.v2f64"]
    fn vabsq_f64_(a: float64x2_t) -> float64x2_t;

    #[link_name = "llvm.aarch64.neon.suqadd.v8i8"]
    fn vuqadd_s8_(a: int8x8_t, b: uint8x8_t) -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.suqadd.v16i8"]
    fn vuqaddq_s8_(a: int8x16_t, b: uint8x16_t) -> int8x16_t;
    #[link_name = "llvm.aarch64.neon.suqadd.v4i16"]
    fn vuqadd_s16_(a: int16x4_t, b: uint16x4_t) -> int16x4_t;
    #[link_name = "llvm.aarch64.neon.suqadd.v8i16"]
    fn vuqaddq_s16_(a: int16x8_t, b: uint16x8_t) -> int16x8_t;
    #[link_name = "llvm.aarch64.neon.suqadd.v2i32"]
    fn vuqadd_s32_(a: int32x2_t, b: uint32x2_t) -> int32x2_t;
    #[link_name = "llvm.aarch64.neon.suqadd.v4i32"]
    fn vuqaddq_s32_(a: int32x4_t, b: uint32x4_t) -> int32x4_t;
    #[link_name = "llvm.aarch64.neon.suqadd.v1i64"]
    fn vuqadd_s64_(a: int64x1_t, b: uint64x1_t) -> int64x1_t;
    #[link_name = "llvm.aarch64.neon.suqadd.v2i64"]
    fn vuqaddq_s64_(a: int64x2_t, b: uint64x2_t) -> int64x2_t;

    #[link_name = "llvm.aarch64.neon.usqadd.v8i8"]
    fn vsqadd_u8_(a: uint8x8_t, b: int8x8_t) -> uint8x8_t;
    #[link_name = "llvm.aarch64.neon.usqadd.v16i8"]
    fn vsqaddq_u8_(a: uint8x16_t, b: int8x16_t) -> uint8x16_t;
    #[link_name = "llvm.aarch64.neon.usqadd.v4i16"]
    fn vsqadd_u16_(a: uint16x4_t, b: int16x4_t) -> uint16x4_t;
    #[link_name = "llvm.aarch64.neon.usqadd.v8i16"]
    fn vsqaddq_u16_(a: uint16x8_t, b: int16x8_t) -> uint16x8_t;
    #[link_name = "llvm.aarch64.neon.usqadd.v2i32"]
    fn vsqadd_u32_(a: uint32x2_t, b: int32x2_t) -> uint32x2_t;
    #[link_name = "llvm.aarch64.neon.usqadd.v4i32"]
    fn vsqaddq_u32_(a: uint32x4_t, b: int32x4_t) -> uint32x4_t;
    #[link_name = "llvm.aarch64.neon.usqadd.v1i64"]
    fn vsqadd_u64_(a: uint64x1_t, b: int64x1_t) -> uint64x1_t;
    #[link_name = "llvm.aarch64.neon.usqadd.v2i64"]
    fn vsqaddq_u64_(a: uint64x2_t, b: int64x2_t) -> uint64x2_t;

    #[link_name = "llvm.aarch64.neon.pmull64"]
    fn vmull_p64_(a: i64, b: i64) -> int8x16_t;

    #[link_name = "llvm.aarch64.neon.addp.v8i16"]
    fn vpaddq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    #[link_name = "llvm.aarch64.neon.addp.v4i32"]
    fn vpaddq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    #[link_name = "llvm.aarch64.neon.addp.v16i8"]
    fn vpaddq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;

    #[link_name = "llvm.aarch64.neon.saddv.i32.v4i16"]
    fn vaddv_s16_(a: int16x4_t) -> i16;
    #[link_name = "llvm.aarch64.neon.saddv.i32.v2i32"]
    fn vaddv_s32_(a: int32x2_t) -> i32;
    #[link_name = "llvm.aarch64.neon.saddv.i32.v8i8"]
    fn vaddv_s8_(a: int8x8_t) -> i8;
    #[link_name = "llvm.aarch64.neon.uaddv.i32.v4i16"]
    fn vaddv_u16_(a: uint16x4_t) -> u16;
    #[link_name = "llvm.aarch64.neon.uaddv.i32.v2i32"]
    fn vaddv_u32_(a: uint32x2_t) -> u32;
    #[link_name = "llvm.aarch64.neon.uaddv.i32.v8i8"]
    fn vaddv_u8_(a: uint8x8_t) -> u8;
    #[link_name = "llvm.aarch64.neon.saddv.i32.v8i16"]
    fn vaddvq_s16_(a: int16x8_t) -> i16;
    #[link_name = "llvm.aarch64.neon.saddv.i32.v4i32"]
    fn vaddvq_s32_(a: int32x4_t) -> i32;
    #[link_name = "llvm.aarch64.neon.saddv.i32.v16i8"]
    fn vaddvq_s8_(a: int8x16_t) -> i8;
    #[link_name = "llvm.aarch64.neon.uaddv.i32.v8i16"]
    fn vaddvq_u16_(a: uint16x8_t) -> u16;
    #[link_name = "llvm.aarch64.neon.uaddv.i32.v4i32"]
    fn vaddvq_u32_(a: uint32x4_t) -> u32;
    #[link_name = "llvm.aarch64.neon.uaddv.i32.v16i8"]
    fn vaddvq_u8_(a: uint8x16_t) -> u8;
    #[link_name = "llvm.aarch64.neon.saddv.i64.v2i64"]
    fn vaddvq_s64_(a: int64x2_t) -> i64;
    #[link_name = "llvm.aarch64.neon.uaddv.i64.v2i64"]
    fn vaddvq_u64_(a: uint64x2_t) -> u64;

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
    fn vqtbx2(a: int8x8_t, b0: int8x16_t, b1: int8x16_t, c: uint8x8_t) -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.tbx2.v16i8"]
    fn vqtbx2q(a: int8x16_t, b0: int8x16_t, b1: int8x16_t, c: uint8x16_t) -> int8x16_t;

    #[link_name = "llvm.aarch64.neon.tbl3.v8i8"]
    fn vqtbl3(a0: int8x16_t, a1: int8x16_t, a2: int8x16_t, b: uint8x8_t) -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.tbl3.v16i8"]
    fn vqtbl3q(a0: int8x16_t, a1: int8x16_t, a2: int8x16_t, b: uint8x16_t) -> int8x16_t;

    #[link_name = "llvm.aarch64.neon.tbx3.v8i8"]
    fn vqtbx3(a: int8x8_t, b0: int8x16_t, b1: int8x16_t, b2: int8x16_t, c: uint8x8_t) -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.tbx3.v16i8"]
    fn vqtbx3q(
        a: int8x16_t,
        b0: int8x16_t,
        b1: int8x16_t,
        b2: int8x16_t,
        c: uint8x16_t,
    ) -> int8x16_t;

    #[link_name = "llvm.aarch64.neon.tbl4.v8i8"]
    fn vqtbl4(a0: int8x16_t, a1: int8x16_t, a2: int8x16_t, a3: int8x16_t, b: uint8x8_t)
        -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.tbl4.v16i8"]
    fn vqtbl4q(
        a0: int8x16_t,
        a1: int8x16_t,
        a2: int8x16_t,
        a3: int8x16_t,
        b: uint8x16_t,
    ) -> int8x16_t;

    #[link_name = "llvm.aarch64.neon.tbx4.v8i8"]
    fn vqtbx4(
        a: int8x8_t,
        b0: int8x16_t,
        b1: int8x16_t,
        b2: int8x16_t,
        b3: int8x16_t,
        c: uint8x8_t,
    ) -> int8x8_t;

    #[link_name = "llvm.aarch64.neon.tbx4.v16i8"]
    fn vqtbx4q(
        a: int8x16_t,
        b0: int8x16_t,
        b1: int8x16_t,
        b2: int8x16_t,
        b3: int8x16_t,
        c: uint8x16_t,
    ) -> int8x16_t;

    #[link_name = "llvm.aarch64.neon.fcvtzu.v4i32.v4f32"]
    fn vcvtq_u32_f32_(a: float32x4_t) -> uint32x4_t;
    #[link_name = "llvm.aarch64.neon.fcvtzs.v4i32.v4f32"]
    fn vcvtq_s32_f32_(a: float32x4_t) -> int32x4_t;

    #[link_name = "llvm.aarch64.neon.vsli.v8i8"]
    fn vsli_n_s8_(a: int8x8_t, b: int8x8_t, n: i32) -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.vsli.v16i8"]
    fn vsliq_n_s8_(a: int8x16_t, b: int8x16_t, n: i32) -> int8x16_t;
    #[link_name = "llvm.aarch64.neon.vsli.v4i16"]
    fn vsli_n_s16_(a: int16x4_t, b: int16x4_t, n: i32) -> int16x4_t;
    #[link_name = "llvm.aarch64.neon.vsli.v8i16"]
    fn vsliq_n_s16_(a: int16x8_t, b: int16x8_t, n: i32) -> int16x8_t;
    #[link_name = "llvm.aarch64.neon.vsli.v2i32"]
    fn vsli_n_s32_(a: int32x2_t, b: int32x2_t, n: i32) -> int32x2_t;
    #[link_name = "llvm.aarch64.neon.vsli.v4i32"]
    fn vsliq_n_s32_(a: int32x4_t, b: int32x4_t, n: i32) -> int32x4_t;
    #[link_name = "llvm.aarch64.neon.vsli.v1i64"]
    fn vsli_n_s64_(a: int64x1_t, b: int64x1_t, n: i32) -> int64x1_t;
    #[link_name = "llvm.aarch64.neon.vsli.v2i64"]
    fn vsliq_n_s64_(a: int64x2_t, b: int64x2_t, n: i32) -> int64x2_t;

    #[link_name = "llvm.aarch64.neon.vsri.v8i8"]
    fn vsri_n_s8_(a: int8x8_t, b: int8x8_t, n: i32) -> int8x8_t;
    #[link_name = "llvm.aarch64.neon.vsri.v16i8"]
    fn vsriq_n_s8_(a: int8x16_t, b: int8x16_t, n: i32) -> int8x16_t;
    #[link_name = "llvm.aarch64.neon.vsri.v4i16"]
    fn vsri_n_s16_(a: int16x4_t, b: int16x4_t, n: i32) -> int16x4_t;
    #[link_name = "llvm.aarch64.neon.vsri.v8i16"]
    fn vsriq_n_s16_(a: int16x8_t, b: int16x8_t, n: i32) -> int16x8_t;
    #[link_name = "llvm.aarch64.neon.vsri.v2i32"]
    fn vsri_n_s32_(a: int32x2_t, b: int32x2_t, n: i32) -> int32x2_t;
    #[link_name = "llvm.aarch64.neon.vsri.v4i32"]
    fn vsriq_n_s32_(a: int32x4_t, b: int32x4_t, n: i32) -> int32x4_t;
    #[link_name = "llvm.aarch64.neon.vsri.v1i64"]
    fn vsri_n_s64_(a: int64x1_t, b: int64x1_t, n: i32) -> int64x1_t;
    #[link_name = "llvm.aarch64.neon.vsri.v2i64"]
    fn vsriq_n_s64_(a: int64x2_t, b: int64x2_t, n: i32) -> int64x2_t;
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1_s8(ptr: *const i8) -> int8x8_t {
    transmute(i8x8::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
        *ptr.offset(4),
        *ptr.offset(5),
        *ptr.offset(6),
        *ptr.offset(7),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1q_s8(ptr: *const i8) -> int8x16_t {
    transmute(i8x16::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
        *ptr.offset(4),
        *ptr.offset(5),
        *ptr.offset(6),
        *ptr.offset(7),
        *ptr.offset(8),
        *ptr.offset(9),
        *ptr.offset(10),
        *ptr.offset(11),
        *ptr.offset(12),
        *ptr.offset(13),
        *ptr.offset(14),
        *ptr.offset(15),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1_s16(ptr: *const i16) -> int16x4_t {
    transmute(i16x4::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1q_s16(ptr: *const i16) -> int16x8_t {
    transmute(i16x8::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
        *ptr.offset(4),
        *ptr.offset(5),
        *ptr.offset(6),
        *ptr.offset(7),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1_s32(ptr: *const i32) -> int32x2_t {
    transmute(i32x2::new(*ptr, *ptr.offset(1)))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1q_s32(ptr: *const i32) -> int32x4_t {
    transmute(i32x4::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1_s64(ptr: *const i64) -> int64x1_t {
    transmute(i64x1::new(*ptr))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1q_s64(ptr: *const i64) -> int64x2_t {
    transmute(i64x2::new(*ptr, *ptr.offset(1)))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1_u8(ptr: *const u8) -> uint8x8_t {
    transmute(u8x8::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
        *ptr.offset(4),
        *ptr.offset(5),
        *ptr.offset(6),
        *ptr.offset(7),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1q_u8(ptr: *const u8) -> uint8x16_t {
    transmute(u8x16::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
        *ptr.offset(4),
        *ptr.offset(5),
        *ptr.offset(6),
        *ptr.offset(7),
        *ptr.offset(8),
        *ptr.offset(9),
        *ptr.offset(10),
        *ptr.offset(11),
        *ptr.offset(12),
        *ptr.offset(13),
        *ptr.offset(14),
        *ptr.offset(15),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1_u16(ptr: *const u16) -> uint16x4_t {
    transmute(u16x4::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1q_u16(ptr: *const u16) -> uint16x8_t {
    transmute(u16x8::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
        *ptr.offset(4),
        *ptr.offset(5),
        *ptr.offset(6),
        *ptr.offset(7),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1_u32(ptr: *const u32) -> uint32x2_t {
    transmute(u32x2::new(*ptr, *ptr.offset(1)))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1q_u32(ptr: *const u32) -> uint32x4_t {
    transmute(u32x4::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1_u64(ptr: *const u64) -> uint64x1_t {
    transmute(u64x1::new(*ptr))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1q_u64(ptr: *const u64) -> uint64x2_t {
    transmute(u64x2::new(*ptr, *ptr.offset(1)))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1_p8(ptr: *const p8) -> poly8x8_t {
    transmute(u8x8::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
        *ptr.offset(4),
        *ptr.offset(5),
        *ptr.offset(6),
        *ptr.offset(7),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1q_p8(ptr: *const p8) -> poly8x16_t {
    transmute(u8x16::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
        *ptr.offset(4),
        *ptr.offset(5),
        *ptr.offset(6),
        *ptr.offset(7),
        *ptr.offset(8),
        *ptr.offset(9),
        *ptr.offset(10),
        *ptr.offset(11),
        *ptr.offset(12),
        *ptr.offset(13),
        *ptr.offset(14),
        *ptr.offset(15),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1_p16(ptr: *const p16) -> poly16x4_t {
    transmute(u16x4::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1q_p16(ptr: *const p16) -> poly16x8_t {
    transmute(u16x8::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
        *ptr.offset(4),
        *ptr.offset(5),
        *ptr.offset(6),
        *ptr.offset(7),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1_f32(ptr: *const f32) -> float32x2_t {
    transmute(f32x2::new(*ptr, *ptr.offset(1)))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1q_f32(ptr: *const f32) -> float32x4_t {
    transmute(f32x4::new(
        *ptr,
        *ptr.offset(1),
        *ptr.offset(2),
        *ptr.offset(3),
    ))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1_f64(ptr: *const f64) -> float64x1_t {
    transmute(f64x1::new(*ptr))
}

/// Load multiple single-element structures to one, two, three, or four registers.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
pub unsafe fn vld1q_f64(ptr: *const f64) -> float64x2_t {
    transmute(f64x2::new(*ptr, *ptr.offset(1)))
}

/// Absolute Value (wrapping).
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(abs))]
pub unsafe fn vabsd_s64(a: i64) -> i64 {
    vabsd_s64_(a)
}

/// Absolute Value (wrapping).
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(abs))]
pub unsafe fn vabs_s64(a: int64x1_t) -> int64x1_t {
    vabs_s64_(a)
}
/// Absolute Value (wrapping).
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(abs))]
pub unsafe fn vabsq_s64(a: int64x2_t) -> int64x2_t {
    vabsq_s64_(a)
}

/// Floating-point absolute value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fabs))]
pub unsafe fn vabs_f64(a: float64x1_t) -> float64x1_t {
    vabs_f64_(a)
}
/// Floating-point absolute value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fabs))]
pub unsafe fn vabsq_f64(a: float64x2_t) -> float64x2_t {
    vabsq_f64_(a)
}

/// Signed saturating Accumulate of Unsigned value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(suqadd))]
pub unsafe fn vuqadd_s8(a: int8x8_t, b: uint8x8_t) -> int8x8_t {
    vuqadd_s8_(a, b)
}
/// Signed saturating Accumulate of Unsigned value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(suqadd))]
pub unsafe fn vuqaddq_s8(a: int8x16_t, b: uint8x16_t) -> int8x16_t {
    vuqaddq_s8_(a, b)
}
/// Signed saturating Accumulate of Unsigned value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(suqadd))]
pub unsafe fn vuqadd_s16(a: int16x4_t, b: uint16x4_t) -> int16x4_t {
    vuqadd_s16_(a, b)
}
/// Signed saturating Accumulate of Unsigned value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(suqadd))]
pub unsafe fn vuqaddq_s16(a: int16x8_t, b: uint16x8_t) -> int16x8_t {
    vuqaddq_s16_(a, b)
}
/// Signed saturating Accumulate of Unsigned value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(suqadd))]
pub unsafe fn vuqadd_s32(a: int32x2_t, b: uint32x2_t) -> int32x2_t {
    vuqadd_s32_(a, b)
}
/// Signed saturating Accumulate of Unsigned value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(suqadd))]
pub unsafe fn vuqaddq_s32(a: int32x4_t, b: uint32x4_t) -> int32x4_t {
    vuqaddq_s32_(a, b)
}
/// Signed saturating Accumulate of Unsigned value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(suqadd))]
pub unsafe fn vuqadd_s64(a: int64x1_t, b: uint64x1_t) -> int64x1_t {
    vuqadd_s64_(a, b)
}
/// Signed saturating Accumulate of Unsigned value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(suqadd))]
pub unsafe fn vuqaddq_s64(a: int64x2_t, b: uint64x2_t) -> int64x2_t {
    vuqaddq_s64_(a, b)
}

/// Unsigned saturating Accumulate of Signed value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usqadd))]
pub unsafe fn vsqadd_u8(a: uint8x8_t, b: int8x8_t) -> uint8x8_t {
    vsqadd_u8_(a, b)
}
/// Unsigned saturating Accumulate of Signed value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usqadd))]
pub unsafe fn vsqaddq_u8(a: uint8x16_t, b: int8x16_t) -> uint8x16_t {
    vsqaddq_u8_(a, b)
}
/// Unsigned saturating Accumulate of Signed value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usqadd))]
pub unsafe fn vsqadd_u16(a: uint16x4_t, b: int16x4_t) -> uint16x4_t {
    vsqadd_u16_(a, b)
}
/// Unsigned saturating Accumulate of Signed value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usqadd))]
pub unsafe fn vsqaddq_u16(a: uint16x8_t, b: int16x8_t) -> uint16x8_t {
    vsqaddq_u16_(a, b)
}
/// Unsigned saturating Accumulate of Signed value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usqadd))]
pub unsafe fn vsqadd_u32(a: uint32x2_t, b: int32x2_t) -> uint32x2_t {
    vsqadd_u32_(a, b)
}
/// Unsigned saturating Accumulate of Signed value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usqadd))]
pub unsafe fn vsqaddq_u32(a: uint32x4_t, b: int32x4_t) -> uint32x4_t {
    vsqaddq_u32_(a, b)
}
/// Unsigned saturating Accumulate of Signed value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usqadd))]
pub unsafe fn vsqadd_u64(a: uint64x1_t, b: int64x1_t) -> uint64x1_t {
    vsqadd_u64_(a, b)
}
/// Unsigned saturating Accumulate of Signed value.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usqadd))]
pub unsafe fn vsqaddq_u64(a: uint64x2_t, b: int64x2_t) -> uint64x2_t {
    vsqaddq_u64_(a, b)
}

/// Add pairwise
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addp))]
pub unsafe fn vpaddq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vpaddq_s16_(a, b)
}
/// Add pairwise
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addp))]
pub unsafe fn vpaddq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    transmute(vpaddq_s16_(transmute(a), transmute(b)))
}
/// Add pairwise
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addp))]
pub unsafe fn vpaddq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vpaddq_s32_(a, b)
}
/// Add pairwise
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addp))]
pub unsafe fn vpaddq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    transmute(vpaddq_s32_(transmute(a), transmute(b)))
}
/// Add pairwise
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addp))]
pub unsafe fn vpaddq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vpaddq_s8_(a, b)
}
/// Add pairwise
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addp))]
pub unsafe fn vpaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    transmute(vpaddq_s8_(transmute(a), transmute(b)))
}
/// Add pairwise
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addp))]
pub unsafe fn vpaddd_s64(a: int64x2_t) -> i64 {
    transmute(vaddvq_u64_(transmute(a)))
}
/// Add pairwise
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addp))]
pub unsafe fn vpaddd_u64(a: uint64x2_t) -> u64 {
    transmute(vaddvq_u64_(transmute(a)))
}

/// Add across vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addv))]
pub unsafe fn vaddv_s16(a: int16x4_t) -> i16 {
    vaddv_s16_(a)
}
/// Add across vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addp))]
pub unsafe fn vaddv_s32(a: int32x2_t) -> i32 {
    vaddv_s32_(a)
}
/// Add across vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addv))]
pub unsafe fn vaddv_s8(a: int8x8_t) -> i8 {
    vaddv_s8_(a)
}
/// Add across vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addv))]
pub unsafe fn vaddv_u16(a: uint16x4_t) -> u16 {
    vaddv_u16_(a)
}
/// Add across vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addp))]
pub unsafe fn vaddv_u32(a: uint32x2_t) -> u32 {
    vaddv_u32_(a)
}
/// Add across vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addv))]
pub unsafe fn vaddv_u8(a: uint8x8_t) -> u8 {
    vaddv_u8_(a)
}
/// Add across vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addv))]
pub unsafe fn vaddvq_s16(a: int16x8_t) -> i16 {
    vaddvq_s16_(a)
}
/// Add across vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addv))]
pub unsafe fn vaddvq_s32(a: int32x4_t) -> i32 {
    vaddvq_s32_(a)
}
/// Add across vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addv))]
pub unsafe fn vaddvq_s8(a: int8x16_t) -> i8 {
    vaddvq_s8_(a)
}
/// Add across vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addv))]
pub unsafe fn vaddvq_u16(a: uint16x8_t) -> u16 {
    vaddvq_u16_(a)
}
/// Add across vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addv))]
pub unsafe fn vaddvq_u32(a: uint32x4_t) -> u32 {
    vaddvq_u32_(a)
}
/// Add across vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addv))]
pub unsafe fn vaddvq_u8(a: uint8x16_t) -> u8 {
    vaddvq_u8_(a)
}
/// Add across vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addp))]
pub unsafe fn vaddvq_s64(a: int64x2_t) -> i64 {
    vaddvq_s64_(a)
}
/// Add across vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(addp))]
pub unsafe fn vaddvq_u64(a: uint64x2_t) -> u64 {
    vaddvq_u64_(a)
}

/// Polynomial multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(pmull))]
pub unsafe fn vmull_p64(a: p64, b: p64) -> p128 {
    transmute(vmull_p64_(transmute(a), transmute(b)))
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
pub unsafe fn vadd_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vadd_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_add(a, b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddd_s64(a: i64, b: i64) -> i64 {
    let a: int64x1_t = transmute(a);
    let b: int64x1_t = transmute(b);
    simd_extract(simd_add(a, b), 0)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddd_u64(a: u64, b: u64) -> u64 {
    let a: uint64x1_t = transmute(a);
    let b: uint64x1_t = transmute(b);
    simd_extract(simd_add(a, b), 0)
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
pub unsafe fn vcombine_f32(low: float32x2_t, high: float32x2_t) -> float32x4_t {
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
pub unsafe fn vcombine_f64(low: float64x1_t, high: float64x1_t) -> float64x2_t {
    simd_shuffle2(low, high, [0, 1])
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl1_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    vqtbl1_s8(vcombine_s8(a, zeroed()), transmute(b))
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl1_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    vqtbl1_u8(vcombine_u8(a, zeroed()), b)
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl1_p8(a: poly8x8_t, b: uint8x8_t) -> poly8x8_t {
    vqtbl1_p8(vcombine_p8(a, zeroed()), b)
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl2_s8(a: int8x8x2_t, b: int8x8_t) -> int8x8_t {
    vqtbl1_s8(vcombine_s8(a.0, a.1), transmute(b))
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
        int8x16x2_t(vcombine_s8(a.0, a.1), vcombine_s8(a.2, zeroed())),
        transmute(b),
    )
}

/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vtbl3_u8(a: uint8x8x3_t, b: uint8x8_t) -> uint8x8_t {
    vqtbl2_u8(
        uint8x16x2_t(vcombine_u8(a.0, a.1), vcombine_u8(a.2, zeroed())),
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
        poly8x16x2_t(vcombine_p8(a.0, a.1), vcombine_p8(a.2, zeroed())),
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
        transmute(b),
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
    let r = vqtbx1_s8(a, vcombine_s8(b, zeroed()), transmute(c));
    let m: int8x8_t = simd_lt(c, transmute(i8x8::splat(8)));
    simd_select(m, r, a)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx1_u8(a: uint8x8_t, b: uint8x8_t, c: uint8x8_t) -> uint8x8_t {
    let r = vqtbx1_u8(a, vcombine_u8(b, zeroed()), c);
    let m: int8x8_t = simd_lt(c, transmute(u8x8::splat(8)));
    simd_select(m, r, a)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx1_p8(a: poly8x8_t, b: poly8x8_t, c: uint8x8_t) -> poly8x8_t {
    let r = vqtbx1_p8(a, vcombine_p8(b, zeroed()), c);
    let m: int8x8_t = simd_lt(c, transmute(u8x8::splat(8)));
    simd_select(m, r, a)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx2_s8(a: int8x8_t, b: int8x8x2_t, c: int8x8_t) -> int8x8_t {
    vqtbx1_s8(a, vcombine_s8(b.0, b.1), transmute(c))
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx2_u8(a: uint8x8_t, b: uint8x8x2_t, c: uint8x8_t) -> uint8x8_t {
    vqtbx1_u8(a, vcombine_u8(b.0, b.1), c)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx2_p8(a: poly8x8_t, b: poly8x8x2_t, c: uint8x8_t) -> poly8x8_t {
    vqtbx1_p8(a, vcombine_p8(b.0, b.1), c)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx3_s8(a: int8x8_t, b: int8x8x3_t, c: int8x8_t) -> int8x8_t {
    let r = vqtbx2_s8(
        a,
        int8x16x2_t(vcombine_s8(b.0, b.1), vcombine_s8(b.2, zeroed())),
        transmute(c),
    );
    let m: int8x8_t = simd_lt(c, transmute(i8x8::splat(24)));
    simd_select(m, r, a)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx3_u8(a: uint8x8_t, b: uint8x8x3_t, c: uint8x8_t) -> uint8x8_t {
    let r = vqtbx2_u8(
        a,
        uint8x16x2_t(vcombine_u8(b.0, b.1), vcombine_u8(b.2, zeroed())),
        c,
    );
    let m: int8x8_t = simd_lt(c, transmute(u8x8::splat(24)));
    simd_select(m, r, a)
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx3_p8(a: poly8x8_t, b: poly8x8x3_t, c: uint8x8_t) -> poly8x8_t {
    let r = vqtbx2_p8(
        a,
        poly8x16x2_t(vcombine_p8(b.0, b.1), vcombine_p8(b.2, zeroed())),
        c,
    );
    let m: int8x8_t = simd_lt(c, transmute(u8x8::splat(24)));
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
        transmute(c),
    )
}

/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vtbx4_u8(a: uint8x8_t, b: uint8x8x4_t, c: uint8x8_t) -> uint8x8_t {
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
pub unsafe fn vtbx4_p8(a: poly8x8_t, b: poly8x8x4_t, c: uint8x8_t) -> poly8x8_t {
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
    transmute(vqtbl1(transmute(t), transmute(idx)))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl1q_u8(t: uint8x16_t, idx: uint8x16_t) -> uint8x16_t {
    transmute(vqtbl1q(transmute(t), transmute(idx)))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl1_p8(t: poly8x16_t, idx: uint8x8_t) -> poly8x8_t {
    transmute(vqtbl1(transmute(t), transmute(idx)))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl1q_p8(t: poly8x16_t, idx: uint8x16_t) -> poly8x16_t {
    transmute(vqtbl1q(transmute(t), transmute(idx)))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx1_s8(a: int8x8_t, t: int8x16_t, idx: uint8x8_t) -> int8x8_t {
    vqtbx1(a, t, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx1q_s8(a: int8x16_t, t: int8x16_t, idx: uint8x16_t) -> int8x16_t {
    vqtbx1q(a, t, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx1_u8(a: uint8x8_t, t: uint8x16_t, idx: uint8x8_t) -> uint8x8_t {
    transmute(vqtbx1(transmute(a), transmute(t), transmute(idx)))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx1q_u8(a: uint8x16_t, t: uint8x16_t, idx: uint8x16_t) -> uint8x16_t {
    transmute(vqtbx1q(transmute(a), transmute(t), transmute(idx)))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx1_p8(a: poly8x8_t, t: poly8x16_t, idx: uint8x8_t) -> poly8x8_t {
    transmute(vqtbx1(transmute(a), transmute(t), transmute(idx)))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx1q_p8(a: poly8x16_t, t: poly8x16_t, idx: uint8x16_t) -> poly8x16_t {
    transmute(vqtbx1q(transmute(a), transmute(t), transmute(idx)))
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
    transmute(vqtbl2(transmute(t.0), transmute(t.1), transmute(idx)))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl2q_u8(t: uint8x16x2_t, idx: uint8x16_t) -> uint8x16_t {
    transmute(vqtbl2q(transmute(t.0), transmute(t.1), transmute(idx)))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl2_p8(t: poly8x16x2_t, idx: uint8x8_t) -> poly8x8_t {
    transmute(vqtbl2(transmute(t.0), transmute(t.1), transmute(idx)))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl2q_p8(t: poly8x16x2_t, idx: uint8x16_t) -> poly8x16_t {
    transmute(vqtbl2q(transmute(t.0), transmute(t.1), transmute(idx)))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx2_s8(a: int8x8_t, t: int8x16x2_t, idx: uint8x8_t) -> int8x8_t {
    vqtbx2(a, t.0, t.1, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx2q_s8(a: int8x16_t, t: int8x16x2_t, idx: uint8x16_t) -> int8x16_t {
    vqtbx2q(a, t.0, t.1, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx2_u8(a: uint8x8_t, t: uint8x16x2_t, idx: uint8x8_t) -> uint8x8_t {
    transmute(vqtbx2(
        transmute(a),
        transmute(t.0),
        transmute(t.1),
        transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx2q_u8(a: uint8x16_t, t: uint8x16x2_t, idx: uint8x16_t) -> uint8x16_t {
    transmute(vqtbx2q(
        transmute(a),
        transmute(t.0),
        transmute(t.1),
        transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx2_p8(a: poly8x8_t, t: poly8x16x2_t, idx: uint8x8_t) -> poly8x8_t {
    transmute(vqtbx2(
        transmute(a),
        transmute(t.0),
        transmute(t.1),
        transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx2q_p8(a: poly8x16_t, t: poly8x16x2_t, idx: uint8x16_t) -> poly8x16_t {
    transmute(vqtbx2q(
        transmute(a),
        transmute(t.0),
        transmute(t.1),
        transmute(idx),
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
    transmute(vqtbl3(
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl3q_u8(t: uint8x16x3_t, idx: uint8x16_t) -> uint8x16_t {
    transmute(vqtbl3q(
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl3_p8(t: poly8x16x3_t, idx: uint8x8_t) -> poly8x8_t {
    transmute(vqtbl3(
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl3q_p8(t: poly8x16x3_t, idx: uint8x16_t) -> poly8x16_t {
    transmute(vqtbl3q(
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx3_s8(a: int8x8_t, t: int8x16x3_t, idx: uint8x8_t) -> int8x8_t {
    vqtbx3(a, t.0, t.1, t.2, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx3q_s8(a: int8x16_t, t: int8x16x3_t, idx: uint8x16_t) -> int8x16_t {
    vqtbx3q(a, t.0, t.1, t.2, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx3_u8(a: uint8x8_t, t: uint8x16x3_t, idx: uint8x8_t) -> uint8x8_t {
    transmute(vqtbx3(
        transmute(a),
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx3q_u8(a: uint8x16_t, t: uint8x16x3_t, idx: uint8x16_t) -> uint8x16_t {
    transmute(vqtbx3q(
        transmute(a),
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx3_p8(a: poly8x8_t, t: poly8x16x3_t, idx: uint8x8_t) -> poly8x8_t {
    transmute(vqtbx3(
        transmute(a),
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx3q_p8(a: poly8x16_t, t: poly8x16x3_t, idx: uint8x16_t) -> poly8x16_t {
    transmute(vqtbx3q(
        transmute(a),
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(idx),
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
    transmute(vqtbl4(
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(t.3),
        transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl4q_u8(t: uint8x16x4_t, idx: uint8x16_t) -> uint8x16_t {
    transmute(vqtbl4q(
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(t.3),
        transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl4_p8(t: poly8x16x4_t, idx: uint8x8_t) -> poly8x8_t {
    transmute(vqtbl4(
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(t.3),
        transmute(idx),
    ))
}
/// Table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbl))]
pub unsafe fn vqtbl4q_p8(t: poly8x16x4_t, idx: uint8x16_t) -> poly8x16_t {
    transmute(vqtbl4q(
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(t.3),
        transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx4_s8(a: int8x8_t, t: int8x16x4_t, idx: uint8x8_t) -> int8x8_t {
    vqtbx4(a, t.0, t.1, t.2, t.3, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx4q_s8(a: int8x16_t, t: int8x16x4_t, idx: uint8x16_t) -> int8x16_t {
    vqtbx4q(a, t.0, t.1, t.2, t.3, idx)
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx4_u8(a: uint8x8_t, t: uint8x16x4_t, idx: uint8x8_t) -> uint8x8_t {
    transmute(vqtbx4(
        transmute(a),
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(t.3),
        transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx4q_u8(a: uint8x16_t, t: uint8x16x4_t, idx: uint8x16_t) -> uint8x16_t {
    transmute(vqtbx4q(
        transmute(a),
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(t.3),
        transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx4_p8(a: poly8x8_t, t: poly8x16x4_t, idx: uint8x8_t) -> poly8x8_t {
    transmute(vqtbx4(
        transmute(a),
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(t.3),
        transmute(idx),
    ))
}
/// Extended table look-up
#[inline]
#[cfg(target_endian = "little")]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tbx))]
pub unsafe fn vqtbx4q_p8(a: poly8x16_t, t: poly8x16x4_t, idx: uint8x16_t) -> poly8x16_t {
    transmute(vqtbx4q(
        transmute(a),
        transmute(t.0),
        transmute(t.1),
        transmute(t.2),
        transmute(t.3),
        transmute(idx),
    ))
}

#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs))]
pub unsafe fn vcvtq_s32_f32(a: float32x4_t) -> int32x4_t {
    vcvtq_s32_f32_(a)
}

/// Floating-point Convert to Unsigned fixed-point, rounding toward Zero (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu))]
pub unsafe fn vcvtq_u32_f32(a: float32x4_t) -> uint32x4_t {
    vcvtq_u32_f32_(a)
}

/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_s8<const N: i32>(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    static_assert_imm3!(N);
    vsli_n_s8_(a, b, N)
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_s8<const N: i32>(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    static_assert_imm3!(N);
    vsliq_n_s8_(a, b, N)
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_s16<const N: i32>(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    static_assert_imm4!(N);
    vsli_n_s16_(a, b, N)
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_s16<const N: i32>(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    static_assert_imm4!(N);
    vsliq_n_s16_(a, b, N)
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_s32<const N: i32>(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    static_assert!(N: i32 where N >= 0 && N <= 31);
    vsli_n_s32_(a, b, N)
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_s32<const N: i32>(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    static_assert!(N: i32 where N >= 0 && N <= 31);
    vsliq_n_s32_(a, b, N)
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_s64<const N: i32>(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    static_assert!(N: i32 where N >= 0 && N <= 63);
    vsli_n_s64_(a, b, N)
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_s64<const N: i32>(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    static_assert!(N: i32 where N >= 0 && N <= 63);
    vsliq_n_s64_(a, b, N)
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_u8<const N: i32>(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    static_assert_imm3!(N);
    transmute(vsli_n_s8_(transmute(a), transmute(b), N))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_u8<const N: i32>(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    static_assert_imm3!(N);
    transmute(vsliq_n_s8_(transmute(a), transmute(b), N))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_u16<const N: i32>(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    static_assert_imm4!(N);
    transmute(vsli_n_s16_(transmute(a), transmute(b), N))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_u16<const N: i32>(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    static_assert_imm4!(N);
    transmute(vsliq_n_s16_(transmute(a), transmute(b), N))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_u32<const N: i32>(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    static_assert!(N: i32 where N >= 0 && N <= 31);
    transmute(vsli_n_s32_(transmute(a), transmute(b), N))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_u32<const N: i32>(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    static_assert!(N: i32 where N >= 0 && N <= 31);
    transmute(vsliq_n_s32_(transmute(a), transmute(b), N))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_u64<const N: i32>(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    static_assert!(N: i32 where N >= 0 && N <= 63);
    transmute(vsli_n_s64_(transmute(a), transmute(b), N))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_u64<const N: i32>(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    static_assert!(N: i32 where N >= 0 && N <= 63);
    transmute(vsliq_n_s64_(transmute(a), transmute(b), N))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_p8<const N: i32>(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    static_assert_imm3!(N);
    transmute(vsli_n_s8_(transmute(a), transmute(b), N))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_p8<const N: i32>(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    static_assert_imm3!(N);
    transmute(vsliq_n_s8_(transmute(a), transmute(b), N))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_p16<const N: i32>(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    static_assert_imm4!(N);
    transmute(vsli_n_s16_(transmute(a), transmute(b), N))
}
/// Shift Left and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sli, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_p16<const N: i32>(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    static_assert_imm4!(N);
    transmute(vsliq_n_s16_(transmute(a), transmute(b), N))
}

/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_s8<const N: i32>(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    static_assert!(N: i32 where N >= 1 && N <= 8);
    vsri_n_s8_(a, b, N)
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_s8<const N: i32>(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    static_assert!(N: i32 where N >= 1 && N <= 8);
    vsriq_n_s8_(a, b, N)
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_s16<const N: i32>(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    static_assert!(N: i32 where N >= 1 && N <= 16);
    vsri_n_s16_(a, b, N)
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_s16<const N: i32>(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    static_assert!(N: i32 where N >= 1 && N <= 16);
    vsriq_n_s16_(a, b, N)
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_s32<const N: i32>(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    static_assert!(N: i32 where N >= 1 && N <= 32);
    vsri_n_s32_(a, b, N)
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_s32<const N: i32>(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    static_assert!(N: i32 where N >= 1 && N <= 32);
    vsriq_n_s32_(a, b, N)
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_s64<const N: i32>(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    static_assert!(N: i32 where N >= 1 && N <= 64);
    vsri_n_s64_(a, b, N)
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_s64<const N: i32>(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    static_assert!(N: i32 where N >= 1 && N <= 64);
    vsriq_n_s64_(a, b, N)
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_u8<const N: i32>(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    static_assert!(N: i32 where N >= 1 && N <= 8);
    transmute(vsri_n_s8_(transmute(a), transmute(b), N))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_u8<const N: i32>(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    static_assert!(N: i32 where N >= 1 && N <= 8);
    transmute(vsriq_n_s8_(transmute(a), transmute(b), N))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_u16<const N: i32>(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    static_assert!(N: i32 where N >= 1 && N <= 16);
    transmute(vsri_n_s16_(transmute(a), transmute(b), N))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_u16<const N: i32>(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    static_assert!(N: i32 where N >= 1 && N <= 16);
    transmute(vsriq_n_s16_(transmute(a), transmute(b), N))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_u32<const N: i32>(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    static_assert!(N: i32 where N >= 1 && N <= 32);
    transmute(vsri_n_s32_(transmute(a), transmute(b), N))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_u32<const N: i32>(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    static_assert!(N: i32 where N >= 1 && N <= 32);
    transmute(vsriq_n_s32_(transmute(a), transmute(b), N))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_u64<const N: i32>(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    static_assert!(N: i32 where N >= 1 && N <= 64);
    transmute(vsri_n_s64_(transmute(a), transmute(b), N))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_u64<const N: i32>(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    static_assert!(N: i32 where N >= 1 && N <= 64);
    transmute(vsriq_n_s64_(transmute(a), transmute(b), N))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_p8<const N: i32>(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    static_assert!(N: i32 where N >= 1 && N <= 8);
    transmute(vsri_n_s8_(transmute(a), transmute(b), N))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_p8<const N: i32>(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    static_assert!(N: i32 where N >= 1 && N <= 8);
    transmute(vsriq_n_s8_(transmute(a), transmute(b), N))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_p16<const N: i32>(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    static_assert!(N: i32 where N >= 1 && N <= 16);
    transmute(vsri_n_s16_(transmute(a), transmute(b), N))
}
/// Shift Right and Insert (immediate)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sri, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_p16<const N: i32>(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    static_assert!(N: i32 where N >= 1 && N <= 16);
    transmute(vsriq_n_s16_(transmute(a), transmute(b), N))
}

#[cfg(test)]
mod tests {
    use crate::core_arch::aarch64::test_support::*;
    use crate::core_arch::arm::test_support::*;
    use crate::core_arch::{aarch64::neon::*, aarch64::*, simd::*};
    use std::mem::transmute;
    use stdarch_test::simd_test;

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_s32_f32() {
        let f = f32x4::new(-1., 2., 3., 4.);
        let e = i32x4::new(-1, 2, 3, 4);
        let r: i32x4 = transmute(vcvtq_s32_f32(transmute(f)));
        assert_eq!(r, e);

        let f = f32x4::new(10e37, 2., 3., 4.);
        let e = i32x4::new(0x7fffffff, 2, 3, 4);
        let r: i32x4 = transmute(vcvtq_s32_f32(transmute(f)));
        assert_eq!(r, e);

        let f = f32x4::new(-10e37, 2., 3., 4.);
        let e = i32x4::new(-0x80000000, 2, 3, 4);
        let r: i32x4 = transmute(vcvtq_s32_f32(transmute(f)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_u32_f32() {
        let f = f32x4::new(1., 2., 3., 4.);
        let e = u32x4::new(1, 2, 3, 4);
        let r: u32x4 = transmute(vcvtq_u32_f32(transmute(f)));
        assert_eq!(r, e);

        let f = f32x4::new(-1., 2., 3., 4.);
        let e = u32x4::new(0, 2, 3, 4);
        let r: u32x4 = transmute(vcvtq_u32_f32(transmute(f)));
        assert_eq!(r, e);

        let f = f32x4::new(10e37, 2., 3., 4.);
        let e = u32x4::new(0xffffffff, 2, 3, 4);
        let r: u32x4 = transmute(vcvtq_u32_f32(transmute(f)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuqadd_s8() {
        let a = i8x8::new(i8::MIN, -3, -2, -1, 0, 1, 2, i8::MAX);
        let b = u8x8::new(u8::MAX, 1, 2, 3, 4, 5, 6, 7);
        let e = i8x8::new(i8::MAX, -2, 0, 2, 4, 6, 8, i8::MAX);
        let r: i8x8 = transmute(vuqadd_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vuqaddq_s8() {
        let a = i8x16::new(
            i8::MIN,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            i8::MAX,
        );
        let b = u8x16::new(u8::MAX, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e = i8x16::new(
            i8::MAX,
            -6,
            -4,
            -2,
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            i8::MAX,
        );
        let r: i8x16 = transmute(vuqaddq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vuqadd_s16() {
        let a = i16x4::new(i16::MIN, -1, 0, i16::MAX);
        let b = u16x4::new(u16::MAX, 1, 2, 3);
        let e = i16x4::new(i16::MAX, 0, 2, i16::MAX);
        let r: i16x4 = transmute(vuqadd_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vuqaddq_s16() {
        let a = i16x8::new(i16::MIN, -3, -2, -1, 0, 1, 2, i16::MAX);
        let b = u16x8::new(u16::MAX, 1, 2, 3, 4, 5, 6, 7);
        let e = i16x8::new(i16::MAX, -2, 0, 2, 4, 6, 8, i16::MAX);
        let r: i16x8 = transmute(vuqaddq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vuqadd_s32() {
        let a = i32x2::new(i32::MIN, i32::MAX);
        let b = u32x2::new(u32::MAX, 1);
        let e = i32x2::new(i32::MAX, i32::MAX);
        let r: i32x2 = transmute(vuqadd_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vuqaddq_s32() {
        let a = i32x4::new(i32::MIN, -1, 0, i32::MAX);
        let b = u32x4::new(u32::MAX, 1, 2, 3);
        let e = i32x4::new(i32::MAX, 0, 2, i32::MAX);
        let r: i32x4 = transmute(vuqaddq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vuqadd_s64() {
        let a = i64x1::new(i64::MIN);
        let b = u64x1::new(u64::MAX);
        let e = i64x1::new(i64::MAX);
        let r: i64x1 = transmute(vuqadd_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vuqaddq_s64() {
        let a = i64x2::new(i64::MIN, i64::MAX);
        let b = u64x2::new(u64::MAX, 1);
        let e = i64x2::new(i64::MAX, i64::MAX);
        let r: i64x2 = transmute(vuqaddq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsqadd_u8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, u8::MAX);
        let b = i8x8::new(i8::MIN, -3, -2, -1, 0, 1, 2, 3);
        let e = u8x8::new(0, 0, 0, 2, 4, 6, 8, u8::MAX);
        let r: u8x8 = transmute(vsqadd_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsqaddq_u8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, u8::MAX);
        let b = i8x16::new(i8::MIN, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        let e = u8x16::new(0, 0, 0, 0, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, u8::MAX);
        let r: u8x16 = transmute(vsqaddq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsqadd_u16() {
        let a = u16x4::new(0, 1, 2, u16::MAX);
        let b = i16x4::new(i16::MIN, -1, 0, 1);
        let e = u16x4::new(0, 0, 2, u16::MAX);
        let r: u16x4 = transmute(vsqadd_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsqaddq_u16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, u16::MAX);
        let b = i16x8::new(i16::MIN, -3, -2, -1, 0, 1, 2, 3);
        let e = u16x8::new(0, 0, 0, 2, 4, 6, 8, u16::MAX);
        let r: u16x8 = transmute(vsqaddq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsqadd_u32() {
        let a = u32x2::new(0, u32::MAX);
        let b = i32x2::new(i32::MIN, 1);
        let e = u32x2::new(0, u32::MAX);
        let r: u32x2 = transmute(vsqadd_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsqaddq_u32() {
        let a = u32x4::new(0, 1, 2, u32::MAX);
        let b = i32x4::new(i32::MIN, -1, 0, 1);
        let e = u32x4::new(0, 0, 2, u32::MAX);
        let r: u32x4 = transmute(vsqaddq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsqadd_u64() {
        let a = u64x1::new(0);
        let b = i64x1::new(i64::MIN);
        let e = u64x1::new(0);
        let r: u64x1 = transmute(vsqadd_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsqaddq_u64() {
        let a = u64x2::new(0, u64::MAX);
        let b = i64x2::new(i64::MIN, 1);
        let e = u64x2::new(0, u64::MAX);
        let r: u64x2 = transmute(vsqaddq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpaddq_s16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(0, -1, -2, -3, -4, -5, -6, -7);
        let r: i16x8 = transmute(vpaddq_s16(transmute(a), transmute(b)));
        let e = i16x8::new(3, 7, 11, 15, -1, -5, -9, -13);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vpaddq_s32() {
        let a = i32x4::new(1, 2, 3, 4);
        let b = i32x4::new(0, -1, -2, -3);
        let r: i32x4 = transmute(vpaddq_s32(transmute(a), transmute(b)));
        let e = i32x4::new(3, 7, -1, -5);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vpaddq_s8() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = i8x16::new(
            0, -1, -2, -3, -4, -5, -6, -7, -8, -8, -10, -11, -12, -13, -14, -15,
        );
        let r: i8x16 = transmute(vpaddq_s8(transmute(a), transmute(b)));
        let e = i8x16::new(
            3, 7, 11, 15, 19, 23, 27, 31, -1, -5, -9, -13, -16, -21, -25, -29,
        );
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vpaddq_u16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = u16x8::new(17, 18, 19, 20, 20, 21, 22, 23);
        let r: u16x8 = transmute(vpaddq_u16(transmute(a), transmute(b)));
        let e = u16x8::new(1, 5, 9, 13, 35, 39, 41, 45);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vpaddq_u32() {
        let a = u32x4::new(0, 1, 2, 3);
        let b = u32x4::new(17, 18, 19, 20);
        let r: u32x4 = transmute(vpaddq_u32(transmute(a), transmute(b)));
        let e = u32x4::new(1, 5, 35, 39);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vpaddq_u8() {
        let a = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = i8x16::new(
            17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 29, 29, 30, 31,
        );
        let r = i8x16(1, 5, 9, 13, 17, 21, 25, 29, 35, 39, 41, 45, 49, 53, 58, 61);
        let e: i8x16 = transmute(vpaddq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vpaddd_s64() {
        let a = i64x2::new(2, -3);
        let r: i64 = transmute(vpaddd_s64(transmute(a)));
        let e = -1_i64;
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vpaddd_u64() {
        let a = i64x2::new(2, 3);
        let r: u64 = transmute(vpaddd_u64(transmute(a)));
        let e = 5_u64;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_p64() {
        // FIXME: I've a hard time writing a test for this as the documentation
        // from arm is a bit thin as to waht exactly it does
        let a: i64 = 8;
        let b: i64 = 7;
        let e: i128 = 56;
        let r: i128 = transmute(vmull_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        /*
        let a: i64 = 5;
        let b: i64 = 5;
        let e: i128 = 25;
        let r: i128 = transmute(vmull_p64(a, b));

        assert_eq!(r, e);
        let a: i64 = 6;
        let b: i64 = 6;
        let e: i128 = 36;
        let r: i128 = transmute(vmull_p64(a, b));
        assert_eq!(r, e);

        let a: i64 = 7;
        let b: i64 = 6;
        let e: i128 = 42;
        let r: i128 = transmute(vmull_p64(a, b));
        assert_eq!(r, e);
        */
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_f64() {
        let a = 1.;
        let b = 8.;
        let e = 9.;
        let r: f64 = transmute(vadd_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_f64() {
        let a = f64x2::new(1., 2.);
        let b = f64x2::new(8., 7.);
        let e = f64x2::new(9., 9.);
        let r: f64x2 = transmute(vaddq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_s64() {
        let a = 1_i64;
        let b = 8_i64;
        let e = 9_i64;
        let r: i64 = transmute(vadd_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_u64() {
        let a = 1_u64;
        let b = 8_u64;
        let e = 9_u64;
        let r: u64 = transmute(vadd_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddd_s64() {
        let a = 1_i64;
        let b = 8_i64;
        let e = 9_i64;
        let r: i64 = transmute(vaddd_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddd_u64() {
        let a = 1_u64;
        let b = 8_u64;
        let e = 9_u64;
        let r: u64 = transmute(vaddd_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxv_s8() {
        let r = vmaxv_s8(transmute(i8x8::new(1, 2, 3, 4, -8, 6, 7, 5)));
        assert_eq!(r, 7_i8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_s8() {
        #[rustfmt::skip]
        let r = vmaxvq_s8(transmute(i8x16::new(
            1, 2, 3, 4,
            -16, 6, 7, 5,
            8, 1, 1, 1,
            1, 1, 1, 1,
        )));
        assert_eq!(r, 8_i8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxv_s16() {
        let r = vmaxv_s16(transmute(i16x4::new(1, 2, -4, 3)));
        assert_eq!(r, 3_i16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_s16() {
        let r = vmaxvq_s16(transmute(i16x8::new(1, 2, 7, 4, -16, 6, 7, 5)));
        assert_eq!(r, 7_i16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxv_s32() {
        let r = vmaxv_s32(transmute(i32x2::new(1, -4)));
        assert_eq!(r, 1_i32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_s32() {
        let r = vmaxvq_s32(transmute(i32x4::new(1, 2, -32, 4)));
        assert_eq!(r, 4_i32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxv_u8() {
        let r = vmaxv_u8(transmute(u8x8::new(1, 2, 3, 4, 8, 6, 7, 5)));
        assert_eq!(r, 8_u8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_u8() {
        #[rustfmt::skip]
        let r = vmaxvq_u8(transmute(u8x16::new(
            1, 2, 3, 4,
            16, 6, 7, 5,
            8, 1, 1, 1,
            1, 1, 1, 1,
        )));
        assert_eq!(r, 16_u8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxv_u16() {
        let r = vmaxv_u16(transmute(u16x4::new(1, 2, 4, 3)));
        assert_eq!(r, 4_u16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_u16() {
        let r = vmaxvq_u16(transmute(u16x8::new(1, 2, 7, 4, 16, 6, 7, 5)));
        assert_eq!(r, 16_u16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxv_u32() {
        let r = vmaxv_u32(transmute(u32x2::new(1, 4)));
        assert_eq!(r, 4_u32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_u32() {
        let r = vmaxvq_u32(transmute(u32x4::new(1, 2, 32, 4)));
        assert_eq!(r, 32_u32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxv_f32() {
        let r = vmaxv_f32(transmute(f32x2::new(1., 4.)));
        assert_eq!(r, 4_f32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_f32() {
        let r = vmaxvq_f32(transmute(f32x4::new(1., 2., 32., 4.)));
        assert_eq!(r, 32_f32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxvq_f64() {
        let r = vmaxvq_f64(transmute(f64x2::new(1., 4.)));
        assert_eq!(r, 4_f64);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminv_s8() {
        let r = vminv_s8(transmute(i8x8::new(1, 2, 3, 4, -8, 6, 7, 5)));
        assert_eq!(r, -8_i8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_s8() {
        #[rustfmt::skip]
        let r = vminvq_s8(transmute(i8x16::new(
            1, 2, 3, 4,
            -16, 6, 7, 5,
            8, 1, 1, 1,
            1, 1, 1, 1,
        )));
        assert_eq!(r, -16_i8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminv_s16() {
        let r = vminv_s16(transmute(i16x4::new(1, 2, -4, 3)));
        assert_eq!(r, -4_i16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_s16() {
        let r = vminvq_s16(transmute(i16x8::new(1, 2, 7, 4, -16, 6, 7, 5)));
        assert_eq!(r, -16_i16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminv_s32() {
        let r = vminv_s32(transmute(i32x2::new(1, -4)));
        assert_eq!(r, -4_i32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_s32() {
        let r = vminvq_s32(transmute(i32x4::new(1, 2, -32, 4)));
        assert_eq!(r, -32_i32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminv_u8() {
        let r = vminv_u8(transmute(u8x8::new(1, 2, 3, 4, 8, 6, 7, 5)));
        assert_eq!(r, 1_u8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_u8() {
        #[rustfmt::skip]
        let r = vminvq_u8(transmute(u8x16::new(
            1, 2, 3, 4,
            16, 6, 7, 5,
            8, 1, 1, 1,
            1, 1, 1, 1,
        )));
        assert_eq!(r, 1_u8);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminv_u16() {
        let r = vminv_u16(transmute(u16x4::new(1, 2, 4, 3)));
        assert_eq!(r, 1_u16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_u16() {
        let r = vminvq_u16(transmute(u16x8::new(1, 2, 7, 4, 16, 6, 7, 5)));
        assert_eq!(r, 1_u16);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminv_u32() {
        let r = vminv_u32(transmute(u32x2::new(1, 4)));
        assert_eq!(r, 1_u32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_u32() {
        let r = vminvq_u32(transmute(u32x4::new(1, 2, 32, 4)));
        assert_eq!(r, 1_u32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminv_f32() {
        let r = vminv_f32(transmute(f32x2::new(1., 4.)));
        assert_eq!(r, 1_f32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_f32() {
        let r = vminvq_f32(transmute(f32x4::new(1., 2., 32., 4.)));
        assert_eq!(r, 1_f32);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminvq_f64() {
        let r = vminvq_f64(transmute(f64x2::new(1., 4.)));
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
        let r: i8x16 = transmute(vpminq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpminq_s16() {
        let a = i16x8::new(1, -2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = i16x8::new(-2, 3, 5, 7, 0, 2, 4, 6);
        let r: i16x8 = transmute(vpminq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpminq_s32() {
        let a = i32x4::new(1, -2, 3, 4);
        let b = i32x4::new(0, 3, 2, 5);
        let e = i32x4::new(-2, 3, 0, 2);
        let r: i32x4 = transmute(vpminq_s32(transmute(a), transmute(b)));
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
        let r: u8x16 = transmute(vpminq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpminq_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u16x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = u16x8::new(1, 3, 5, 7, 0, 2, 4, 6);
        let r: u16x8 = transmute(vpminq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpminq_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let b = u32x4::new(0, 3, 2, 5);
        let e = u32x4::new(1, 3, 0, 2);
        let r: u32x4 = transmute(vpminq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmin_f32() {
        let a = f32x4::new(1., -2., 3., 4.);
        let b = f32x4::new(0., 3., 2., 5.);
        let e = f32x4::new(-2., 3., 0., 2.);
        let r: f32x4 = transmute(vpminq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmin_f64() {
        let a = f64x2::new(1., -2.);
        let b = f64x2::new(0., 3.);
        let e = f64x2::new(-2., 0.);
        let r: f64x2 = transmute(vpminq_f64(transmute(a), transmute(b)));
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
        let r: i8x16 = transmute(vpmaxq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmaxq_s16() {
        let a = i16x8::new(1, -2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = i16x8::new(1, 4, 6, 8, 3, 5, 7, 9);
        let r: i16x8 = transmute(vpmaxq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmaxq_s32() {
        let a = i32x4::new(1, -2, 3, 4);
        let b = i32x4::new(0, 3, 2, 5);
        let e = i32x4::new(1, 4, 3, 5);
        let r: i32x4 = transmute(vpmaxq_s32(transmute(a), transmute(b)));
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
        let r: u8x16 = transmute(vpmaxq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmaxq_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u16x8::new(0, 3, 2, 5, 4, 7, 6, 9);
        let e = u16x8::new(2, 4, 6, 8, 3, 5, 7, 9);
        let r: u16x8 = transmute(vpmaxq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmaxq_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let b = u32x4::new(0, 3, 2, 5);
        let e = u32x4::new(2, 4, 3, 5);
        let r: u32x4 = transmute(vpmaxq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmax_f32() {
        let a = f32x4::new(1., -2., 3., 4.);
        let b = f32x4::new(0., 3., 2., 5.);
        let e = f32x4::new(1., 4., 3., 5.);
        let r: f32x4 = transmute(vpmaxq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmax_f64() {
        let a = f64x2::new(1., -2.);
        let b = f64x2::new(0., 3.);
        let e = f64x2::new(1., 3.);
        let r: f64x2 = transmute(vpmaxq_f64(transmute(a), transmute(b)));
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
                let c = $fn_id(transmute(a), transmute(b));
                let mut d = e;
                d = transmute(c);
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

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u64() {
        test_cmp_u64(
            |i, j| vceq_u64(i, j),
            |a: u64, b: u64| -> u64 {
                if a == b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u64() {
        testq_cmp_u64(
            |i, j| vceqq_u64(i, j),
            |a: u64, b: u64| -> u64 {
                if a == b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s64() {
        test_cmp_s64(
            |i, j| vceq_s64(i, j),
            |a: i64, b: i64| -> u64 {
                if a == b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s64() {
        testq_cmp_s64(
            |i, j| vceqq_s64(i, j),
            |a: i64, b: i64| -> u64 {
                if a == b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_p64() {
        test_cmp_p64(
            |i, j| vceq_p64(i, j),
            |a: u64, b: u64| -> u64 {
                if a == b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_p64() {
        testq_cmp_p64(
            |i, j| vceqq_p64(i, j),
            |a: u64, b: u64| -> u64 {
                if a == b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_f64() {
        test_cmp_f64(
            |i, j| vceq_f64(i, j),
            |a: f64, b: f64| -> u64 {
                if a == b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_f64() {
        testq_cmp_f64(
            |i, j| vceqq_f64(i, j),
            |a: f64, b: f64| -> u64 {
                if a == b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s64() {
        test_cmp_s64(
            |i, j| vcgt_s64(i, j),
            |a: i64, b: i64| -> u64 {
                if a > b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s64() {
        testq_cmp_s64(
            |i, j| vcgtq_s64(i, j),
            |a: i64, b: i64| -> u64 {
                if a > b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u64() {
        test_cmp_u64(
            |i, j| vcgt_u64(i, j),
            |a: u64, b: u64| -> u64 {
                if a > b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u64() {
        testq_cmp_u64(
            |i, j| vcgtq_u64(i, j),
            |a: u64, b: u64| -> u64 {
                if a > b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_f64() {
        test_cmp_f64(
            |i, j| vcgt_f64(i, j),
            |a: f64, b: f64| -> u64 {
                if a > b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_f64() {
        testq_cmp_f64(
            |i, j| vcgtq_f64(i, j),
            |a: f64, b: f64| -> u64 {
                if a > b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s64() {
        test_cmp_s64(
            |i, j| vclt_s64(i, j),
            |a: i64, b: i64| -> u64 {
                if a < b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s64() {
        testq_cmp_s64(
            |i, j| vcltq_s64(i, j),
            |a: i64, b: i64| -> u64 {
                if a < b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u64() {
        test_cmp_u64(
            |i, j| vclt_u64(i, j),
            |a: u64, b: u64| -> u64 {
                if a < b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u64() {
        testq_cmp_u64(
            |i, j| vcltq_u64(i, j),
            |a: u64, b: u64| -> u64 {
                if a < b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vltq_f64() {
        test_cmp_f64(
            |i, j| vclt_f64(i, j),
            |a: f64, b: f64| -> u64 {
                if a < b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_f64() {
        testq_cmp_f64(
            |i, j| vcltq_f64(i, j),
            |a: f64, b: f64| -> u64 {
                if a < b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s64() {
        test_cmp_s64(
            |i, j| vcle_s64(i, j),
            |a: i64, b: i64| -> u64 {
                if a <= b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s64() {
        testq_cmp_s64(
            |i, j| vcleq_s64(i, j),
            |a: i64, b: i64| -> u64 {
                if a <= b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u64() {
        test_cmp_u64(
            |i, j| vcle_u64(i, j),
            |a: u64, b: u64| -> u64 {
                if a <= b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u64() {
        testq_cmp_u64(
            |i, j| vcleq_u64(i, j),
            |a: u64, b: u64| -> u64 {
                if a <= b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vleq_f64() {
        test_cmp_f64(
            |i, j| vcle_f64(i, j),
            |a: f64, b: f64| -> u64 {
                if a <= b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_f64() {
        testq_cmp_f64(
            |i, j| vcleq_f64(i, j),
            |a: f64, b: f64| -> u64 {
                if a <= b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s64() {
        test_cmp_s64(
            |i, j| vcge_s64(i, j),
            |a: i64, b: i64| -> u64 {
                if a >= b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s64() {
        testq_cmp_s64(
            |i, j| vcgeq_s64(i, j),
            |a: i64, b: i64| -> u64 {
                if a >= b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u64() {
        test_cmp_u64(
            |i, j| vcge_u64(i, j),
            |a: u64, b: u64| -> u64 {
                if a >= b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u64() {
        testq_cmp_u64(
            |i, j| vcgeq_u64(i, j),
            |a: u64, b: u64| -> u64 {
                if a >= b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgeq_f64() {
        test_cmp_f64(
            |i, j| vcge_f64(i, j),
            |a: f64, b: f64| -> u64 {
                if a >= b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_f64() {
        testq_cmp_f64(
            |i, j| vcgeq_f64(i, j),
            |a: f64, b: f64| -> u64 {
                if a >= b {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_f64() {
        test_ari_f64(|i, j| vmul_f64(i, j), |a: f64, b: f64| -> f64 { a * b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_f64() {
        testq_ari_f64(|i, j| vmulq_f64(i, j), |a: f64, b: f64| -> f64 { a * b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_f64() {
        test_ari_f64(|i, j| vsub_f64(i, j), |a: f64, b: f64| -> f64 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_f64() {
        testq_ari_f64(|i, j| vsubq_f64(i, j), |a: f64, b: f64| -> f64 { a - b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabsd_s64() {
        assert_eq!(vabsd_s64(-1), 1);
        assert_eq!(vabsd_s64(0), 0);
        assert_eq!(vabsd_s64(1), 1);
        assert_eq!(vabsd_s64(i64::MIN), i64::MIN);
        assert_eq!(vabsd_s64(i64::MIN + 1), i64::MAX);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vabs_s64() {
        let a = i64x1::new(i64::MIN);
        let r: i64x1 = transmute(vabs_s64(transmute(a)));
        let e = i64x1::new(i64::MIN);
        assert_eq!(r, e);
        let a = i64x1::new(i64::MIN + 1);
        let r: i64x1 = transmute(vabs_s64(transmute(a)));
        let e = i64x1::new(i64::MAX);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vabsq_s64() {
        let a = i64x2::new(i64::MIN, i64::MIN + 1);
        let r: i64x2 = transmute(vabsq_s64(transmute(a)));
        let e = i64x2::new(i64::MIN, i64::MAX);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vabs_f64() {
        let a = f64x1::new(f64::MIN);
        let r: f64x1 = transmute(vabs_f64(transmute(a)));
        let e = f64x1::new(f64::MAX);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vabsq_f64() {
        let a = f64x2::new(f64::MIN, -4.2);
        let r: f64x2 = transmute(vabsq_f64(transmute(a)));
        let e = f64x2::new(f64::MAX, 4.2);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddv_s16() {
        let a = i16x4::new(1, 2, 3, -4);
        let r: i16 = transmute(vaddv_s16(transmute(a)));
        let e = 2_i16;
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddv_u16() {
        let a = u16x4::new(1, 2, 3, 4);
        let r: u16 = transmute(vaddv_u16(transmute(a)));
        let e = 10_u16;
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddv_s32() {
        let a = i32x2::new(1, -2);
        let r: i32 = transmute(vaddv_s32(transmute(a)));
        let e = -1_i32;
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddv_u32() {
        let a = u32x2::new(1, 2);
        let r: u32 = transmute(vaddv_u32(transmute(a)));
        let e = 3_u32;
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddv_s8() {
        let a = i8x8::new(1, 2, 3, 4, 5, 6, 7, -8);
        let r: i8 = transmute(vaddv_s8(transmute(a)));
        let e = 20_i8;
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddv_u8() {
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u8 = transmute(vaddv_u8(transmute(a)));
        let e = 36_u8;
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddvq_s16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, -8);
        let r: i16 = transmute(vaddvq_s16(transmute(a)));
        let e = 20_i16;
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddvq_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u16 = transmute(vaddvq_u16(transmute(a)));
        let e = 36_u16;
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddvq_s32() {
        let a = i32x4::new(1, 2, 3, -4);
        let r: i32 = transmute(vaddvq_s32(transmute(a)));
        let e = 2_i32;
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddvq_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let r: u32 = transmute(vaddvq_u32(transmute(a)));
        let e = 10_u32;
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddvq_s8() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -16);
        let r: i8 = transmute(vaddvq_s8(transmute(a)));
        let e = 104_i8;
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddvq_u8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: u8 = transmute(vaddvq_u8(transmute(a)));
        let e = 136_u8;
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddvq_s64() {
        let a = i64x2::new(1, -2);
        let r: i64 = transmute(vaddvq_s64(transmute(a)));
        let e = -1_i64;
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddvq_u64() {
        let a = u64x2::new(1, 2);
        let r: u64 = transmute(vaddvq_u64(transmute(a)));
        let e = 3_u64;
        assert_eq!(r, e);
    }
}

#[cfg(test)]
#[cfg(target_endian = "little")]
#[path = "../../arm/neon/table_lookup_tests.rs"]
mod table_lookup_tests;

#[cfg(test)]
#[path = "../../arm/neon/shift_and_insert_tests.rs"]
mod shift_and_insert_tests;

#[cfg(test)]
#[path = "../../arm/neon/load_tests.rs"]
mod load_tests;
