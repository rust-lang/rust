// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `crates/stdarch-gen/neon.spec` and run the following command to re-generate this file:
//
// ```
// OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen -- crates/stdarch-gen/neon.spec
// ```
use super::*;
#[cfg(test)]
use stdarch_test::assert_instr;

/// Absolute difference between the arguments of Floating
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fabd))]
pub unsafe fn vabd_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fabd.v1f64")]
        fn vabd_f64_(a: float64x1_t, b: float64x1_t) -> float64x1_t;
    }
    vabd_f64_(a, b)
}

/// Absolute difference between the arguments of Floating
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fabd))]
pub unsafe fn vabdq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fabd.v2f64")]
        fn vabdq_f64_(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    vabdq_f64_(a, b)
}

/// Unsigned Absolute difference Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uabdl))]
pub unsafe fn vabdl_high_u8(a: uint8x16_t, b: uint8x16_t) -> uint16x8_t {
    let c: uint8x8_t = simd_shuffle8!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let d: uint8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    simd_cast(vabd_u8(c, d))
}

/// Unsigned Absolute difference Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uabdl))]
pub unsafe fn vabdl_high_u16(a: uint16x8_t, b: uint16x8_t) -> uint32x4_t {
    let c: uint16x4_t = simd_shuffle4!(a, a, [4, 5, 6, 7]);
    let d: uint16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    simd_cast(vabd_u16(c, d))
}

/// Unsigned Absolute difference Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uabdl))]
pub unsafe fn vabdl_high_u32(a: uint32x4_t, b: uint32x4_t) -> uint64x2_t {
    let c: uint32x2_t = simd_shuffle2!(a, a, [2, 3]);
    let d: uint32x2_t = simd_shuffle2!(b, b, [2, 3]);
    simd_cast(vabd_u32(c, d))
}

/// Signed Absolute difference Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sabdl))]
pub unsafe fn vabdl_high_s8(a: int8x16_t, b: int8x16_t) -> int16x8_t {
    let c: int8x8_t = simd_shuffle8!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let d: int8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let e: uint8x8_t = simd_cast(vabd_s8(c, d));
    simd_cast(e)
}

/// Signed Absolute difference Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sabdl))]
pub unsafe fn vabdl_high_s16(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    let c: int16x4_t = simd_shuffle4!(a, a, [4, 5, 6, 7]);
    let d: int16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    let e: uint16x4_t = simd_cast(vabd_s16(c, d));
    simd_cast(e)
}

/// Signed Absolute difference Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sabdl))]
pub unsafe fn vabdl_high_s32(a: int32x4_t, b: int32x4_t) -> int64x2_t {
    let c: int32x2_t = simd_shuffle2!(a, a, [2, 3]);
    let d: int32x2_t = simd_shuffle2!(b, b, [2, 3]);
    let e: uint32x2_t = simd_cast(vabd_s32(c, d));
    simd_cast(e)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceq_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceq_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceq_p64(a: poly64x1_t, b: poly64x1_t) -> uint64x1_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqq_p64(a: poly64x2_t, b: poly64x2_t) -> uint64x2_t {
    simd_eq(a, b)
}

/// Floating-point compare equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmeq))]
pub unsafe fn vceq_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_eq(a, b)
}

/// Floating-point compare equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmeq))]
pub unsafe fn vceqq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_eq(a, b)
}

/// Signed compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqz_s8(a: int8x8_t) -> uint8x8_t {
    let b: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

/// Signed compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqzq_s8(a: int8x16_t) -> uint8x16_t {
    let b: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

/// Signed compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqz_s16(a: int16x4_t) -> uint16x4_t {
    let b: i16x4 = i16x4::new(0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

/// Signed compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqzq_s16(a: int16x8_t) -> uint16x8_t {
    let b: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

/// Signed compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqz_s32(a: int32x2_t) -> uint32x2_t {
    let b: i32x2 = i32x2::new(0, 0);
    simd_eq(a, transmute(b))
}

/// Signed compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqzq_s32(a: int32x4_t) -> uint32x4_t {
    let b: i32x4 = i32x4::new(0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

/// Signed compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqz_s64(a: int64x1_t) -> uint64x1_t {
    let b: i64x1 = i64x1::new(0);
    simd_eq(a, transmute(b))
}

/// Signed compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqzq_s64(a: int64x2_t) -> uint64x2_t {
    let b: i64x2 = i64x2::new(0, 0);
    simd_eq(a, transmute(b))
}

/// Signed compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqz_p8(a: poly8x8_t) -> uint8x8_t {
    let b: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

/// Signed compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqzq_p8(a: poly8x16_t) -> uint8x16_t {
    let b: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

/// Signed compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqz_p64(a: poly64x1_t) -> uint64x1_t {
    let b: i64x1 = i64x1::new(0);
    simd_eq(a, transmute(b))
}

/// Signed compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqzq_p64(a: poly64x2_t) -> uint64x2_t {
    let b: i64x2 = i64x2::new(0, 0);
    simd_eq(a, transmute(b))
}

/// Unsigned compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqz_u8(a: uint8x8_t) -> uint8x8_t {
    let b: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

/// Unsigned compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqzq_u8(a: uint8x16_t) -> uint8x16_t {
    let b: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

/// Unsigned compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqz_u16(a: uint16x4_t) -> uint16x4_t {
    let b: u16x4 = u16x4::new(0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

/// Unsigned compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqzq_u16(a: uint16x8_t) -> uint16x8_t {
    let b: u16x8 = u16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

/// Unsigned compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqz_u32(a: uint32x2_t) -> uint32x2_t {
    let b: u32x2 = u32x2::new(0, 0);
    simd_eq(a, transmute(b))
}

/// Unsigned compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqzq_u32(a: uint32x4_t) -> uint32x4_t {
    let b: u32x4 = u32x4::new(0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

/// Unsigned compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqz_u64(a: uint64x1_t) -> uint64x1_t {
    let b: u64x1 = u64x1::new(0);
    simd_eq(a, transmute(b))
}

/// Unsigned compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
pub unsafe fn vceqzq_u64(a: uint64x2_t) -> uint64x2_t {
    let b: u64x2 = u64x2::new(0, 0);
    simd_eq(a, transmute(b))
}

/// Floating-point compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmeq))]
pub unsafe fn vceqz_f32(a: float32x2_t) -> uint32x2_t {
    let b: f32x2 = f32x2::new(0.0, 0.0);
    simd_eq(a, transmute(b))
}

/// Floating-point compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmeq))]
pub unsafe fn vceqzq_f32(a: float32x4_t) -> uint32x4_t {
    let b: f32x4 = f32x4::new(0.0, 0.0, 0.0, 0.0);
    simd_eq(a, transmute(b))
}

/// Floating-point compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmeq))]
pub unsafe fn vceqz_f64(a: float64x1_t) -> uint64x1_t {
    let b: f64 = 0.0;
    simd_eq(a, transmute(b))
}

/// Floating-point compare bitwise equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmeq))]
pub unsafe fn vceqzq_f64(a: float64x2_t) -> uint64x2_t {
    let b: f64x2 = f64x2::new(0.0, 0.0);
    simd_eq(a, transmute(b))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmtst))]
pub unsafe fn vtst_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    let c: int64x1_t = simd_and(a, b);
    let d: i64x1 = i64x1::new(0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmtst))]
pub unsafe fn vtstq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    let c: int64x2_t = simd_and(a, b);
    let d: i64x2 = i64x2::new(0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmtst))]
pub unsafe fn vtst_p64(a: poly64x1_t, b: poly64x1_t) -> uint64x1_t {
    let c: poly64x1_t = simd_and(a, b);
    let d: i64x1 = i64x1::new(0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmtst))]
pub unsafe fn vtstq_p64(a: poly64x2_t, b: poly64x2_t) -> uint64x2_t {
    let c: poly64x2_t = simd_and(a, b);
    let d: i64x2 = i64x2::new(0, 0);
    simd_ne(c, transmute(d))
}

/// Unsigned compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmtst))]
pub unsafe fn vtst_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    let c: uint64x1_t = simd_and(a, b);
    let d: u64x1 = u64x1::new(0);
    simd_ne(c, transmute(d))
}

/// Unsigned compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmtst))]
pub unsafe fn vtstq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    let c: uint64x2_t = simd_and(a, b);
    let d: u64x2 = u64x2::new(0, 0);
    simd_ne(c, transmute(d))
}

/// Floating-point absolute value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fabs))]
pub unsafe fn vabs_f64(a: float64x1_t) -> float64x1_t {
    simd_fabs(a)
}

/// Floating-point absolute value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fabs))]
pub unsafe fn vabsq_f64(a: float64x2_t) -> float64x2_t {
    simd_fabs(a)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vcgt_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vcgtq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhi))]
pub unsafe fn vcgt_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhi))]
pub unsafe fn vcgtq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_gt(a, b)
}

/// Floating-point compare greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub unsafe fn vcgt_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_gt(a, b)
}

/// Floating-point compare greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub unsafe fn vcgtq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_gt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vclt_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vcltq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhi))]
pub unsafe fn vclt_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhi))]
pub unsafe fn vcltq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_lt(a, b)
}

/// Floating-point compare less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub unsafe fn vclt_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_lt(a, b)
}

/// Floating-point compare less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub unsafe fn vcltq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_lt(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcle_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcleq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhs))]
pub unsafe fn vcle_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhs))]
pub unsafe fn vcleq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_le(a, b)
}

/// Floating-point compare less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
pub unsafe fn vcle_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_le(a, b)
}

/// Floating-point compare less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
pub unsafe fn vcleq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_le(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcge_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcgeq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhs))]
pub unsafe fn vcge_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhs))]
pub unsafe fn vcgeq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_ge(a, b)
}

/// Floating-point compare greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
pub unsafe fn vcge_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_ge(a, b)
}

/// Floating-point compare greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
pub unsafe fn vcgeq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcgez_s8(a: int8x8_t) -> uint8x8_t {
    let b: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_ge(a, transmute(b))
}

/// Compare signed greater than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcgezq_s8(a: int8x16_t) -> uint8x16_t {
    let b: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_ge(a, transmute(b))
}

/// Compare signed greater than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcgez_s16(a: int16x4_t) -> uint16x4_t {
    let b: i16x4 = i16x4::new(0, 0, 0, 0);
    simd_ge(a, transmute(b))
}

/// Compare signed greater than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcgezq_s16(a: int16x8_t) -> uint16x8_t {
    let b: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_ge(a, transmute(b))
}

/// Compare signed greater than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcgez_s32(a: int32x2_t) -> uint32x2_t {
    let b: i32x2 = i32x2::new(0, 0);
    simd_ge(a, transmute(b))
}

/// Compare signed greater than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcgezq_s32(a: int32x4_t) -> uint32x4_t {
    let b: i32x4 = i32x4::new(0, 0, 0, 0);
    simd_ge(a, transmute(b))
}

/// Compare signed greater than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcgez_s64(a: int64x1_t) -> uint64x1_t {
    let b: i64x1 = i64x1::new(0);
    simd_ge(a, transmute(b))
}

/// Compare signed greater than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
pub unsafe fn vcgezq_s64(a: int64x2_t) -> uint64x2_t {
    let b: i64x2 = i64x2::new(0, 0);
    simd_ge(a, transmute(b))
}

/// Floating-point compare greater than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
pub unsafe fn vcgez_f32(a: float32x2_t) -> uint32x2_t {
    let b: f32x2 = f32x2::new(0.0, 0.0);
    simd_ge(a, transmute(b))
}

/// Floating-point compare greater than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
pub unsafe fn vcgezq_f32(a: float32x4_t) -> uint32x4_t {
    let b: f32x4 = f32x4::new(0.0, 0.0, 0.0, 0.0);
    simd_ge(a, transmute(b))
}

/// Floating-point compare greater than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
pub unsafe fn vcgez_f64(a: float64x1_t) -> uint64x1_t {
    let b: f64 = 0.0;
    simd_ge(a, transmute(b))
}

/// Floating-point compare greater than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
pub unsafe fn vcgezq_f64(a: float64x2_t) -> uint64x2_t {
    let b: f64x2 = f64x2::new(0.0, 0.0);
    simd_ge(a, transmute(b))
}

/// Compare signed greater than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vcgtz_s8(a: int8x8_t) -> uint8x8_t {
    let b: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_gt(a, transmute(b))
}

/// Compare signed greater than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vcgtzq_s8(a: int8x16_t) -> uint8x16_t {
    let b: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_gt(a, transmute(b))
}

/// Compare signed greater than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vcgtz_s16(a: int16x4_t) -> uint16x4_t {
    let b: i16x4 = i16x4::new(0, 0, 0, 0);
    simd_gt(a, transmute(b))
}

/// Compare signed greater than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vcgtzq_s16(a: int16x8_t) -> uint16x8_t {
    let b: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_gt(a, transmute(b))
}

/// Compare signed greater than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vcgtz_s32(a: int32x2_t) -> uint32x2_t {
    let b: i32x2 = i32x2::new(0, 0);
    simd_gt(a, transmute(b))
}

/// Compare signed greater than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vcgtzq_s32(a: int32x4_t) -> uint32x4_t {
    let b: i32x4 = i32x4::new(0, 0, 0, 0);
    simd_gt(a, transmute(b))
}

/// Compare signed greater than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vcgtz_s64(a: int64x1_t) -> uint64x1_t {
    let b: i64x1 = i64x1::new(0);
    simd_gt(a, transmute(b))
}

/// Compare signed greater than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vcgtzq_s64(a: int64x2_t) -> uint64x2_t {
    let b: i64x2 = i64x2::new(0, 0);
    simd_gt(a, transmute(b))
}

/// Floating-point compare greater than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub unsafe fn vcgtz_f32(a: float32x2_t) -> uint32x2_t {
    let b: f32x2 = f32x2::new(0.0, 0.0);
    simd_gt(a, transmute(b))
}

/// Floating-point compare greater than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub unsafe fn vcgtzq_f32(a: float32x4_t) -> uint32x4_t {
    let b: f32x4 = f32x4::new(0.0, 0.0, 0.0, 0.0);
    simd_gt(a, transmute(b))
}

/// Floating-point compare greater than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub unsafe fn vcgtz_f64(a: float64x1_t) -> uint64x1_t {
    let b: f64 = 0.0;
    simd_gt(a, transmute(b))
}

/// Floating-point compare greater than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub unsafe fn vcgtzq_f64(a: float64x2_t) -> uint64x2_t {
    let b: f64x2 = f64x2::new(0.0, 0.0);
    simd_gt(a, transmute(b))
}

/// Compare signed less than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vclez_s8(a: int8x8_t) -> uint8x8_t {
    let b: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_le(a, transmute(b))
}

/// Compare signed less than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vclezq_s8(a: int8x16_t) -> uint8x16_t {
    let b: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_le(a, transmute(b))
}

/// Compare signed less than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vclez_s16(a: int16x4_t) -> uint16x4_t {
    let b: i16x4 = i16x4::new(0, 0, 0, 0);
    simd_le(a, transmute(b))
}

/// Compare signed less than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vclezq_s16(a: int16x8_t) -> uint16x8_t {
    let b: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_le(a, transmute(b))
}

/// Compare signed less than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vclez_s32(a: int32x2_t) -> uint32x2_t {
    let b: i32x2 = i32x2::new(0, 0);
    simd_le(a, transmute(b))
}

/// Compare signed less than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vclezq_s32(a: int32x4_t) -> uint32x4_t {
    let b: i32x4 = i32x4::new(0, 0, 0, 0);
    simd_le(a, transmute(b))
}

/// Compare signed less than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vclez_s64(a: int64x1_t) -> uint64x1_t {
    let b: i64x1 = i64x1::new(0);
    simd_le(a, transmute(b))
}

/// Compare signed less than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
pub unsafe fn vclezq_s64(a: int64x2_t) -> uint64x2_t {
    let b: i64x2 = i64x2::new(0, 0);
    simd_le(a, transmute(b))
}

/// Floating-point compare less than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmle))]
pub unsafe fn vclez_f32(a: float32x2_t) -> uint32x2_t {
    let b: f32x2 = f32x2::new(0.0, 0.0);
    simd_le(a, transmute(b))
}

/// Floating-point compare less than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmle))]
pub unsafe fn vclezq_f32(a: float32x4_t) -> uint32x4_t {
    let b: f32x4 = f32x4::new(0.0, 0.0, 0.0, 0.0);
    simd_le(a, transmute(b))
}

/// Floating-point compare less than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmle))]
pub unsafe fn vclez_f64(a: float64x1_t) -> uint64x1_t {
    let b: f64 = 0.0;
    simd_le(a, transmute(b))
}

/// Floating-point compare less than or equal to zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmle))]
pub unsafe fn vclezq_f64(a: float64x2_t) -> uint64x2_t {
    let b: f64x2 = f64x2::new(0.0, 0.0);
    simd_le(a, transmute(b))
}

/// Compare signed less than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshr))]
pub unsafe fn vcltz_s8(a: int8x8_t) -> uint8x8_t {
    let b: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_lt(a, transmute(b))
}

/// Compare signed less than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshr))]
pub unsafe fn vcltzq_s8(a: int8x16_t) -> uint8x16_t {
    let b: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_lt(a, transmute(b))
}

/// Compare signed less than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshr))]
pub unsafe fn vcltz_s16(a: int16x4_t) -> uint16x4_t {
    let b: i16x4 = i16x4::new(0, 0, 0, 0);
    simd_lt(a, transmute(b))
}

/// Compare signed less than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshr))]
pub unsafe fn vcltzq_s16(a: int16x8_t) -> uint16x8_t {
    let b: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_lt(a, transmute(b))
}

/// Compare signed less than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshr))]
pub unsafe fn vcltz_s32(a: int32x2_t) -> uint32x2_t {
    let b: i32x2 = i32x2::new(0, 0);
    simd_lt(a, transmute(b))
}

/// Compare signed less than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshr))]
pub unsafe fn vcltzq_s32(a: int32x4_t) -> uint32x4_t {
    let b: i32x4 = i32x4::new(0, 0, 0, 0);
    simd_lt(a, transmute(b))
}

/// Compare signed less than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshr))]
pub unsafe fn vcltz_s64(a: int64x1_t) -> uint64x1_t {
    let b: i64x1 = i64x1::new(0);
    simd_lt(a, transmute(b))
}

/// Compare signed less than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshr))]
pub unsafe fn vcltzq_s64(a: int64x2_t) -> uint64x2_t {
    let b: i64x2 = i64x2::new(0, 0);
    simd_lt(a, transmute(b))
}

/// Floating-point compare less than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmlt))]
pub unsafe fn vcltz_f32(a: float32x2_t) -> uint32x2_t {
    let b: f32x2 = f32x2::new(0.0, 0.0);
    simd_lt(a, transmute(b))
}

/// Floating-point compare less than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmlt))]
pub unsafe fn vcltzq_f32(a: float32x4_t) -> uint32x4_t {
    let b: f32x4 = f32x4::new(0.0, 0.0, 0.0, 0.0);
    simd_lt(a, transmute(b))
}

/// Floating-point compare less than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmlt))]
pub unsafe fn vcltz_f64(a: float64x1_t) -> uint64x1_t {
    let b: f64 = 0.0;
    simd_lt(a, transmute(b))
}

/// Floating-point compare less than zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmlt))]
pub unsafe fn vcltzq_f64(a: float64x2_t) -> uint64x2_t {
    let b: f64x2 = f64x2::new(0.0, 0.0);
    simd_lt(a, transmute(b))
}

/// Floating-point absolute compare greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facgt))]
pub unsafe fn vcagt_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.facgt.v1i64.v1f64")]
        fn vcagt_f64_(a: float64x1_t, b: float64x1_t) -> uint64x1_t;
    }
    vcagt_f64_(a, b)
}

/// Floating-point absolute compare greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facgt))]
pub unsafe fn vcagtq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.facgt.v2i64.v2f64")]
        fn vcagtq_f64_(a: float64x2_t, b: float64x2_t) -> uint64x2_t;
    }
    vcagtq_f64_(a, b)
}

/// Floating-point absolute compare greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facge))]
pub unsafe fn vcage_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.facge.v1i64.v1f64")]
        fn vcage_f64_(a: float64x1_t, b: float64x1_t) -> uint64x1_t;
    }
    vcage_f64_(a, b)
}

/// Floating-point absolute compare greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facge))]
pub unsafe fn vcageq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.facge.v2i64.v2f64")]
        fn vcageq_f64_(a: float64x2_t, b: float64x2_t) -> uint64x2_t;
    }
    vcageq_f64_(a, b)
}

/// Floating-point absolute compare less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facgt))]
pub unsafe fn vcalt_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    vcagt_f64(b, a)
}

/// Floating-point absolute compare less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facgt))]
pub unsafe fn vcaltq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    vcagtq_f64(b, a)
}

/// Floating-point absolute compare less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facge))]
pub unsafe fn vcale_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    vcage_f64(b, a)
}

/// Floating-point absolute compare less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facge))]
pub unsafe fn vcaleq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    vcageq_f64(b, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_lane_s8<const LANE1: i32, const LANE2: i32>(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    static_assert_imm3!(LANE1);
    static_assert_imm3!(LANE2);
    match LANE1 & 0b111 {
        0 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_laneq_s8<const LANE1: i32, const LANE2: i32>(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    static_assert_imm4!(LANE1);
    static_assert_imm4!(LANE2);
    match LANE1 & 0b1111 {
        0 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [16 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        1 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 16 + LANE2 as u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        2 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 16 + LANE2 as u32, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        3 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 16 + LANE2 as u32, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        4 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 16 + LANE2 as u32, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        5 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 16 + LANE2 as u32, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        6 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 16 + LANE2 as u32, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        7 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 16 + LANE2 as u32, 8, 9, 10, 11, 12, 13, 14, 15]),
        8 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 16 + LANE2 as u32, 9, 10, 11, 12, 13, 14, 15]),
        9 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 16 + LANE2 as u32, 10, 11, 12, 13, 14, 15]),
        10 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16 + LANE2 as u32, 11, 12, 13, 14, 15]),
        11 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16 + LANE2 as u32, 12, 13, 14, 15]),
        12 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16 + LANE2 as u32, 13, 14, 15]),
        13 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16 + LANE2 as u32, 14, 15]),
        14 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16 + LANE2 as u32, 15]),
        15 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_lane_s16<const LANE1: i32, const LANE2: i32>(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    static_assert_imm2!(LANE1);
    static_assert_imm2!(LANE2);
    match LANE1 & 0b11 {
        0 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_laneq_s16<const LANE1: i32, const LANE2: i32>(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    static_assert_imm3!(LANE1);
    static_assert_imm3!(LANE2);
    match LANE1 & 0b111 {
        0 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_lane_s32<const LANE1: i32, const LANE2: i32>(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    static_assert_imm1!(LANE1);
    static_assert_imm1!(LANE2);
    match LANE1 & 0b1 {
        0 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [2 + LANE2 as u32, 1]),
        1 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_laneq_s32<const LANE1: i32, const LANE2: i32>(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    static_assert_imm2!(LANE1);
    static_assert_imm2!(LANE2);
    match LANE1 & 0b11 {
        0 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_laneq_s64<const LANE1: i32, const LANE2: i32>(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    static_assert_imm1!(LANE1);
    static_assert_imm1!(LANE2);
    match LANE1 & 0b1 {
        0 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [2 + LANE2 as u32, 1]),
        1 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_lane_u8<const LANE1: i32, const LANE2: i32>(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    static_assert_imm3!(LANE1);
    static_assert_imm3!(LANE2);
    match LANE1 & 0b111 {
        0 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_laneq_u8<const LANE1: i32, const LANE2: i32>(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    static_assert_imm4!(LANE1);
    static_assert_imm4!(LANE2);
    match LANE1 & 0b1111 {
        0 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [16 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        1 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 16 + LANE2 as u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        2 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 16 + LANE2 as u32, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        3 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 16 + LANE2 as u32, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        4 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 16 + LANE2 as u32, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        5 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 16 + LANE2 as u32, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        6 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 16 + LANE2 as u32, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        7 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 16 + LANE2 as u32, 8, 9, 10, 11, 12, 13, 14, 15]),
        8 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 16 + LANE2 as u32, 9, 10, 11, 12, 13, 14, 15]),
        9 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 16 + LANE2 as u32, 10, 11, 12, 13, 14, 15]),
        10 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16 + LANE2 as u32, 11, 12, 13, 14, 15]),
        11 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16 + LANE2 as u32, 12, 13, 14, 15]),
        12 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16 + LANE2 as u32, 13, 14, 15]),
        13 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16 + LANE2 as u32, 14, 15]),
        14 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16 + LANE2 as u32, 15]),
        15 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_lane_u16<const LANE1: i32, const LANE2: i32>(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    static_assert_imm2!(LANE1);
    static_assert_imm2!(LANE2);
    match LANE1 & 0b11 {
        0 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_laneq_u16<const LANE1: i32, const LANE2: i32>(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    static_assert_imm3!(LANE1);
    static_assert_imm3!(LANE2);
    match LANE1 & 0b111 {
        0 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_lane_u32<const LANE1: i32, const LANE2: i32>(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    static_assert_imm1!(LANE1);
    static_assert_imm1!(LANE2);
    match LANE1 & 0b1 {
        0 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [2 + LANE2 as u32, 1]),
        1 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_laneq_u32<const LANE1: i32, const LANE2: i32>(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    static_assert_imm2!(LANE1);
    static_assert_imm2!(LANE2);
    match LANE1 & 0b11 {
        0 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_laneq_u64<const LANE1: i32, const LANE2: i32>(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    static_assert_imm1!(LANE1);
    static_assert_imm1!(LANE2);
    match LANE1 & 0b1 {
        0 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [2 + LANE2 as u32, 1]),
        1 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_lane_p8<const LANE1: i32, const LANE2: i32>(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    static_assert_imm3!(LANE1);
    static_assert_imm3!(LANE2);
    match LANE1 & 0b111 {
        0 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_laneq_p8<const LANE1: i32, const LANE2: i32>(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    static_assert_imm4!(LANE1);
    static_assert_imm4!(LANE2);
    match LANE1 & 0b1111 {
        0 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [16 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        1 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 16 + LANE2 as u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        2 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 16 + LANE2 as u32, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        3 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 16 + LANE2 as u32, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        4 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 16 + LANE2 as u32, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        5 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 16 + LANE2 as u32, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        6 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 16 + LANE2 as u32, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        7 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 16 + LANE2 as u32, 8, 9, 10, 11, 12, 13, 14, 15]),
        8 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 16 + LANE2 as u32, 9, 10, 11, 12, 13, 14, 15]),
        9 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 16 + LANE2 as u32, 10, 11, 12, 13, 14, 15]),
        10 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16 + LANE2 as u32, 11, 12, 13, 14, 15]),
        11 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16 + LANE2 as u32, 12, 13, 14, 15]),
        12 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16 + LANE2 as u32, 13, 14, 15]),
        13 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16 + LANE2 as u32, 14, 15]),
        14 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16 + LANE2 as u32, 15]),
        15 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_lane_p16<const LANE1: i32, const LANE2: i32>(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    static_assert_imm2!(LANE1);
    static_assert_imm2!(LANE2);
    match LANE1 & 0b11 {
        0 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_laneq_p16<const LANE1: i32, const LANE2: i32>(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    static_assert_imm3!(LANE1);
    static_assert_imm3!(LANE2);
    match LANE1 & 0b111 {
        0 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_laneq_p64<const LANE1: i32, const LANE2: i32>(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    static_assert_imm1!(LANE1);
    static_assert_imm1!(LANE2);
    match LANE1 & 0b1 {
        0 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [2 + LANE2 as u32, 1]),
        1 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_lane_f32<const LANE1: i32, const LANE2: i32>(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    static_assert_imm1!(LANE1);
    static_assert_imm1!(LANE2);
    match LANE1 & 0b1 {
        0 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [2 + LANE2 as u32, 1]),
        1 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_laneq_f32<const LANE1: i32, const LANE2: i32>(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    static_assert_imm2!(LANE1);
    static_assert_imm2!(LANE2);
    match LANE1 & 0b11 {
        0 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_laneq_f64<const LANE1: i32, const LANE2: i32>(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    static_assert_imm1!(LANE1);
    static_assert_imm1!(LANE2);
    match LANE1 & 0b1 {
        0 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [2 + LANE2 as u32, 1]),
        1 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_laneq_s8<const LANE1: i32, const LANE2: i32>(a: int8x8_t, b: int8x16_t) -> int8x8_t {
    static_assert_imm3!(LANE1);
    static_assert_imm4!(LANE2);
    let a: int8x16_t = simd_shuffle16!(a, a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    match LANE1 & 0b111 {
        0 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [16 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 16 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 16 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 16 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 16 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 16 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 16 + LANE2 as u32, 7]),
        7 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 16 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_laneq_s16<const LANE1: i32, const LANE2: i32>(a: int16x4_t, b: int16x8_t) -> int16x4_t {
    static_assert_imm2!(LANE1);
    static_assert_imm3!(LANE2);
    let a: int16x8_t = simd_shuffle8!(a, a, [0, 1, 2, 3, 4, 5, 6, 7]);
    match LANE1 & 0b11 {
        0 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [8 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 8 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 8 + LANE2 as u32, 3]),
        3 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_laneq_s32<const LANE1: i32, const LANE2: i32>(a: int32x2_t, b: int32x4_t) -> int32x2_t {
    static_assert_imm1!(LANE1);
    static_assert_imm2!(LANE2);
    let a: int32x4_t = simd_shuffle4!(a, a, [0, 1, 2, 3]);
    match LANE1 & 0b1 {
        0 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [4 + LANE2 as u32, 1]),
        1 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [0, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_laneq_u8<const LANE1: i32, const LANE2: i32>(a: uint8x8_t, b: uint8x16_t) -> uint8x8_t {
    static_assert_imm3!(LANE1);
    static_assert_imm4!(LANE2);
    let a: uint8x16_t = simd_shuffle16!(a, a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    match LANE1 & 0b111 {
        0 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [16 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 16 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 16 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 16 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 16 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 16 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 16 + LANE2 as u32, 7]),
        7 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 16 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_laneq_u16<const LANE1: i32, const LANE2: i32>(a: uint16x4_t, b: uint16x8_t) -> uint16x4_t {
    static_assert_imm2!(LANE1);
    static_assert_imm3!(LANE2);
    let a: uint16x8_t = simd_shuffle8!(a, a, [0, 1, 2, 3, 4, 5, 6, 7]);
    match LANE1 & 0b11 {
        0 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [8 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 8 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 8 + LANE2 as u32, 3]),
        3 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_laneq_u32<const LANE1: i32, const LANE2: i32>(a: uint32x2_t, b: uint32x4_t) -> uint32x2_t {
    static_assert_imm1!(LANE1);
    static_assert_imm2!(LANE2);
    let a: uint32x4_t = simd_shuffle4!(a, a, [0, 1, 2, 3]);
    match LANE1 & 0b1 {
        0 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [4 + LANE2 as u32, 1]),
        1 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [0, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_laneq_p8<const LANE1: i32, const LANE2: i32>(a: poly8x8_t, b: poly8x16_t) -> poly8x8_t {
    static_assert_imm3!(LANE1);
    static_assert_imm4!(LANE2);
    let a: poly8x16_t = simd_shuffle16!(a, a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    match LANE1 & 0b111 {
        0 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [16 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 16 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 16 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 16 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 16 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 16 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 16 + LANE2 as u32, 7]),
        7 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 16 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_laneq_p16<const LANE1: i32, const LANE2: i32>(a: poly16x4_t, b: poly16x8_t) -> poly16x4_t {
    static_assert_imm2!(LANE1);
    static_assert_imm3!(LANE2);
    let a: poly16x8_t = simd_shuffle8!(a, a, [0, 1, 2, 3, 4, 5, 6, 7]);
    match LANE1 & 0b11 {
        0 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [8 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 8 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 8 + LANE2 as u32, 3]),
        3 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopy_laneq_f32<const LANE1: i32, const LANE2: i32>(a: float32x2_t, b: float32x4_t) -> float32x2_t {
    static_assert_imm1!(LANE1);
    static_assert_imm2!(LANE2);
    let a: float32x4_t = simd_shuffle4!(a, a, [0, 1, 2, 3]);
    match LANE1 & 0b1 {
        0 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [4 + LANE2 as u32, 1]),
        1 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [0, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_lane_s8<const LANE1: i32, const LANE2: i32>(a: int8x16_t, b: int8x8_t) -> int8x16_t {
    static_assert_imm4!(LANE1);
    static_assert_imm3!(LANE2);
    let b: int8x16_t = simd_shuffle16!(b, b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    match LANE1 & 0b1111 {
        0 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [16 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        1 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 16 + LANE2 as u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        2 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 16 + LANE2 as u32, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        3 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 16 + LANE2 as u32, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        4 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 16 + LANE2 as u32, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        5 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 16 + LANE2 as u32, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        6 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 16 + LANE2 as u32, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        7 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 16 + LANE2 as u32, 8, 9, 10, 11, 12, 13, 14, 15]),
        8 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 16 + LANE2 as u32, 9, 10, 11, 12, 13, 14, 15]),
        9 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 16 + LANE2 as u32, 10, 11, 12, 13, 14, 15]),
        10 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16 + LANE2 as u32, 11, 12, 13, 14, 15]),
        11 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16 + LANE2 as u32, 12, 13, 14, 15]),
        12 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16 + LANE2 as u32, 13, 14, 15]),
        13 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16 + LANE2 as u32, 14, 15]),
        14 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16 + LANE2 as u32, 15]),
        15 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_lane_s16<const LANE1: i32, const LANE2: i32>(a: int16x8_t, b: int16x4_t) -> int16x8_t {
    static_assert_imm3!(LANE1);
    static_assert_imm2!(LANE2);
    let b: int16x8_t = simd_shuffle8!(b, b, [0, 1, 2, 3, 4, 5, 6, 7]);
    match LANE1 & 0b111 {
        0 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_lane_s32<const LANE1: i32, const LANE2: i32>(a: int32x4_t, b: int32x2_t) -> int32x4_t {
    static_assert_imm2!(LANE1);
    static_assert_imm1!(LANE2);
    let b: int32x4_t = simd_shuffle4!(b, b, [0, 1, 2, 3]);
    match LANE1 & 0b11 {
        0 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_lane_u8<const LANE1: i32, const LANE2: i32>(a: uint8x16_t, b: uint8x8_t) -> uint8x16_t {
    static_assert_imm4!(LANE1);
    static_assert_imm3!(LANE2);
    let b: uint8x16_t = simd_shuffle16!(b, b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    match LANE1 & 0b1111 {
        0 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [16 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        1 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 16 + LANE2 as u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        2 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 16 + LANE2 as u32, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        3 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 16 + LANE2 as u32, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        4 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 16 + LANE2 as u32, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        5 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 16 + LANE2 as u32, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        6 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 16 + LANE2 as u32, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        7 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 16 + LANE2 as u32, 8, 9, 10, 11, 12, 13, 14, 15]),
        8 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 16 + LANE2 as u32, 9, 10, 11, 12, 13, 14, 15]),
        9 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 16 + LANE2 as u32, 10, 11, 12, 13, 14, 15]),
        10 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16 + LANE2 as u32, 11, 12, 13, 14, 15]),
        11 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16 + LANE2 as u32, 12, 13, 14, 15]),
        12 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16 + LANE2 as u32, 13, 14, 15]),
        13 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16 + LANE2 as u32, 14, 15]),
        14 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16 + LANE2 as u32, 15]),
        15 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_lane_u16<const LANE1: i32, const LANE2: i32>(a: uint16x8_t, b: uint16x4_t) -> uint16x8_t {
    static_assert_imm3!(LANE1);
    static_assert_imm2!(LANE2);
    let b: uint16x8_t = simd_shuffle8!(b, b, [0, 1, 2, 3, 4, 5, 6, 7]);
    match LANE1 & 0b111 {
        0 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_lane_u32<const LANE1: i32, const LANE2: i32>(a: uint32x4_t, b: uint32x2_t) -> uint32x4_t {
    static_assert_imm2!(LANE1);
    static_assert_imm1!(LANE2);
    let b: uint32x4_t = simd_shuffle4!(b, b, [0, 1, 2, 3]);
    match LANE1 & 0b11 {
        0 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_lane_p8<const LANE1: i32, const LANE2: i32>(a: poly8x16_t, b: poly8x8_t) -> poly8x16_t {
    static_assert_imm4!(LANE1);
    static_assert_imm3!(LANE2);
    let b: poly8x16_t = simd_shuffle16!(b, b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    match LANE1 & 0b1111 {
        0 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [16 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        1 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 16 + LANE2 as u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        2 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 16 + LANE2 as u32, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        3 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 16 + LANE2 as u32, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        4 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 16 + LANE2 as u32, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        5 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 16 + LANE2 as u32, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        6 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 16 + LANE2 as u32, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        7 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 16 + LANE2 as u32, 8, 9, 10, 11, 12, 13, 14, 15]),
        8 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 16 + LANE2 as u32, 9, 10, 11, 12, 13, 14, 15]),
        9 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 16 + LANE2 as u32, 10, 11, 12, 13, 14, 15]),
        10 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16 + LANE2 as u32, 11, 12, 13, 14, 15]),
        11 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16 + LANE2 as u32, 12, 13, 14, 15]),
        12 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16 + LANE2 as u32, 13, 14, 15]),
        13 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16 + LANE2 as u32, 14, 15]),
        14 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16 + LANE2 as u32, 15]),
        15 => simd_shuffle16!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_lane_p16<const LANE1: i32, const LANE2: i32>(a: poly16x8_t, b: poly16x4_t) -> poly16x8_t {
    static_assert_imm3!(LANE1);
    static_assert_imm2!(LANE2);
    let b: poly16x8_t = simd_shuffle8!(b, b, [0, 1, 2, 3, 4, 5, 6, 7]);
    match LANE1 & 0b111 {
        0 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle8!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1, LANE1 = 1, LANE2 = 0))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_lane_s64<const LANE1: i32, const LANE2: i32>(a: int64x2_t, b: int64x1_t) -> int64x2_t {
    static_assert_imm1!(LANE1);
    static_assert!(LANE2 : i32 where LANE2 == 0);
    let b: int64x2_t = simd_shuffle2!(b, b, [0, 1]);
    match LANE1 & 0b1 {
        0 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [2 + LANE2 as u32, 1]),
        1 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1, LANE1 = 1, LANE2 = 0))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_lane_u64<const LANE1: i32, const LANE2: i32>(a: uint64x2_t, b: uint64x1_t) -> uint64x2_t {
    static_assert_imm1!(LANE1);
    static_assert!(LANE2 : i32 where LANE2 == 0);
    let b: uint64x2_t = simd_shuffle2!(b, b, [0, 1]);
    match LANE1 & 0b1 {
        0 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [2 + LANE2 as u32, 1]),
        1 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1, LANE1 = 1, LANE2 = 0))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_lane_p64<const LANE1: i32, const LANE2: i32>(a: poly64x2_t, b: poly64x1_t) -> poly64x2_t {
    static_assert_imm1!(LANE1);
    static_assert!(LANE2 : i32 where LANE2 == 0);
    let b: poly64x2_t = simd_shuffle2!(b, b, [0, 1]);
    match LANE1 & 0b1 {
        0 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [2 + LANE2 as u32, 1]),
        1 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 1, LANE2 = 0))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_lane_f32<const LANE1: i32, const LANE2: i32>(a: float32x4_t, b: float32x2_t) -> float32x4_t {
    static_assert_imm2!(LANE1);
    static_assert_imm1!(LANE2);
    let b: float32x4_t = simd_shuffle4!(b, b, [0, 1, 2, 3]);
    match LANE1 & 0b11 {
        0 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle4!(a, b, <const LANE1: i32, const LANE2: i32> [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1, LANE1 = 1, LANE2 = 0))]
#[rustc_legacy_const_generics(1, 3)]
pub unsafe fn vcopyq_lane_f64<const LANE1: i32, const LANE2: i32>(a: float64x2_t, b: float64x1_t) -> float64x2_t {
    static_assert_imm1!(LANE1);
    static_assert!(LANE2 : i32 where LANE2 == 0);
    let b: float64x2_t = simd_shuffle2!(b, b, [0, 1]);
    match LANE1 & 0b1 {
        0 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [2 + LANE2 as u32, 1]),
        1 => simd_shuffle2!(a, b, <const LANE1: i32, const LANE2: i32> [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vcreate_f64(a: u64) -> float64x1_t {
    transmute(a)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf))]
pub unsafe fn vcvt_f64_s64(a: int64x1_t) -> float64x1_t {
    simd_cast(a)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf))]
pub unsafe fn vcvtq_f64_s64(a: int64x2_t) -> float64x2_t {
    simd_cast(a)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub unsafe fn vcvt_f64_u64(a: uint64x1_t) -> float64x1_t {
    simd_cast(a)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub unsafe fn vcvtq_f64_u64(a: uint64x2_t) -> float64x2_t {
    simd_cast(a)
}

/// Floating-point convert to higher precision long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtl))]
pub unsafe fn vcvt_f64_f32(a: float32x2_t) -> float64x2_t {
    simd_cast(a)
}

/// Floating-point convert to higher precision long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtl))]
pub unsafe fn vcvt_high_f64_f32(a: float32x4_t) -> float64x2_t {
    let b: float32x2_t = simd_shuffle2!(a, a, [2, 3]);
    simd_cast(b)
}

/// Floating-point convert to lower precision narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtn))]
pub unsafe fn vcvt_f32_f64(a: float64x2_t) -> float32x2_t {
    simd_cast(a)
}

/// Floating-point convert to lower precision narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtn))]
pub unsafe fn vcvt_high_f32_f64(a: float32x2_t, b: float64x2_t) -> float32x4_t {
    simd_shuffle4!(a, simd_cast(b), [0, 1, 2, 3])
}

/// Floating-point convert to lower precision narrow, rounding to odd
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtxn))]
pub unsafe fn vcvtx_f32_f64(a: float64x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtxn.v2f32.v2f64")]
        fn vcvtx_f32_f64_(a: float64x2_t) -> float32x2_t;
    }
    vcvtx_f32_f64_(a)
}

/// Floating-point convert to lower precision narrow, rounding to odd
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtxn))]
pub unsafe fn vcvtx_high_f32_f64(a: float32x2_t, b: float64x2_t) -> float32x4_t {
    simd_shuffle4!(a, vcvtx_f32_f64(b), [0, 1, 2, 3])
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvt_n_f64_s64<const N: i32>(a: int64x1_t) -> float64x1_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfxs2fp.v1f64.v1i64")]
        fn vcvt_n_f64_s64_(a: int64x1_t, n: i32) -> float64x1_t;
    }
    vcvt_n_f64_s64_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtq_n_f64_s64<const N: i32>(a: int64x2_t) -> float64x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfxs2fp.v2f64.v2i64")]
        fn vcvtq_n_f64_s64_(a: int64x2_t, n: i32) -> float64x2_t;
    }
    vcvtq_n_f64_s64_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvts_n_f32_s32<const N: i32>(a: i32) -> f32 {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfxs2fp.f32.i32")]
        fn vcvts_n_f32_s32_(a: i32, n: i32) -> f32;
    }
    vcvts_n_f32_s32_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtd_n_f64_s64<const N: i32>(a: i64) -> f64 {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfxs2fp.f64.i64")]
        fn vcvtd_n_f64_s64_(a: i64, n: i32) -> f64;
    }
    vcvtd_n_f64_s64_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvt_n_f64_u64<const N: i32>(a: uint64x1_t) -> float64x1_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfxu2fp.v1f64.v1i64")]
        fn vcvt_n_f64_u64_(a: uint64x1_t, n: i32) -> float64x1_t;
    }
    vcvt_n_f64_u64_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtq_n_f64_u64<const N: i32>(a: uint64x2_t) -> float64x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfxu2fp.v2f64.v2i64")]
        fn vcvtq_n_f64_u64_(a: uint64x2_t, n: i32) -> float64x2_t;
    }
    vcvtq_n_f64_u64_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvts_n_f32_u32<const N: i32>(a: u32) -> f32 {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfxu2fp.f32.i32")]
        fn vcvts_n_f32_u32_(a: u32, n: i32) -> f32;
    }
    vcvts_n_f32_u32_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtd_n_f64_u64<const N: i32>(a: u64) -> f64 {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfxu2fp.f64.i64")]
        fn vcvtd_n_f64_u64_(a: u64, n: i32) -> f64;
    }
    vcvtd_n_f64_u64_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvt_n_s64_f64<const N: i32>(a: float64x1_t) -> int64x1_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfp2fxs.v1i64.v1f64")]
        fn vcvt_n_s64_f64_(a: float64x1_t, n: i32) -> int64x1_t;
    }
    vcvt_n_s64_f64_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtq_n_s64_f64<const N: i32>(a: float64x2_t) -> int64x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfp2fxs.v2i64.v2f64")]
        fn vcvtq_n_s64_f64_(a: float64x2_t, n: i32) -> int64x2_t;
    }
    vcvtq_n_s64_f64_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvts_n_s32_f32<const N: i32>(a: f32) -> i32 {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfp2fxs.i32.f32")]
        fn vcvts_n_s32_f32_(a: f32, n: i32) -> i32;
    }
    vcvts_n_s32_f32_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtd_n_s64_f64<const N: i32>(a: f64) -> i64 {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfp2fxs.i64.f64")]
        fn vcvtd_n_s64_f64_(a: f64, n: i32) -> i64;
    }
    vcvtd_n_s64_f64_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvt_n_u64_f64<const N: i32>(a: float64x1_t) -> uint64x1_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfp2fxu.v1i64.v1f64")]
        fn vcvt_n_u64_f64_(a: float64x1_t, n: i32) -> uint64x1_t;
    }
    vcvt_n_u64_f64_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtq_n_u64_f64<const N: i32>(a: float64x2_t) -> uint64x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfp2fxu.v2i64.v2f64")]
        fn vcvtq_n_u64_f64_(a: float64x2_t, n: i32) -> uint64x2_t;
    }
    vcvtq_n_u64_f64_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvts_n_u32_f32<const N: i32>(a: f32) -> u32 {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfp2fxu.i32.f32")]
        fn vcvts_n_u32_f32_(a: f32, n: i32) -> u32;
    }
    vcvts_n_u32_f32_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtd_n_u64_f64<const N: i32>(a: f64) -> u64 {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfp2fxu.i64.f64")]
        fn vcvtd_n_u64_f64_(a: f64, n: i32) -> u64;
    }
    vcvtd_n_u64_f64_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf))]
pub unsafe fn vcvts_f32_s32(a: i32) -> f32 {
    a as f32
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf))]
pub unsafe fn vcvtd_f64_s64(a: i64) -> f64 {
    a as f64
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub unsafe fn vcvts_f32_u32(a: u32) -> f32 {
    a as f32
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub unsafe fn vcvtd_f64_u64(a: u64) -> f64 {
    a as f64
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs))]
pub unsafe fn vcvts_s32_f32(a: f32) -> i32 {
    a as i32
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs))]
pub unsafe fn vcvtd_s64_f64(a: f64) -> i64 {
    a as i64
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu))]
pub unsafe fn vcvts_u32_f32(a: f32) -> u32 {
    a as u32
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu))]
pub unsafe fn vcvtd_u64_f64(a: f64) -> u64 {
    a as u64
}

/// Floating-point convert to signed fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs))]
pub unsafe fn vcvt_s64_f64(a: float64x1_t) -> int64x1_t {
    simd_cast(a)
}

/// Floating-point convert to signed fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs))]
pub unsafe fn vcvtq_s64_f64(a: float64x2_t) -> int64x2_t {
    simd_cast(a)
}

/// Floating-point convert to unsigned fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu))]
pub unsafe fn vcvt_u64_f64(a: float64x1_t) -> uint64x1_t {
    simd_cast(a)
}

/// Floating-point convert to unsigned fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu))]
pub unsafe fn vcvtq_u64_f64(a: float64x2_t) -> uint64x2_t {
    simd_cast(a)
}

/// Floating-point convert to signed integer, rounding to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtas))]
pub unsafe fn vcvta_s32_f32(a: float32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtas.v2i32.v2f32")]
        fn vcvta_s32_f32_(a: float32x2_t) -> int32x2_t;
    }
    vcvta_s32_f32_(a)
}

/// Floating-point convert to signed integer, rounding to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtas))]
pub unsafe fn vcvtaq_s32_f32(a: float32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtas.v4i32.v4f32")]
        fn vcvtaq_s32_f32_(a: float32x4_t) -> int32x4_t;
    }
    vcvtaq_s32_f32_(a)
}

/// Floating-point convert to signed integer, rounding to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtas))]
pub unsafe fn vcvta_s64_f64(a: float64x1_t) -> int64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtas.v1i64.v1f64")]
        fn vcvta_s64_f64_(a: float64x1_t) -> int64x1_t;
    }
    vcvta_s64_f64_(a)
}

/// Floating-point convert to signed integer, rounding to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtas))]
pub unsafe fn vcvtaq_s64_f64(a: float64x2_t) -> int64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtas.v2i64.v2f64")]
        fn vcvtaq_s64_f64_(a: float64x2_t) -> int64x2_t;
    }
    vcvtaq_s64_f64_(a)
}

/// Floating-point convert to integer, rounding to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtas))]
pub unsafe fn vcvtas_s32_f32(a: f32) -> i32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtas.i32.f32")]
        fn vcvtas_s32_f32_(a: f32) -> i32;
    }
    vcvtas_s32_f32_(a)
}

/// Floating-point convert to integer, rounding to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtas))]
pub unsafe fn vcvtad_s64_f64(a: f64) -> i64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtas.i64.f64")]
        fn vcvtad_s64_f64_(a: f64) -> i64;
    }
    vcvtad_s64_f64_(a)
}

/// Floating-point convert to integer, rounding to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtau))]
pub unsafe fn vcvtas_u32_f32(a: f32) -> u32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtau.i32.f32")]
        fn vcvtas_u32_f32_(a: f32) -> u32;
    }
    vcvtas_u32_f32_(a)
}

/// Floating-point convert to integer, rounding to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtau))]
pub unsafe fn vcvtad_u64_f64(a: f64) -> u64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtau.i64.f64")]
        fn vcvtad_u64_f64_(a: f64) -> u64;
    }
    vcvtad_u64_f64_(a)
}

/// Floating-point convert to signed integer, rounding to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtns))]
pub unsafe fn vcvtn_s32_f32(a: float32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtns.v2i32.v2f32")]
        fn vcvtn_s32_f32_(a: float32x2_t) -> int32x2_t;
    }
    vcvtn_s32_f32_(a)
}

/// Floating-point convert to signed integer, rounding to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtns))]
pub unsafe fn vcvtnq_s32_f32(a: float32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtns.v4i32.v4f32")]
        fn vcvtnq_s32_f32_(a: float32x4_t) -> int32x4_t;
    }
    vcvtnq_s32_f32_(a)
}

/// Floating-point convert to signed integer, rounding to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtns))]
pub unsafe fn vcvtn_s64_f64(a: float64x1_t) -> int64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtns.v1i64.v1f64")]
        fn vcvtn_s64_f64_(a: float64x1_t) -> int64x1_t;
    }
    vcvtn_s64_f64_(a)
}

/// Floating-point convert to signed integer, rounding to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtns))]
pub unsafe fn vcvtnq_s64_f64(a: float64x2_t) -> int64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtns.v2i64.v2f64")]
        fn vcvtnq_s64_f64_(a: float64x2_t) -> int64x2_t;
    }
    vcvtnq_s64_f64_(a)
}

/// Floating-point convert to signed integer, rounding to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtns))]
pub unsafe fn vcvtns_s32_f32(a: f32) -> i32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtns.i32.f32")]
        fn vcvtns_s32_f32_(a: f32) -> i32;
    }
    vcvtns_s32_f32_(a)
}

/// Floating-point convert to signed integer, rounding to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtns))]
pub unsafe fn vcvtnd_s64_f64(a: f64) -> i64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtns.i64.f64")]
        fn vcvtnd_s64_f64_(a: f64) -> i64;
    }
    vcvtnd_s64_f64_(a)
}

/// Floating-point convert to signed integer, rounding toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtms))]
pub unsafe fn vcvtm_s32_f32(a: float32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtms.v2i32.v2f32")]
        fn vcvtm_s32_f32_(a: float32x2_t) -> int32x2_t;
    }
    vcvtm_s32_f32_(a)
}

/// Floating-point convert to signed integer, rounding toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtms))]
pub unsafe fn vcvtmq_s32_f32(a: float32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtms.v4i32.v4f32")]
        fn vcvtmq_s32_f32_(a: float32x4_t) -> int32x4_t;
    }
    vcvtmq_s32_f32_(a)
}

/// Floating-point convert to signed integer, rounding toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtms))]
pub unsafe fn vcvtm_s64_f64(a: float64x1_t) -> int64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtms.v1i64.v1f64")]
        fn vcvtm_s64_f64_(a: float64x1_t) -> int64x1_t;
    }
    vcvtm_s64_f64_(a)
}

/// Floating-point convert to signed integer, rounding toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtms))]
pub unsafe fn vcvtmq_s64_f64(a: float64x2_t) -> int64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtms.v2i64.v2f64")]
        fn vcvtmq_s64_f64_(a: float64x2_t) -> int64x2_t;
    }
    vcvtmq_s64_f64_(a)
}

/// Floating-point convert to signed integer, rounding toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtms))]
pub unsafe fn vcvtms_s32_f32(a: f32) -> i32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtms.i32.f32")]
        fn vcvtms_s32_f32_(a: f32) -> i32;
    }
    vcvtms_s32_f32_(a)
}

/// Floating-point convert to signed integer, rounding toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtms))]
pub unsafe fn vcvtmd_s64_f64(a: f64) -> i64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtms.i64.f64")]
        fn vcvtmd_s64_f64_(a: f64) -> i64;
    }
    vcvtmd_s64_f64_(a)
}

/// Floating-point convert to signed integer, rounding toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtps))]
pub unsafe fn vcvtp_s32_f32(a: float32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtps.v2i32.v2f32")]
        fn vcvtp_s32_f32_(a: float32x2_t) -> int32x2_t;
    }
    vcvtp_s32_f32_(a)
}

/// Floating-point convert to signed integer, rounding toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtps))]
pub unsafe fn vcvtpq_s32_f32(a: float32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtps.v4i32.v4f32")]
        fn vcvtpq_s32_f32_(a: float32x4_t) -> int32x4_t;
    }
    vcvtpq_s32_f32_(a)
}

/// Floating-point convert to signed integer, rounding toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtps))]
pub unsafe fn vcvtp_s64_f64(a: float64x1_t) -> int64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtps.v1i64.v1f64")]
        fn vcvtp_s64_f64_(a: float64x1_t) -> int64x1_t;
    }
    vcvtp_s64_f64_(a)
}

/// Floating-point convert to signed integer, rounding toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtps))]
pub unsafe fn vcvtpq_s64_f64(a: float64x2_t) -> int64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtps.v2i64.v2f64")]
        fn vcvtpq_s64_f64_(a: float64x2_t) -> int64x2_t;
    }
    vcvtpq_s64_f64_(a)
}

/// Floating-point convert to signed integer, rounding toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtps))]
pub unsafe fn vcvtps_s32_f32(a: f32) -> i32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtps.i32.f32")]
        fn vcvtps_s32_f32_(a: f32) -> i32;
    }
    vcvtps_s32_f32_(a)
}

/// Floating-point convert to signed integer, rounding toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtps))]
pub unsafe fn vcvtpd_s64_f64(a: f64) -> i64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtps.i64.f64")]
        fn vcvtpd_s64_f64_(a: f64) -> i64;
    }
    vcvtpd_s64_f64_(a)
}

/// Floating-point convert to unsigned integer, rounding to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtau))]
pub unsafe fn vcvta_u32_f32(a: float32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtau.v2i32.v2f32")]
        fn vcvta_u32_f32_(a: float32x2_t) -> uint32x2_t;
    }
    vcvta_u32_f32_(a)
}

/// Floating-point convert to unsigned integer, rounding to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtau))]
pub unsafe fn vcvtaq_u32_f32(a: float32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtau.v4i32.v4f32")]
        fn vcvtaq_u32_f32_(a: float32x4_t) -> uint32x4_t;
    }
    vcvtaq_u32_f32_(a)
}

/// Floating-point convert to unsigned integer, rounding to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtau))]
pub unsafe fn vcvta_u64_f64(a: float64x1_t) -> uint64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtau.v1i64.v1f64")]
        fn vcvta_u64_f64_(a: float64x1_t) -> uint64x1_t;
    }
    vcvta_u64_f64_(a)
}

/// Floating-point convert to unsigned integer, rounding to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtau))]
pub unsafe fn vcvtaq_u64_f64(a: float64x2_t) -> uint64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtau.v2i64.v2f64")]
        fn vcvtaq_u64_f64_(a: float64x2_t) -> uint64x2_t;
    }
    vcvtaq_u64_f64_(a)
}

/// Floating-point convert to unsigned integer, rounding to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtnu))]
pub unsafe fn vcvtn_u32_f32(a: float32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtnu.v2i32.v2f32")]
        fn vcvtn_u32_f32_(a: float32x2_t) -> uint32x2_t;
    }
    vcvtn_u32_f32_(a)
}

/// Floating-point convert to unsigned integer, rounding to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtnu))]
pub unsafe fn vcvtnq_u32_f32(a: float32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtnu.v4i32.v4f32")]
        fn vcvtnq_u32_f32_(a: float32x4_t) -> uint32x4_t;
    }
    vcvtnq_u32_f32_(a)
}

/// Floating-point convert to unsigned integer, rounding to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtnu))]
pub unsafe fn vcvtn_u64_f64(a: float64x1_t) -> uint64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtnu.v1i64.v1f64")]
        fn vcvtn_u64_f64_(a: float64x1_t) -> uint64x1_t;
    }
    vcvtn_u64_f64_(a)
}

/// Floating-point convert to unsigned integer, rounding to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtnu))]
pub unsafe fn vcvtnq_u64_f64(a: float64x2_t) -> uint64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtnu.v2i64.v2f64")]
        fn vcvtnq_u64_f64_(a: float64x2_t) -> uint64x2_t;
    }
    vcvtnq_u64_f64_(a)
}

/// Floating-point convert to unsigned integer, rounding to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtnu))]
pub unsafe fn vcvtns_u32_f32(a: f32) -> u32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtnu.i32.f32")]
        fn vcvtns_u32_f32_(a: f32) -> u32;
    }
    vcvtns_u32_f32_(a)
}

/// Floating-point convert to unsigned integer, rounding to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtnu))]
pub unsafe fn vcvtnd_u64_f64(a: f64) -> u64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtnu.i64.f64")]
        fn vcvtnd_u64_f64_(a: f64) -> u64;
    }
    vcvtnd_u64_f64_(a)
}

/// Floating-point convert to unsigned integer, rounding toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtmu))]
pub unsafe fn vcvtm_u32_f32(a: float32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtmu.v2i32.v2f32")]
        fn vcvtm_u32_f32_(a: float32x2_t) -> uint32x2_t;
    }
    vcvtm_u32_f32_(a)
}

/// Floating-point convert to unsigned integer, rounding toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtmu))]
pub unsafe fn vcvtmq_u32_f32(a: float32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtmu.v4i32.v4f32")]
        fn vcvtmq_u32_f32_(a: float32x4_t) -> uint32x4_t;
    }
    vcvtmq_u32_f32_(a)
}

/// Floating-point convert to unsigned integer, rounding toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtmu))]
pub unsafe fn vcvtm_u64_f64(a: float64x1_t) -> uint64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtmu.v1i64.v1f64")]
        fn vcvtm_u64_f64_(a: float64x1_t) -> uint64x1_t;
    }
    vcvtm_u64_f64_(a)
}

/// Floating-point convert to unsigned integer, rounding toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtmu))]
pub unsafe fn vcvtmq_u64_f64(a: float64x2_t) -> uint64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtmu.v2i64.v2f64")]
        fn vcvtmq_u64_f64_(a: float64x2_t) -> uint64x2_t;
    }
    vcvtmq_u64_f64_(a)
}

/// Floating-point convert to unsigned integer, rounding toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtmu))]
pub unsafe fn vcvtms_u32_f32(a: f32) -> u32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtmu.i32.f32")]
        fn vcvtms_u32_f32_(a: f32) -> u32;
    }
    vcvtms_u32_f32_(a)
}

/// Floating-point convert to unsigned integer, rounding toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtmu))]
pub unsafe fn vcvtmd_u64_f64(a: f64) -> u64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtmu.i64.f64")]
        fn vcvtmd_u64_f64_(a: f64) -> u64;
    }
    vcvtmd_u64_f64_(a)
}

/// Floating-point convert to unsigned integer, rounding toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtpu))]
pub unsafe fn vcvtp_u32_f32(a: float32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtpu.v2i32.v2f32")]
        fn vcvtp_u32_f32_(a: float32x2_t) -> uint32x2_t;
    }
    vcvtp_u32_f32_(a)
}

/// Floating-point convert to unsigned integer, rounding toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtpu))]
pub unsafe fn vcvtpq_u32_f32(a: float32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtpu.v4i32.v4f32")]
        fn vcvtpq_u32_f32_(a: float32x4_t) -> uint32x4_t;
    }
    vcvtpq_u32_f32_(a)
}

/// Floating-point convert to unsigned integer, rounding toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtpu))]
pub unsafe fn vcvtp_u64_f64(a: float64x1_t) -> uint64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtpu.v1i64.v1f64")]
        fn vcvtp_u64_f64_(a: float64x1_t) -> uint64x1_t;
    }
    vcvtp_u64_f64_(a)
}

/// Floating-point convert to unsigned integer, rounding toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtpu))]
pub unsafe fn vcvtpq_u64_f64(a: float64x2_t) -> uint64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtpu.v2i64.v2f64")]
        fn vcvtpq_u64_f64_(a: float64x2_t) -> uint64x2_t;
    }
    vcvtpq_u64_f64_(a)
}

/// Floating-point convert to unsigned integer, rounding toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtpu))]
pub unsafe fn vcvtps_u32_f32(a: f32) -> u32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtpu.i32.f32")]
        fn vcvtps_u32_f32_(a: f32) -> u32;
    }
    vcvtps_u32_f32_(a)
}

/// Floating-point convert to unsigned integer, rounding toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtpu))]
pub unsafe fn vcvtpd_u64_f64(a: f64) -> u64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fcvtpu.i64.f64")]
        fn vcvtpd_u64_f64_(a: f64) -> u64;
    }
    vcvtpd_u64_f64_(a)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(dup, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_laneq_p64<const N: i32>(a: poly64x2_t) -> poly64x2_t {
    static_assert_imm1!(N);
    simd_shuffle2!(a, a, <const N: i32> [N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(dup, N = 0))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_lane_p64<const N: i32>(a: poly64x1_t) -> poly64x2_t {
    static_assert!(N : i32 where N == 0);
    simd_shuffle2!(a, a, <const N: i32> [N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(dup, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_laneq_f64<const N: i32>(a: float64x2_t) -> float64x2_t {
    static_assert_imm1!(N);
    simd_shuffle2!(a, a, <const N: i32> [N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(dup, N = 0))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_lane_f64<const N: i32>(a: float64x1_t) -> float64x2_t {
    static_assert!(N : i32 where N == 0);
    simd_shuffle2!(a, a, <const N: i32> [N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 0))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_lane_p64<const N: i32>(a: poly64x1_t) -> poly64x1_t {
    static_assert!(N : i32 where N == 0);
    a
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 0))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_lane_f64<const N: i32>(a: float64x1_t) -> float64x1_t {
    static_assert!(N : i32 where N == 0);
    a
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_laneq_p64<const N: i32>(a: poly64x2_t) -> poly64x1_t {
    static_assert_imm1!(N);
    transmute::<u64, _>(simd_extract(a, N as u32))
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_laneq_f64<const N: i32>(a: float64x2_t) -> float64x1_t {
    static_assert_imm1!(N);
    transmute::<f64, _>(simd_extract(a, N as u32))
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupb_lane_s8<const N: i32>(a: int8x8_t) -> i8 {
    static_assert_imm3!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 8))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupb_laneq_s8<const N: i32>(a: int8x16_t) -> i8 {
    static_assert_imm4!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vduph_lane_s16<const N: i32>(a: int16x4_t) -> i16 {
    static_assert_imm2!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vduph_laneq_s16<const N: i32>(a: int16x8_t) -> i16 {
    static_assert_imm3!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdups_lane_s32<const N: i32>(a: int32x2_t) -> i32 {
    static_assert_imm1!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdups_laneq_s32<const N: i32>(a: int32x4_t) -> i32 {
    static_assert_imm2!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 0))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupd_lane_s64<const N: i32>(a: int64x1_t) -> i64 {
    static_assert!(N : i32 where N == 0);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupd_laneq_s64<const N: i32>(a: int64x2_t) -> i64 {
    static_assert_imm1!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupb_lane_u8<const N: i32>(a: uint8x8_t) -> u8 {
    static_assert_imm3!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 8))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupb_laneq_u8<const N: i32>(a: uint8x16_t) -> u8 {
    static_assert_imm4!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vduph_lane_u16<const N: i32>(a: uint16x4_t) -> u16 {
    static_assert_imm2!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vduph_laneq_u16<const N: i32>(a: uint16x8_t) -> u16 {
    static_assert_imm3!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdups_lane_u32<const N: i32>(a: uint32x2_t) -> u32 {
    static_assert_imm1!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdups_laneq_u32<const N: i32>(a: uint32x4_t) -> u32 {
    static_assert_imm2!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 0))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupd_lane_u64<const N: i32>(a: uint64x1_t) -> u64 {
    static_assert!(N : i32 where N == 0);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupd_laneq_u64<const N: i32>(a: uint64x2_t) -> u64 {
    static_assert_imm1!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupb_lane_p8<const N: i32>(a: poly8x8_t) -> p8 {
    static_assert_imm3!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 8))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupb_laneq_p8<const N: i32>(a: poly8x16_t) -> p8 {
    static_assert_imm4!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vduph_lane_p16<const N: i32>(a: poly16x4_t) -> p16 {
    static_assert_imm2!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vduph_laneq_p16<const N: i32>(a: poly16x8_t) -> p16 {
    static_assert_imm3!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdups_lane_f32<const N: i32>(a: float32x2_t) -> f32 {
    static_assert_imm1!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdups_laneq_f32<const N: i32>(a: float32x4_t) -> f32 {
    static_assert_imm2!(N);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 0))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupd_lane_f64<const N: i32>(a: float64x1_t) -> f64 {
    static_assert!(N : i32 where N == 0);
    simd_extract(a, N as u32)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupd_laneq_f64<const N: i32>(a: float64x2_t) -> f64 {
    static_assert_imm1!(N);
    simd_extract(a, N as u32)
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ext, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vextq_p64<const N: i32>(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    static_assert_imm1!(N);
    match N & 0b1 {
        0 => simd_shuffle2!(a, b, [0, 1]),
        1 => simd_shuffle2!(a, b, [1, 2]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ext, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vextq_f64<const N: i32>(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    static_assert_imm1!(N);
    match N & 0b1 {
        0 => simd_shuffle2!(a, b, [0, 1]),
        1 => simd_shuffle2!(a, b, [1, 2]),
        _ => unreachable_unchecked(),
    }
}

/// Floating-point multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul))]
pub unsafe fn vmla_f64(a: float64x1_t, b: float64x1_t, c: float64x1_t) -> float64x1_t {
    simd_add(a, simd_mul(b, c))
}

/// Floating-point multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul))]
pub unsafe fn vmlaq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    simd_add(a, simd_mul(b, c))
}

/// Signed multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2))]
pub unsafe fn vmlal_high_s8(a: int16x8_t, b: int8x16_t, c: int8x16_t) -> int16x8_t {
    let b: int8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let c: int8x8_t = simd_shuffle8!(c, c, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmlal_s8(a, b, c)
}

/// Signed multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2))]
pub unsafe fn vmlal_high_s16(a: int32x4_t, b: int16x8_t, c: int16x8_t) -> int32x4_t {
    let b: int16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    let c: int16x4_t = simd_shuffle4!(c, c, [4, 5, 6, 7]);
    vmlal_s16(a, b, c)
}

/// Signed multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2))]
pub unsafe fn vmlal_high_s32(a: int64x2_t, b: int32x4_t, c: int32x4_t) -> int64x2_t {
    let b: int32x2_t = simd_shuffle2!(b, b, [2, 3]);
    let c: int32x2_t = simd_shuffle2!(c, c, [2, 3]);
    vmlal_s32(a, b, c)
}

/// Unsigned multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2))]
pub unsafe fn vmlal_high_u8(a: uint16x8_t, b: uint8x16_t, c: uint8x16_t) -> uint16x8_t {
    let b: uint8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let c: uint8x8_t = simd_shuffle8!(c, c, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmlal_u8(a, b, c)
}

/// Unsigned multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2))]
pub unsafe fn vmlal_high_u16(a: uint32x4_t, b: uint16x8_t, c: uint16x8_t) -> uint32x4_t {
    let b: uint16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    let c: uint16x4_t = simd_shuffle4!(c, c, [4, 5, 6, 7]);
    vmlal_u16(a, b, c)
}

/// Unsigned multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2))]
pub unsafe fn vmlal_high_u32(a: uint64x2_t, b: uint32x4_t, c: uint32x4_t) -> uint64x2_t {
    let b: uint32x2_t = simd_shuffle2!(b, b, [2, 3]);
    let c: uint32x2_t = simd_shuffle2!(c, c, [2, 3]);
    vmlal_u32(a, b, c)
}

/// Multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2))]
pub unsafe fn vmlal_high_n_s16(a: int32x4_t, b: int16x8_t, c: i16) -> int32x4_t {
    vmlal_high_s16(a, b, vdupq_n_s16(c))
}

/// Multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2))]
pub unsafe fn vmlal_high_n_s32(a: int64x2_t, b: int32x4_t, c: i32) -> int64x2_t {
    vmlal_high_s32(a, b, vdupq_n_s32(c))
}

/// Multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2))]
pub unsafe fn vmlal_high_n_u16(a: uint32x4_t, b: uint16x8_t, c: u16) -> uint32x4_t {
    vmlal_high_u16(a, b, vdupq_n_u16(c))
}

/// Multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2))]
pub unsafe fn vmlal_high_n_u32(a: uint64x2_t, b: uint32x4_t, c: u32) -> uint64x2_t {
    vmlal_high_u32(a, b, vdupq_n_u32(c))
}

/// Multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_high_lane_s16<const LANE: i32>(a: int32x4_t, b: int16x8_t, c: int16x4_t) -> int32x4_t {
    static_assert_imm2!(LANE);
    vmlal_high_s16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_high_laneq_s16<const LANE: i32>(a: int32x4_t, b: int16x8_t, c: int16x8_t) -> int32x4_t {
    static_assert_imm3!(LANE);
    vmlal_high_s16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_high_lane_s32<const LANE: i32>(a: int64x2_t, b: int32x4_t, c: int32x2_t) -> int64x2_t {
    static_assert_imm1!(LANE);
    vmlal_high_s32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_high_laneq_s32<const LANE: i32>(a: int64x2_t, b: int32x4_t, c: int32x4_t) -> int64x2_t {
    static_assert_imm2!(LANE);
    vmlal_high_s32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_high_lane_u16<const LANE: i32>(a: uint32x4_t, b: uint16x8_t, c: uint16x4_t) -> uint32x4_t {
    static_assert_imm2!(LANE);
    vmlal_high_u16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_high_laneq_u16<const LANE: i32>(a: uint32x4_t, b: uint16x8_t, c: uint16x8_t) -> uint32x4_t {
    static_assert_imm3!(LANE);
    vmlal_high_u16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_high_lane_u32<const LANE: i32>(a: uint64x2_t, b: uint32x4_t, c: uint32x2_t) -> uint64x2_t {
    static_assert_imm1!(LANE);
    vmlal_high_u32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_high_laneq_u32<const LANE: i32>(a: uint64x2_t, b: uint32x4_t, c: uint32x4_t) -> uint64x2_t {
    static_assert_imm2!(LANE);
    vmlal_high_u32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Floating-point multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul))]
pub unsafe fn vmls_f64(a: float64x1_t, b: float64x1_t, c: float64x1_t) -> float64x1_t {
    simd_sub(a, simd_mul(b, c))
}

/// Floating-point multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul))]
pub unsafe fn vmlsq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    simd_sub(a, simd_mul(b, c))
}

/// Signed multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2))]
pub unsafe fn vmlsl_high_s8(a: int16x8_t, b: int8x16_t, c: int8x16_t) -> int16x8_t {
    let b: int8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let c: int8x8_t = simd_shuffle8!(c, c, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmlsl_s8(a, b, c)
}

/// Signed multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2))]
pub unsafe fn vmlsl_high_s16(a: int32x4_t, b: int16x8_t, c: int16x8_t) -> int32x4_t {
    let b: int16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    let c: int16x4_t = simd_shuffle4!(c, c, [4, 5, 6, 7]);
    vmlsl_s16(a, b, c)
}

/// Signed multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2))]
pub unsafe fn vmlsl_high_s32(a: int64x2_t, b: int32x4_t, c: int32x4_t) -> int64x2_t {
    let b: int32x2_t = simd_shuffle2!(b, b, [2, 3]);
    let c: int32x2_t = simd_shuffle2!(c, c, [2, 3]);
    vmlsl_s32(a, b, c)
}

/// Unsigned multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2))]
pub unsafe fn vmlsl_high_u8(a: uint16x8_t, b: uint8x16_t, c: uint8x16_t) -> uint16x8_t {
    let b: uint8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let c: uint8x8_t = simd_shuffle8!(c, c, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmlsl_u8(a, b, c)
}

/// Unsigned multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2))]
pub unsafe fn vmlsl_high_u16(a: uint32x4_t, b: uint16x8_t, c: uint16x8_t) -> uint32x4_t {
    let b: uint16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    let c: uint16x4_t = simd_shuffle4!(c, c, [4, 5, 6, 7]);
    vmlsl_u16(a, b, c)
}

/// Unsigned multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2))]
pub unsafe fn vmlsl_high_u32(a: uint64x2_t, b: uint32x4_t, c: uint32x4_t) -> uint64x2_t {
    let b: uint32x2_t = simd_shuffle2!(b, b, [2, 3]);
    let c: uint32x2_t = simd_shuffle2!(c, c, [2, 3]);
    vmlsl_u32(a, b, c)
}

/// Multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2))]
pub unsafe fn vmlsl_high_n_s16(a: int32x4_t, b: int16x8_t, c: i16) -> int32x4_t {
    vmlsl_high_s16(a, b, vdupq_n_s16(c))
}

/// Multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2))]
pub unsafe fn vmlsl_high_n_s32(a: int64x2_t, b: int32x4_t, c: i32) -> int64x2_t {
    vmlsl_high_s32(a, b, vdupq_n_s32(c))
}

/// Multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2))]
pub unsafe fn vmlsl_high_n_u16(a: uint32x4_t, b: uint16x8_t, c: u16) -> uint32x4_t {
    vmlsl_high_u16(a, b, vdupq_n_u16(c))
}

/// Multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2))]
pub unsafe fn vmlsl_high_n_u32(a: uint64x2_t, b: uint32x4_t, c: u32) -> uint64x2_t {
    vmlsl_high_u32(a, b, vdupq_n_u32(c))
}

/// Multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_high_lane_s16<const LANE: i32>(a: int32x4_t, b: int16x8_t, c: int16x4_t) -> int32x4_t {
    static_assert_imm2!(LANE);
    vmlsl_high_s16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_high_laneq_s16<const LANE: i32>(a: int32x4_t, b: int16x8_t, c: int16x8_t) -> int32x4_t {
    static_assert_imm3!(LANE);
    vmlsl_high_s16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_high_lane_s32<const LANE: i32>(a: int64x2_t, b: int32x4_t, c: int32x2_t) -> int64x2_t {
    static_assert_imm1!(LANE);
    vmlsl_high_s32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_high_laneq_s32<const LANE: i32>(a: int64x2_t, b: int32x4_t, c: int32x4_t) -> int64x2_t {
    static_assert_imm2!(LANE);
    vmlsl_high_s32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_high_lane_u16<const LANE: i32>(a: uint32x4_t, b: uint16x8_t, c: uint16x4_t) -> uint32x4_t {
    static_assert_imm2!(LANE);
    vmlsl_high_u16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_high_laneq_u16<const LANE: i32>(a: uint32x4_t, b: uint16x8_t, c: uint16x8_t) -> uint32x4_t {
    static_assert_imm3!(LANE);
    vmlsl_high_u16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_high_lane_u32<const LANE: i32>(a: uint64x2_t, b: uint32x4_t, c: uint32x2_t) -> uint64x2_t {
    static_assert_imm1!(LANE);
    vmlsl_high_u32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_high_laneq_u32<const LANE: i32>(a: uint64x2_t, b: uint32x4_t, c: uint32x4_t) -> uint64x2_t {
    static_assert_imm2!(LANE);
    vmlsl_high_u32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(xtn2))]
pub unsafe fn vmovn_high_s16(a: int8x8_t, b: int16x8_t) -> int8x16_t {
    let c: int8x8_t = simd_cast(b);
    simd_shuffle16!(a, c, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(xtn2))]
pub unsafe fn vmovn_high_s32(a: int16x4_t, b: int32x4_t) -> int16x8_t {
    let c: int16x4_t = simd_cast(b);
    simd_shuffle8!(a, c, [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(xtn2))]
pub unsafe fn vmovn_high_s64(a: int32x2_t, b: int64x2_t) -> int32x4_t {
    let c: int32x2_t = simd_cast(b);
    simd_shuffle4!(a, c, [0, 1, 2, 3])
}

/// Extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(xtn2))]
pub unsafe fn vmovn_high_u16(a: uint8x8_t, b: uint16x8_t) -> uint8x16_t {
    let c: uint8x8_t = simd_cast(b);
    simd_shuffle16!(a, c, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(xtn2))]
pub unsafe fn vmovn_high_u32(a: uint16x4_t, b: uint32x4_t) -> uint16x8_t {
    let c: uint16x4_t = simd_cast(b);
    simd_shuffle8!(a, c, [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(xtn2))]
pub unsafe fn vmovn_high_u64(a: uint32x2_t, b: uint64x2_t) -> uint32x4_t {
    let c: uint32x2_t = simd_cast(b);
    simd_shuffle4!(a, c, [0, 1, 2, 3])
}

/// Negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(neg))]
pub unsafe fn vneg_s64(a: int64x1_t) -> int64x1_t {
    simd_neg(a)
}

/// Negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(neg))]
pub unsafe fn vnegq_s64(a: int64x2_t) -> int64x2_t {
    simd_neg(a)
}

/// Negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fneg))]
pub unsafe fn vneg_f64(a: float64x1_t) -> float64x1_t {
    simd_neg(a)
}

/// Negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fneg))]
pub unsafe fn vnegq_f64(a: float64x2_t) -> float64x2_t {
    simd_neg(a)
}

/// Signed saturating negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqneg))]
pub unsafe fn vqneg_s64(a: int64x1_t) -> int64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqneg.v1i64")]
        fn vqneg_s64_(a: int64x1_t) -> int64x1_t;
    }
    vqneg_s64_(a)
}

/// Signed saturating negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqneg))]
pub unsafe fn vqnegq_s64(a: int64x2_t) -> int64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqneg.v2i64")]
        fn vqnegq_s64_(a: int64x2_t) -> int64x2_t;
    }
    vqnegq_s64_(a)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqsub))]
pub unsafe fn vqsubb_s8(a: i8, b: i8) -> i8 {
    let a: int8x8_t = vdup_n_s8(a);
    let b: int8x8_t = vdup_n_s8(b);
    simd_extract(vqsub_s8(a, b), 0)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqsub))]
pub unsafe fn vqsubh_s16(a: i16, b: i16) -> i16 {
    let a: int16x4_t = vdup_n_s16(a);
    let b: int16x4_t = vdup_n_s16(b);
    simd_extract(vqsub_s16(a, b), 0)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqsub))]
pub unsafe fn vqsubb_u8(a: u8, b: u8) -> u8 {
    let a: uint8x8_t = vdup_n_u8(a);
    let b: uint8x8_t = vdup_n_u8(b);
    simd_extract(vqsub_u8(a, b), 0)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqsub))]
pub unsafe fn vqsubh_u16(a: u16, b: u16) -> u16 {
    let a: uint16x4_t = vdup_n_u16(a);
    let b: uint16x4_t = vdup_n_u16(b);
    simd_extract(vqsub_u16(a, b), 0)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqsub))]
pub unsafe fn vqsubs_u32(a: u32, b: u32) -> u32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.i32")]
        fn vqsubs_u32_(a: u32, b: u32) -> u32;
    }
    vqsubs_u32_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqsub))]
pub unsafe fn vqsubd_u64(a: u64, b: u64) -> u64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.i64")]
        fn vqsubd_u64_(a: u64, b: u64) -> u64;
    }
    vqsubd_u64_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqsub))]
pub unsafe fn vqsubs_s32(a: i32, b: i32) -> i32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.i32")]
        fn vqsubs_s32_(a: i32, b: i32) -> i32;
    }
    vqsubs_s32_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqsub))]
pub unsafe fn vqsubd_s64(a: i64, b: i64) -> i64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.i64")]
        fn vqsubd_s64_(a: i64, b: i64) -> i64;
    }
    vqsubd_s64_(a, b)
}

/// Reverse bit order
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rbit))]
pub unsafe fn vrbit_s8(a: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.rbit.v8i8")]
        fn vrbit_s8_(a: int8x8_t) -> int8x8_t;
    }
    vrbit_s8_(a)
}

/// Reverse bit order
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rbit))]
pub unsafe fn vrbitq_s8(a: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.rbit.v16i8")]
        fn vrbitq_s8_(a: int8x16_t) -> int8x16_t;
    }
    vrbitq_s8_(a)
}

/// Reverse bit order
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rbit))]
pub unsafe fn vrbit_u8(a: uint8x8_t) -> uint8x8_t {
    transmute(vrbit_s8(transmute(a)))
}

/// Reverse bit order
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rbit))]
pub unsafe fn vrbitq_u8(a: uint8x16_t) -> uint8x16_t {
    transmute(vrbitq_s8(transmute(a)))
}

/// Reverse bit order
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rbit))]
pub unsafe fn vrbit_p8(a: poly8x8_t) -> poly8x8_t {
    transmute(vrbit_s8(transmute(a)))
}

/// Reverse bit order
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rbit))]
pub unsafe fn vrbitq_p8(a: poly8x16_t) -> poly8x16_t {
    transmute(vrbitq_s8(transmute(a)))
}

/// Floating-point round to integral exact, using current rounding mode
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintx))]
pub unsafe fn vrndx_f32(a: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.rint.v2f32")]
        fn vrndx_f32_(a: float32x2_t) -> float32x2_t;
    }
    vrndx_f32_(a)
}

/// Floating-point round to integral exact, using current rounding mode
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintx))]
pub unsafe fn vrndxq_f32(a: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.rint.v4f32")]
        fn vrndxq_f32_(a: float32x4_t) -> float32x4_t;
    }
    vrndxq_f32_(a)
}

/// Floating-point round to integral exact, using current rounding mode
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintx))]
pub unsafe fn vrndx_f64(a: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.rint.v1f64")]
        fn vrndx_f64_(a: float64x1_t) -> float64x1_t;
    }
    vrndx_f64_(a)
}

/// Floating-point round to integral exact, using current rounding mode
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintx))]
pub unsafe fn vrndxq_f64(a: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.rint.v2f64")]
        fn vrndxq_f64_(a: float64x2_t) -> float64x2_t;
    }
    vrndxq_f64_(a)
}

/// Floating-point round to integral, to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frinta))]
pub unsafe fn vrnda_f32(a: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.round.v2f32")]
        fn vrnda_f32_(a: float32x2_t) -> float32x2_t;
    }
    vrnda_f32_(a)
}

/// Floating-point round to integral, to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frinta))]
pub unsafe fn vrndaq_f32(a: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.round.v4f32")]
        fn vrndaq_f32_(a: float32x4_t) -> float32x4_t;
    }
    vrndaq_f32_(a)
}

/// Floating-point round to integral, to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frinta))]
pub unsafe fn vrnda_f64(a: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.round.v1f64")]
        fn vrnda_f64_(a: float64x1_t) -> float64x1_t;
    }
    vrnda_f64_(a)
}

/// Floating-point round to integral, to nearest with ties to away
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frinta))]
pub unsafe fn vrndaq_f64(a: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.round.v2f64")]
        fn vrndaq_f64_(a: float64x2_t) -> float64x2_t;
    }
    vrndaq_f64_(a)
}

/// Floating-point round to integral, to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintn))]
pub unsafe fn vrndn_f64(a: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.frintn.v1f64")]
        fn vrndn_f64_(a: float64x1_t) -> float64x1_t;
    }
    vrndn_f64_(a)
}

/// Floating-point round to integral, to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintn))]
pub unsafe fn vrndnq_f64(a: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.frintn.v2f64")]
        fn vrndnq_f64_(a: float64x2_t) -> float64x2_t;
    }
    vrndnq_f64_(a)
}

/// Floating-point round to integral, toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintm))]
pub unsafe fn vrndm_f32(a: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.floor.v2f32")]
        fn vrndm_f32_(a: float32x2_t) -> float32x2_t;
    }
    vrndm_f32_(a)
}

/// Floating-point round to integral, toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintm))]
pub unsafe fn vrndmq_f32(a: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.floor.v4f32")]
        fn vrndmq_f32_(a: float32x4_t) -> float32x4_t;
    }
    vrndmq_f32_(a)
}

/// Floating-point round to integral, toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintm))]
pub unsafe fn vrndm_f64(a: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.floor.v1f64")]
        fn vrndm_f64_(a: float64x1_t) -> float64x1_t;
    }
    vrndm_f64_(a)
}

/// Floating-point round to integral, toward minus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintm))]
pub unsafe fn vrndmq_f64(a: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.floor.v2f64")]
        fn vrndmq_f64_(a: float64x2_t) -> float64x2_t;
    }
    vrndmq_f64_(a)
}

/// Floating-point round to integral, toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintp))]
pub unsafe fn vrndp_f32(a: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.ceil.v2f32")]
        fn vrndp_f32_(a: float32x2_t) -> float32x2_t;
    }
    vrndp_f32_(a)
}

/// Floating-point round to integral, toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintp))]
pub unsafe fn vrndpq_f32(a: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.ceil.v4f32")]
        fn vrndpq_f32_(a: float32x4_t) -> float32x4_t;
    }
    vrndpq_f32_(a)
}

/// Floating-point round to integral, toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintp))]
pub unsafe fn vrndp_f64(a: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.ceil.v1f64")]
        fn vrndp_f64_(a: float64x1_t) -> float64x1_t;
    }
    vrndp_f64_(a)
}

/// Floating-point round to integral, toward plus infinity
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintp))]
pub unsafe fn vrndpq_f64(a: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.ceil.v2f64")]
        fn vrndpq_f64_(a: float64x2_t) -> float64x2_t;
    }
    vrndpq_f64_(a)
}

/// Floating-point round to integral, toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintz))]
pub unsafe fn vrnd_f32(a: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.trunc.v2f32")]
        fn vrnd_f32_(a: float32x2_t) -> float32x2_t;
    }
    vrnd_f32_(a)
}

/// Floating-point round to integral, toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintz))]
pub unsafe fn vrndq_f32(a: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.trunc.v4f32")]
        fn vrndq_f32_(a: float32x4_t) -> float32x4_t;
    }
    vrndq_f32_(a)
}

/// Floating-point round to integral, toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintz))]
pub unsafe fn vrnd_f64(a: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.trunc.v1f64")]
        fn vrnd_f64_(a: float64x1_t) -> float64x1_t;
    }
    vrnd_f64_(a)
}

/// Floating-point round to integral, toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frintz))]
pub unsafe fn vrndq_f64(a: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.trunc.v2f64")]
        fn vrndq_f64_(a: float64x2_t) -> float64x2_t;
    }
    vrndq_f64_(a)
}

/// Floating-point round to integral, using current rounding mode
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frinti))]
pub unsafe fn vrndi_f32(a: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.nearbyint.v2f32")]
        fn vrndi_f32_(a: float32x2_t) -> float32x2_t;
    }
    vrndi_f32_(a)
}

/// Floating-point round to integral, using current rounding mode
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frinti))]
pub unsafe fn vrndiq_f32(a: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.nearbyint.v4f32")]
        fn vrndiq_f32_(a: float32x4_t) -> float32x4_t;
    }
    vrndiq_f32_(a)
}

/// Floating-point round to integral, using current rounding mode
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frinti))]
pub unsafe fn vrndi_f64(a: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.nearbyint.v1f64")]
        fn vrndi_f64_(a: float64x1_t) -> float64x1_t;
    }
    vrndi_f64_(a)
}

/// Floating-point round to integral, using current rounding mode
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frinti))]
pub unsafe fn vrndiq_f64(a: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.nearbyint.v2f64")]
        fn vrndiq_f64_(a: float64x2_t) -> float64x2_t;
    }
    vrndiq_f64_(a)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqadd))]
pub unsafe fn vqaddb_s8(a: i8, b: i8) -> i8 {
    let a: int8x8_t = vdup_n_s8(a);
    let b: int8x8_t = vdup_n_s8(b);
    simd_extract(vqadd_s8(a, b), 0)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqadd))]
pub unsafe fn vqaddh_s16(a: i16, b: i16) -> i16 {
    let a: int16x4_t = vdup_n_s16(a);
    let b: int16x4_t = vdup_n_s16(b);
    simd_extract(vqadd_s16(a, b), 0)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqadd))]
pub unsafe fn vqaddb_u8(a: u8, b: u8) -> u8 {
    let a: uint8x8_t = vdup_n_u8(a);
    let b: uint8x8_t = vdup_n_u8(b);
    simd_extract(vqadd_u8(a, b), 0)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqadd))]
pub unsafe fn vqaddh_u16(a: u16, b: u16) -> u16 {
    let a: uint16x4_t = vdup_n_u16(a);
    let b: uint16x4_t = vdup_n_u16(b);
    simd_extract(vqadd_u16(a, b), 0)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqadd))]
pub unsafe fn vqadds_u32(a: u32, b: u32) -> u32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.i32")]
        fn vqadds_u32_(a: u32, b: u32) -> u32;
    }
    vqadds_u32_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqadd))]
pub unsafe fn vqaddd_u64(a: u64, b: u64) -> u64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.i64")]
        fn vqaddd_u64_(a: u64, b: u64) -> u64;
    }
    vqaddd_u64_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqadd))]
pub unsafe fn vqadds_s32(a: i32, b: i32) -> i32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.i32")]
        fn vqadds_s32_(a: i32, b: i32) -> i32;
    }
    vqadds_s32_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqadd))]
pub unsafe fn vqaddd_s64(a: i64, b: i64) -> i64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.i64")]
        fn vqaddd_s64_(a: i64, b: i64) -> i64;
    }
    vqaddd_s64_(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul))]
pub unsafe fn vmul_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul))]
pub unsafe fn vmulq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_mul(a, b)
}

/// Vector multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul))]
pub unsafe fn vmul_n_f64(a: float64x1_t, b: f64) -> float64x1_t {
    simd_mul(a, vdup_n_f64(b))
}

/// Vector multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul))]
pub unsafe fn vmulq_n_f64(a: float64x2_t, b: f64) -> float64x2_t {
    simd_mul(a, vdupq_n_f64(b))
}

/// Floating-point multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmul_lane_f64<const LANE: i32>(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    static_assert!(LANE : i32 where LANE == 0);
    simd_mul(a, transmute::<f64, _>(simd_extract(b, LANE as u32)))
}

/// Floating-point multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmul_laneq_f64<const LANE: i32>(a: float64x1_t, b: float64x2_t) -> float64x1_t {
    static_assert_imm1!(LANE);
    simd_mul(a, transmute::<f64, _>(simd_extract(b, LANE as u32)))
}

/// Floating-point multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulq_lane_f64<const LANE: i32>(a: float64x2_t, b: float64x1_t) -> float64x2_t {
    static_assert!(LANE : i32 where LANE == 0);
    simd_mul(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Floating-point multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulq_laneq_f64<const LANE: i32>(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    static_assert_imm1!(LANE);
    simd_mul(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Floating-point multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmuls_lane_f32<const LANE: i32>(a: f32, b: float32x2_t) -> f32 {
    static_assert_imm1!(LANE);
    let b: f32 = simd_extract(b, LANE as u32);
    a * b
}

/// Floating-point multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmuls_laneq_f32<const LANE: i32>(a: f32, b: float32x4_t) -> f32 {
    static_assert_imm2!(LANE);
    let b: f32 = simd_extract(b, LANE as u32);
    a * b
}

/// Floating-point multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmuld_lane_f64<const LANE: i32>(a: f64, b: float64x1_t) -> f64 {
    static_assert!(LANE : i32 where LANE == 0);
    let b: f64 = simd_extract(b, LANE as u32);
    a * b
}

/// Floating-point multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmuld_laneq_f64<const LANE: i32>(a: f64, b: float64x2_t) -> f64 {
    static_assert_imm1!(LANE);
    let b: f64 = simd_extract(b, LANE as u32);
    a * b
}

/// Signed multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2))]
pub unsafe fn vmull_high_s8(a: int8x16_t, b: int8x16_t) -> int16x8_t {
    let a: int8x8_t = simd_shuffle8!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let b: int8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmull_s8(a, b)
}

/// Signed multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2))]
pub unsafe fn vmull_high_s16(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    let a: int16x4_t = simd_shuffle4!(a, a, [4, 5, 6, 7]);
    let b: int16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    vmull_s16(a, b)
}

/// Signed multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2))]
pub unsafe fn vmull_high_s32(a: int32x4_t, b: int32x4_t) -> int64x2_t {
    let a: int32x2_t = simd_shuffle2!(a, a, [2, 3]);
    let b: int32x2_t = simd_shuffle2!(b, b, [2, 3]);
    vmull_s32(a, b)
}

/// Unsigned multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2))]
pub unsafe fn vmull_high_u8(a: uint8x16_t, b: uint8x16_t) -> uint16x8_t {
    let a: uint8x8_t = simd_shuffle8!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let b: uint8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmull_u8(a, b)
}

/// Unsigned multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2))]
pub unsafe fn vmull_high_u16(a: uint16x8_t, b: uint16x8_t) -> uint32x4_t {
    let a: uint16x4_t = simd_shuffle4!(a, a, [4, 5, 6, 7]);
    let b: uint16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    vmull_u16(a, b)
}

/// Unsigned multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2))]
pub unsafe fn vmull_high_u32(a: uint32x4_t, b: uint32x4_t) -> uint64x2_t {
    let a: uint32x2_t = simd_shuffle2!(a, a, [2, 3]);
    let b: uint32x2_t = simd_shuffle2!(b, b, [2, 3]);
    vmull_u32(a, b)
}

/// Polynomial multiply long
#[inline]
#[target_feature(enable = "neon,crypto")]
#[cfg_attr(test, assert_instr(pmull))]
pub unsafe fn vmull_p64(a: p64, b: p64) -> p128 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.pmull64")]
        fn vmull_p64_(a: p64, b: p64) -> int8x16_t;
    }
    transmute(vmull_p64_(a, b))
}

/// Polynomial multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(pmull))]
pub unsafe fn vmull_high_p8(a: poly8x16_t, b: poly8x16_t) -> poly16x8_t {
    let a: poly8x8_t = simd_shuffle8!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let b: poly8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmull_p8(a, b)
}

/// Polynomial multiply long
#[inline]
#[target_feature(enable = "neon,crypto")]
#[cfg_attr(test, assert_instr(pmull))]
pub unsafe fn vmull_high_p64(a: poly64x2_t, b: poly64x2_t) -> p128 {
    vmull_p64(simd_extract(a, 1), simd_extract(b, 1))
}

/// Multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2))]
pub unsafe fn vmull_high_n_s16(a: int16x8_t, b: i16) -> int32x4_t {
    vmull_high_s16(a, vdupq_n_s16(b))
}

/// Multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2))]
pub unsafe fn vmull_high_n_s32(a: int32x4_t, b: i32) -> int64x2_t {
    vmull_high_s32(a, vdupq_n_s32(b))
}

/// Multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2))]
pub unsafe fn vmull_high_n_u16(a: uint16x8_t, b: u16) -> uint32x4_t {
    vmull_high_u16(a, vdupq_n_u16(b))
}

/// Multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2))]
pub unsafe fn vmull_high_n_u32(a: uint32x4_t, b: u32) -> uint64x2_t {
    vmull_high_u32(a, vdupq_n_u32(b))
}

/// Multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_high_lane_s16<const LANE: i32>(a: int16x8_t, b: int16x4_t) -> int32x4_t {
    static_assert_imm2!(LANE);
    vmull_high_s16(a, simd_shuffle8!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_high_laneq_s16<const LANE: i32>(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    static_assert_imm3!(LANE);
    vmull_high_s16(a, simd_shuffle8!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_high_lane_s32<const LANE: i32>(a: int32x4_t, b: int32x2_t) -> int64x2_t {
    static_assert_imm1!(LANE);
    vmull_high_s32(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_high_laneq_s32<const LANE: i32>(a: int32x4_t, b: int32x4_t) -> int64x2_t {
    static_assert_imm2!(LANE);
    vmull_high_s32(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_high_lane_u16<const LANE: i32>(a: uint16x8_t, b: uint16x4_t) -> uint32x4_t {
    static_assert_imm2!(LANE);
    vmull_high_u16(a, simd_shuffle8!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_high_laneq_u16<const LANE: i32>(a: uint16x8_t, b: uint16x8_t) -> uint32x4_t {
    static_assert_imm3!(LANE);
    vmull_high_u16(a, simd_shuffle8!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_high_lane_u32<const LANE: i32>(a: uint32x4_t, b: uint32x2_t) -> uint64x2_t {
    static_assert_imm1!(LANE);
    vmull_high_u32(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_high_laneq_u32<const LANE: i32>(a: uint32x4_t, b: uint32x4_t) -> uint64x2_t {
    static_assert_imm2!(LANE);
    vmull_high_u32(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx))]
pub unsafe fn vmulx_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmulx.v2f32")]
        fn vmulx_f32_(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
    vmulx_f32_(a, b)
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx))]
pub unsafe fn vmulxq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmulx.v4f32")]
        fn vmulxq_f32_(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
    vmulxq_f32_(a, b)
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx))]
pub unsafe fn vmulx_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmulx.v1f64")]
        fn vmulx_f64_(a: float64x1_t, b: float64x1_t) -> float64x1_t;
    }
    vmulx_f64_(a, b)
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx))]
pub unsafe fn vmulxq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmulx.v2f64")]
        fn vmulxq_f64_(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    vmulxq_f64_(a, b)
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulx_lane_f64<const LANE: i32>(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    static_assert!(LANE : i32 where LANE == 0);
    vmulx_f64(a, transmute::<f64, _>(simd_extract(b, LANE as u32)))
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulx_laneq_f64<const LANE: i32>(a: float64x1_t, b: float64x2_t) -> float64x1_t {
    static_assert_imm1!(LANE);
    vmulx_f64(a, transmute::<f64, _>(simd_extract(b, LANE as u32)))
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulx_lane_f32<const LANE: i32>(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    static_assert_imm1!(LANE);
    vmulx_f32(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulx_laneq_f32<const LANE: i32>(a: float32x2_t, b: float32x4_t) -> float32x2_t {
    static_assert_imm2!(LANE);
    vmulx_f32(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulxq_lane_f32<const LANE: i32>(a: float32x4_t, b: float32x2_t) -> float32x4_t {
    static_assert_imm1!(LANE);
    vmulxq_f32(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulxq_laneq_f32<const LANE: i32>(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    static_assert_imm2!(LANE);
    vmulxq_f32(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulxq_lane_f64<const LANE: i32>(a: float64x2_t, b: float64x1_t) -> float64x2_t {
    static_assert!(LANE : i32 where LANE == 0);
    vmulxq_f64(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulxq_laneq_f64<const LANE: i32>(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    static_assert_imm1!(LANE);
    vmulxq_f64(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx))]
pub unsafe fn vmulxs_f32(a: f32, b: f32) -> f32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmulx.f32")]
        fn vmulxs_f32_(a: f32, b: f32) -> f32;
    }
    vmulxs_f32_(a, b)
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx))]
pub unsafe fn vmulxd_f64(a: f64, b: f64) -> f64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmulx.f64")]
        fn vmulxd_f64_(a: f64, b: f64) -> f64;
    }
    vmulxd_f64_(a, b)
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulxs_lane_f32<const LANE: i32>(a: f32, b: float32x2_t) -> f32 {
    static_assert_imm1!(LANE);
    vmulxs_f32(a, simd_extract(b, LANE as u32))
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulxs_laneq_f32<const LANE: i32>(a: f32, b: float32x4_t) -> f32 {
    static_assert_imm2!(LANE);
    vmulxs_f32(a, simd_extract(b, LANE as u32))
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulxd_lane_f64<const LANE: i32>(a: f64, b: float64x1_t) -> f64 {
    static_assert!(LANE : i32 where LANE == 0);
    vmulxd_f64(a, simd_extract(b, LANE as u32))
}

/// Floating-point multiply extended
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulxd_laneq_f64<const LANE: i32>(a: f64, b: float64x2_t) -> f64 {
    static_assert_imm1!(LANE);
    vmulxd_f64(a, simd_extract(b, LANE as u32))
}

/// Floating-point fused Multiply-Add to accumulator(vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmadd))]
pub unsafe fn vfma_f64(a: float64x1_t, b: float64x1_t, c: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.fma.v1f64")]
        fn vfma_f64_(a: float64x1_t, b: float64x1_t, c: float64x1_t) -> float64x1_t;
    }
    vfma_f64_(b, c, a)
}

/// Floating-point fused Multiply-Add to accumulator(vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla))]
pub unsafe fn vfmaq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.fma.v2f64")]
        fn vfmaq_f64_(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t;
    }
    vfmaq_f64_(b, c, a)
}

/// Floating-point fused Multiply-Add to accumulator(vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmadd))]
pub unsafe fn vfma_n_f64(a: float64x1_t, b: float64x1_t, c: f64) -> float64x1_t {
    vfma_f64(a, b, vdup_n_f64(c))
}

/// Floating-point fused Multiply-Add to accumulator(vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla))]
pub unsafe fn vfmaq_n_f64(a: float64x2_t, b: float64x2_t, c: f64) -> float64x2_t {
    vfmaq_f64(a, b, vdupq_n_f64(c))
}

/// Floating-point fused multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfma_lane_f32<const LANE: i32>(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t {
    static_assert_imm1!(LANE);
    vfma_f32(a, b, vdup_n_f32(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfma_laneq_f32<const LANE: i32>(a: float32x2_t, b: float32x2_t, c: float32x4_t) -> float32x2_t {
    static_assert_imm2!(LANE);
    vfma_f32(a, b, vdup_n_f32(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmaq_lane_f32<const LANE: i32>(a: float32x4_t, b: float32x4_t, c: float32x2_t) -> float32x4_t {
    static_assert_imm1!(LANE);
    vfmaq_f32(a, b, vdupq_n_f32(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmaq_laneq_f32<const LANE: i32>(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    static_assert_imm2!(LANE);
    vfmaq_f32(a, b, vdupq_n_f32(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmadd, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfma_lane_f64<const LANE: i32>(a: float64x1_t, b: float64x1_t, c: float64x1_t) -> float64x1_t {
    static_assert!(LANE : i32 where LANE == 0);
    vfma_f64(a, b, vdup_n_f64(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfma_laneq_f64<const LANE: i32>(a: float64x1_t, b: float64x1_t, c: float64x2_t) -> float64x1_t {
    static_assert_imm1!(LANE);
    vfma_f64(a, b, vdup_n_f64(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmaq_lane_f64<const LANE: i32>(a: float64x2_t, b: float64x2_t, c: float64x1_t) -> float64x2_t {
    static_assert!(LANE : i32 where LANE == 0);
    vfmaq_f64(a, b, vdupq_n_f64(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmaq_laneq_f64<const LANE: i32>(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    static_assert_imm1!(LANE);
    vfmaq_f64(a, b, vdupq_n_f64(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmas_lane_f32<const LANE: i32>(a: f32, b: f32, c: float32x2_t) -> f32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.fma.f32")]
        fn vfmas_lane_f32_(a: f32, b: f32, c: f32) -> f32;
    }
    static_assert_imm1!(LANE);
    let c: f32 = simd_extract(c, LANE as u32);
    vfmas_lane_f32_(b, c, a)
}

/// Floating-point fused multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmas_laneq_f32<const LANE: i32>(a: f32, b: f32, c: float32x4_t) -> f32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.fma.f32")]
        fn vfmas_laneq_f32_(a: f32, b: f32, c: f32) -> f32;
    }
    static_assert_imm2!(LANE);
    let c: f32 = simd_extract(c, LANE as u32);
    vfmas_laneq_f32_(b, c, a)
}

/// Floating-point fused multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmadd, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmad_lane_f64<const LANE: i32>(a: f64, b: f64, c: float64x1_t) -> f64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.fma.f64")]
        fn vfmad_lane_f64_(a: f64, b: f64, c: f64) -> f64;
    }
    static_assert!(LANE : i32 where LANE == 0);
    let c: f64 = simd_extract(c, LANE as u32);
    vfmad_lane_f64_(b, c, a)
}

/// Floating-point fused multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmad_laneq_f64<const LANE: i32>(a: f64, b: f64, c: float64x2_t) -> f64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.fma.f64")]
        fn vfmad_laneq_f64_(a: f64, b: f64, c: f64) -> f64;
    }
    static_assert_imm1!(LANE);
    let c: f64 = simd_extract(c, LANE as u32);
    vfmad_laneq_f64_(b, c, a)
}

/// Floating-point fused multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmsub))]
pub unsafe fn vfms_f64(a: float64x1_t, b: float64x1_t, c: float64x1_t) -> float64x1_t {
    let b: float64x1_t = simd_neg(b);
    vfma_f64(a, b, c)
}

/// Floating-point fused multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls))]
pub unsafe fn vfmsq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    let b: float64x2_t = simd_neg(b);
    vfmaq_f64(a, b, c)
}

/// Floating-point fused Multiply-subtract to accumulator(vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmsub))]
pub unsafe fn vfms_n_f64(a: float64x1_t, b: float64x1_t, c: f64) -> float64x1_t {
    vfms_f64(a, b, vdup_n_f64(c))
}

/// Floating-point fused Multiply-subtract to accumulator(vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls))]
pub unsafe fn vfmsq_n_f64(a: float64x2_t, b: float64x2_t, c: f64) -> float64x2_t {
    vfmsq_f64(a, b, vdupq_n_f64(c))
}

/// Floating-point fused multiply-subtract to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfms_lane_f32<const LANE: i32>(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t {
    static_assert_imm1!(LANE);
    vfms_f32(a, b, vdup_n_f32(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-subtract to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfms_laneq_f32<const LANE: i32>(a: float32x2_t, b: float32x2_t, c: float32x4_t) -> float32x2_t {
    static_assert_imm2!(LANE);
    vfms_f32(a, b, vdup_n_f32(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-subtract to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmsq_lane_f32<const LANE: i32>(a: float32x4_t, b: float32x4_t, c: float32x2_t) -> float32x4_t {
    static_assert_imm1!(LANE);
    vfmsq_f32(a, b, vdupq_n_f32(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-subtract to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmsq_laneq_f32<const LANE: i32>(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    static_assert_imm2!(LANE);
    vfmsq_f32(a, b, vdupq_n_f32(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-subtract to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmsub, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfms_lane_f64<const LANE: i32>(a: float64x1_t, b: float64x1_t, c: float64x1_t) -> float64x1_t {
    static_assert!(LANE : i32 where LANE == 0);
    vfms_f64(a, b, vdup_n_f64(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-subtract to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfms_laneq_f64<const LANE: i32>(a: float64x1_t, b: float64x1_t, c: float64x2_t) -> float64x1_t {
    static_assert_imm1!(LANE);
    vfms_f64(a, b, vdup_n_f64(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-subtract to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmsq_lane_f64<const LANE: i32>(a: float64x2_t, b: float64x2_t, c: float64x1_t) -> float64x2_t {
    static_assert!(LANE : i32 where LANE == 0);
    vfmsq_f64(a, b, vdupq_n_f64(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-subtract to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmsq_laneq_f64<const LANE: i32>(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    static_assert_imm1!(LANE);
    vfmsq_f64(a, b, vdupq_n_f64(simd_extract(c, LANE as u32)))
}

/// Floating-point fused multiply-subtract to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmss_lane_f32<const LANE: i32>(a: f32, b: f32, c: float32x2_t) -> f32 {
    vfmas_lane_f32::<LANE>(a, -b, c)
}

/// Floating-point fused multiply-subtract to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmss_laneq_f32<const LANE: i32>(a: f32, b: f32, c: float32x4_t) -> f32 {
    vfmas_laneq_f32::<LANE>(a, -b, c)
}

/// Floating-point fused multiply-subtract to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmsub, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmsd_lane_f64<const LANE: i32>(a: f64, b: f64, c: float64x1_t) -> f64 {
    vfmad_lane_f64::<LANE>(a, -b, c)
}

/// Floating-point fused multiply-subtract to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vfmsd_laneq_f64<const LANE: i32>(a: f64, b: f64, c: float64x2_t) -> f64 {
    vfmad_laneq_f64::<LANE>(a, -b, c)
}

/// Divide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fdiv))]
pub unsafe fn vdiv_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_div(a, b)
}

/// Divide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fdiv))]
pub unsafe fn vdivq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_div(a, b)
}

/// Divide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fdiv))]
pub unsafe fn vdiv_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    simd_div(a, b)
}

/// Divide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fdiv))]
pub unsafe fn vdivq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_div(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fsub))]
pub unsafe fn vsub_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fsub))]
pub unsafe fn vsubq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_sub(a, b)
}

/// Signed Add Long across Vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(saddlv))]
pub unsafe fn vaddlv_s16(a: int16x4_t) -> i32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.saddlv.i32.v4i16")]
        fn vaddlv_s16_(a: int16x4_t) -> i32;
    }
    vaddlv_s16_(a)
}

/// Signed Add Long across Vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(saddlv))]
pub unsafe fn vaddlvq_s16(a: int16x8_t) -> i32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.saddlv.i32.v8i16")]
        fn vaddlvq_s16_(a: int16x8_t) -> i32;
    }
    vaddlvq_s16_(a)
}

/// Signed Add Long across Vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(saddlp))]
pub unsafe fn vaddlv_s32(a: int32x2_t) -> i64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.saddlv.i64.v2i32")]
        fn vaddlv_s32_(a: int32x2_t) -> i64;
    }
    vaddlv_s32_(a)
}

/// Signed Add Long across Vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(saddlv))]
pub unsafe fn vaddlvq_s32(a: int32x4_t) -> i64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.saddlv.i64.v4i32")]
        fn vaddlvq_s32_(a: int32x4_t) -> i64;
    }
    vaddlvq_s32_(a)
}

/// Unsigned Add Long across Vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uaddlv))]
pub unsafe fn vaddlv_u16(a: uint16x4_t) -> u32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uaddlv.i32.v4i16")]
        fn vaddlv_u16_(a: uint16x4_t) -> u32;
    }
    vaddlv_u16_(a)
}

/// Unsigned Add Long across Vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uaddlv))]
pub unsafe fn vaddlvq_u16(a: uint16x8_t) -> u32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uaddlv.i32.v8i16")]
        fn vaddlvq_u16_(a: uint16x8_t) -> u32;
    }
    vaddlvq_u16_(a)
}

/// Unsigned Add Long across Vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uaddlp))]
pub unsafe fn vaddlv_u32(a: uint32x2_t) -> u64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uaddlv.i64.v2i32")]
        fn vaddlv_u32_(a: uint32x2_t) -> u64;
    }
    vaddlv_u32_(a)
}

/// Unsigned Add Long across Vector
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uaddlv))]
pub unsafe fn vaddlvq_u32(a: uint32x4_t) -> u64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uaddlv.i64.v4i32")]
        fn vaddlvq_u32_(a: uint32x4_t) -> u64;
    }
    vaddlvq_u32_(a)
}

/// Signed Subtract Wide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ssubw))]
pub unsafe fn vsubw_high_s8(a: int16x8_t, b: int8x16_t) -> int16x8_t {
    let c: int8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    simd_sub(a, simd_cast(c))
}

/// Signed Subtract Wide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ssubw))]
pub unsafe fn vsubw_high_s16(a: int32x4_t, b: int16x8_t) -> int32x4_t {
    let c: int16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    simd_sub(a, simd_cast(c))
}

/// Signed Subtract Wide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ssubw))]
pub unsafe fn vsubw_high_s32(a: int64x2_t, b: int32x4_t) -> int64x2_t {
    let c: int32x2_t = simd_shuffle2!(b, b, [2, 3]);
    simd_sub(a, simd_cast(c))
}

/// Unsigned Subtract Wide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usubw))]
pub unsafe fn vsubw_high_u8(a: uint16x8_t, b: uint8x16_t) -> uint16x8_t {
    let c: uint8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    simd_sub(a, simd_cast(c))
}

/// Unsigned Subtract Wide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usubw))]
pub unsafe fn vsubw_high_u16(a: uint32x4_t, b: uint16x8_t) -> uint32x4_t {
    let c: uint16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    simd_sub(a, simd_cast(c))
}

/// Unsigned Subtract Wide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usubw))]
pub unsafe fn vsubw_high_u32(a: uint64x2_t, b: uint32x4_t) -> uint64x2_t {
    let c: uint32x2_t = simd_shuffle2!(b, b, [2, 3]);
    simd_sub(a, simd_cast(c))
}

/// Signed Subtract Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ssubl))]
pub unsafe fn vsubl_high_s8(a: int8x16_t, b: int8x16_t) -> int16x8_t {
    let c: int8x8_t = simd_shuffle8!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let d: int16x8_t = simd_cast(c);
    let e: int8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let f: int16x8_t = simd_cast(e);
    simd_sub(d, f)
}

/// Signed Subtract Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ssubl))]
pub unsafe fn vsubl_high_s16(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    let c: int16x4_t = simd_shuffle4!(a, a, [4, 5, 6, 7]);
    let d: int32x4_t = simd_cast(c);
    let e: int16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    let f: int32x4_t = simd_cast(e);
    simd_sub(d, f)
}

/// Signed Subtract Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ssubl))]
pub unsafe fn vsubl_high_s32(a: int32x4_t, b: int32x4_t) -> int64x2_t {
    let c: int32x2_t = simd_shuffle2!(a, a, [2, 3]);
    let d: int64x2_t = simd_cast(c);
    let e: int32x2_t = simd_shuffle2!(b, b, [2, 3]);
    let f: int64x2_t = simd_cast(e);
    simd_sub(d, f)
}

/// Unsigned Subtract Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usubl))]
pub unsafe fn vsubl_high_u8(a: uint8x16_t, b: uint8x16_t) -> uint16x8_t {
    let c: uint8x8_t = simd_shuffle8!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let d: uint16x8_t = simd_cast(c);
    let e: uint8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let f: uint16x8_t = simd_cast(e);
    simd_sub(d, f)
}

/// Unsigned Subtract Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usubl))]
pub unsafe fn vsubl_high_u16(a: uint16x8_t, b: uint16x8_t) -> uint32x4_t {
    let c: uint16x4_t = simd_shuffle4!(a, a, [4, 5, 6, 7]);
    let d: uint32x4_t = simd_cast(c);
    let e: uint16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    let f: uint32x4_t = simd_cast(e);
    simd_sub(d, f)
}

/// Unsigned Subtract Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usubl))]
pub unsafe fn vsubl_high_u32(a: uint32x4_t, b: uint32x4_t) -> uint64x2_t {
    let c: uint32x2_t = simd_shuffle2!(a, a, [2, 3]);
    let d: uint64x2_t = simd_cast(c);
    let e: uint32x2_t = simd_shuffle2!(b, b, [2, 3]);
    let f: uint64x2_t = simd_cast(e);
    simd_sub(d, f)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmax))]
pub unsafe fn vmax_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmax.v1f64")]
        fn vmax_f64_(a: float64x1_t, b: float64x1_t) -> float64x1_t;
    }
    vmax_f64_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmax))]
pub unsafe fn vmaxq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmax.v2f64")]
        fn vmaxq_f64_(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    vmaxq_f64_(a, b)
}

/// Floating-point Maximun Number (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxnm))]
pub unsafe fn vmaxnm_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmaxnm.v1f64")]
        fn vmaxnm_f64_(a: float64x1_t, b: float64x1_t) -> float64x1_t;
    }
    vmaxnm_f64_(a, b)
}

/// Floating-point Maximun Number (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxnm))]
pub unsafe fn vmaxnmq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmaxnm.v2f64")]
        fn vmaxnmq_f64_(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    vmaxnmq_f64_(a, b)
}

/// Floating-point Maximum Number Pairwise (vector).
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxnmp))]
pub unsafe fn vpmaxnm_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmaxnmp.v2f32")]
        fn vpmaxnm_f32_(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
    vpmaxnm_f32_(a, b)
}

/// Floating-point Maximum Number Pairwise (vector).
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxnmp))]
pub unsafe fn vpmaxnmq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmaxnmp.v2f64")]
        fn vpmaxnmq_f64_(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    vpmaxnmq_f64_(a, b)
}

/// Floating-point Maximum Number Pairwise (vector).
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxnmp))]
pub unsafe fn vpmaxnmq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmaxnmp.v4f32")]
        fn vpmaxnmq_f32_(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
    vpmaxnmq_f32_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmin))]
pub unsafe fn vmin_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmin.v1f64")]
        fn vmin_f64_(a: float64x1_t, b: float64x1_t) -> float64x1_t;
    }
    vmin_f64_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmin))]
pub unsafe fn vminq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmin.v2f64")]
        fn vminq_f64_(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    vminq_f64_(a, b)
}

/// Floating-point Minimun Number (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminnm))]
pub unsafe fn vminnm_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fminnm.v1f64")]
        fn vminnm_f64_(a: float64x1_t, b: float64x1_t) -> float64x1_t;
    }
    vminnm_f64_(a, b)
}

/// Floating-point Minimun Number (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminnm))]
pub unsafe fn vminnmq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fminnm.v2f64")]
        fn vminnmq_f64_(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    vminnmq_f64_(a, b)
}

/// Floating-point Minimum Number Pairwise (vector).
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminnmp))]
pub unsafe fn vpminnm_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fminnmp.v2f32")]
        fn vpminnm_f32_(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
    vpminnm_f32_(a, b)
}

/// Floating-point Minimum Number Pairwise (vector).
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminnmp))]
pub unsafe fn vpminnmq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fminnmp.v2f64")]
        fn vpminnmq_f64_(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    vpminnmq_f64_(a, b)
}

/// Floating-point Minimum Number Pairwise (vector).
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminnmp))]
pub unsafe fn vpminnmq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fminnmp.v4f32")]
        fn vpminnmq_f32_(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
    vpminnmq_f32_(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull))]
pub unsafe fn vqdmullh_s16(a: i16, b: i16) -> i32 {
    let a: int16x4_t = vdup_n_s16(a);
    let b: int16x4_t = vdup_n_s16(b);
    simd_extract(vqdmull_s16(a, b), 0)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull))]
pub unsafe fn vqdmulls_s32(a: i32, b: i32) -> i64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqdmulls.scalar")]
        fn vqdmulls_s32_(a: i32, b: i32) -> i64;
    }
    vqdmulls_s32_(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2))]
pub unsafe fn vqdmull_high_s16(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    let a: int16x4_t = simd_shuffle4!(a, a, [4, 5, 6, 7]);
    let b: int16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    vqdmull_s16(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2))]
pub unsafe fn vqdmull_high_s32(a: int32x4_t, b: int32x4_t) -> int64x2_t {
    let a: int32x2_t = simd_shuffle2!(a, a, [2, 3]);
    let b: int32x2_t = simd_shuffle2!(b, b, [2, 3]);
    vqdmull_s32(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2))]
pub unsafe fn vqdmull_high_n_s16(a: int16x8_t, b: i16) -> int32x4_t {
    let a: int16x4_t = simd_shuffle4!(a, a, [4, 5, 6, 7]);
    let b: int16x4_t = vdup_n_s16(b);
    vqdmull_s16(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2))]
pub unsafe fn vqdmull_high_n_s32(a: int32x4_t, b: i32) -> int64x2_t {
    let a: int32x2_t = simd_shuffle2!(a, a, [2, 3]);
    let b: int32x2_t = vdup_n_s32(b);
    vqdmull_s32(a, b)
}

/// Vector saturating doubling long multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull, N = 4))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmull_laneq_s16<const N: i32>(a: int16x4_t, b: int16x8_t) -> int32x4_t {
    static_assert_imm3!(N);
    let b: int16x4_t = simd_shuffle4!(b, b, <const N: i32> [N as u32, N as u32, N as u32, N as u32]);
    vqdmull_s16(a, b)
}

/// Vector saturating doubling long multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmull_laneq_s32<const N: i32>(a: int32x2_t, b: int32x4_t) -> int64x2_t {
    static_assert_imm2!(N);
    let b: int32x2_t = simd_shuffle2!(b, b, <const N: i32> [N as u32, N as u32]);
    vqdmull_s32(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmullh_lane_s16<const N: i32>(a: i16, b: int16x4_t) -> i32 {
    static_assert_imm2!(N);
    let b: i16 = simd_extract(b, N as u32);
    vqdmullh_s16(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull, N = 4))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmullh_laneq_s16<const N: i32>(a: i16, b: int16x8_t) -> i32 {
    static_assert_imm3!(N);
    let b: i16 = simd_extract(b, N as u32);
    vqdmullh_s16(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmulls_lane_s32<const N: i32>(a: i32, b: int32x2_t) -> i64 {
    static_assert_imm1!(N);
    let b: i32 = simd_extract(b, N as u32);
    vqdmulls_s32(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmulls_laneq_s32<const N: i32>(a: i32, b: int32x4_t) -> i64 {
    static_assert_imm2!(N);
    let b: i32 = simd_extract(b, N as u32);
    vqdmulls_s32(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmull_high_lane_s16<const N: i32>(a: int16x8_t, b: int16x4_t) -> int32x4_t {
    static_assert_imm2!(N);
    let a: int16x4_t = simd_shuffle4!(a, a, [4, 5, 6, 7]);
    let b: int16x4_t = simd_shuffle4!(b, b, <const N: i32> [N as u32, N as u32, N as u32, N as u32]);
    vqdmull_s16(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmull_high_lane_s32<const N: i32>(a: int32x4_t, b: int32x2_t) -> int64x2_t {
    static_assert_imm1!(N);
    let a: int32x2_t = simd_shuffle2!(a, a, [2, 3]);
    let b: int32x2_t = simd_shuffle2!(b, b, <const N: i32> [N as u32, N as u32]);
    vqdmull_s32(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2, N = 4))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmull_high_laneq_s16<const N: i32>(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    static_assert_imm3!(N);
    let a: int16x4_t = simd_shuffle4!(a, a, [4, 5, 6, 7]);
    let b: int16x4_t = simd_shuffle4!(b, b, <const N: i32> [N as u32, N as u32, N as u32, N as u32]);
    vqdmull_s16(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmull_high_laneq_s32<const N: i32>(a: int32x4_t, b: int32x4_t) -> int64x2_t {
    static_assert_imm2!(N);
    let a: int32x2_t = simd_shuffle2!(a, a, [2, 3]);
    let b: int32x2_t = simd_shuffle2!(b, b, <const N: i32> [N as u32, N as u32]);
    vqdmull_s32(a, b)
}

/// Signed saturating doubling multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2))]
pub unsafe fn vqdmlal_high_s16(a: int32x4_t, b: int16x8_t, c: int16x8_t) -> int32x4_t {
    vqaddq_s32(a, vqdmull_high_s16(b, c))
}

/// Signed saturating doubling multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2))]
pub unsafe fn vqdmlal_high_s32(a: int64x2_t, b: int32x4_t, c: int32x4_t) -> int64x2_t {
    vqaddq_s64(a, vqdmull_high_s32(b, c))
}

/// Signed saturating doubling multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2))]
pub unsafe fn vqdmlal_high_n_s16(a: int32x4_t, b: int16x8_t, c: i16) -> int32x4_t {
    vqaddq_s32(a, vqdmull_high_n_s16(b, c))
}

/// Signed saturating doubling multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2))]
pub unsafe fn vqdmlal_high_n_s32(a: int64x2_t, b: int32x4_t, c: i32) -> int64x2_t {
    vqaddq_s64(a, vqdmull_high_n_s32(b, c))
}

/// Vector widening saturating doubling multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal, N = 2))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlal_laneq_s16<const N: i32>(a: int32x4_t, b: int16x4_t, c: int16x8_t) -> int32x4_t {
    static_assert_imm3!(N);
    vqaddq_s32(a, vqdmull_laneq_s16::<N>(b, c))
}

/// Vector widening saturating doubling multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal, N = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlal_laneq_s32<const N: i32>(a: int64x2_t, b: int32x2_t, c: int32x4_t) -> int64x2_t {
    static_assert_imm2!(N);
    vqaddq_s64(a, vqdmull_laneq_s32::<N>(b, c))
}

/// Signed saturating doubling multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2, N = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlal_high_lane_s16<const N: i32>(a: int32x4_t, b: int16x8_t, c: int16x4_t) -> int32x4_t {
    static_assert_imm2!(N);
    vqaddq_s32(a, vqdmull_high_lane_s16::<N>(b, c))
}

/// Signed saturating doubling multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2, N = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlal_high_laneq_s16<const N: i32>(a: int32x4_t, b: int16x8_t, c: int16x8_t) -> int32x4_t {
    static_assert_imm3!(N);
    vqaddq_s32(a, vqdmull_high_laneq_s16::<N>(b, c))
}

/// Signed saturating doubling multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2, N = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlal_high_lane_s32<const N: i32>(a: int64x2_t, b: int32x4_t, c: int32x2_t) -> int64x2_t {
    static_assert_imm1!(N);
    vqaddq_s64(a, vqdmull_high_lane_s32::<N>(b, c))
}

/// Signed saturating doubling multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2, N = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlal_high_laneq_s32<const N: i32>(a: int64x2_t, b: int32x4_t, c: int32x4_t) -> int64x2_t {
    static_assert_imm2!(N);
    vqaddq_s64(a, vqdmull_high_laneq_s32::<N>(b, c))
}

/// Signed saturating doubling multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2))]
pub unsafe fn vqdmlsl_high_s16(a: int32x4_t, b: int16x8_t, c: int16x8_t) -> int32x4_t {
    vqsubq_s32(a, vqdmull_high_s16(b, c))
}

/// Signed saturating doubling multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2))]
pub unsafe fn vqdmlsl_high_s32(a: int64x2_t, b: int32x4_t, c: int32x4_t) -> int64x2_t {
    vqsubq_s64(a, vqdmull_high_s32(b, c))
}

/// Signed saturating doubling multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2))]
pub unsafe fn vqdmlsl_high_n_s16(a: int32x4_t, b: int16x8_t, c: i16) -> int32x4_t {
    vqsubq_s32(a, vqdmull_high_n_s16(b, c))
}

/// Signed saturating doubling multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2))]
pub unsafe fn vqdmlsl_high_n_s32(a: int64x2_t, b: int32x4_t, c: i32) -> int64x2_t {
    vqsubq_s64(a, vqdmull_high_n_s32(b, c))
}

/// Vector widening saturating doubling multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl, N = 2))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlsl_laneq_s16<const N: i32>(a: int32x4_t, b: int16x4_t, c: int16x8_t) -> int32x4_t {
    static_assert_imm3!(N);
    vqsubq_s32(a, vqdmull_laneq_s16::<N>(b, c))
}

/// Vector widening saturating doubling multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl, N = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlsl_laneq_s32<const N: i32>(a: int64x2_t, b: int32x2_t, c: int32x4_t) -> int64x2_t {
    static_assert_imm2!(N);
    vqsubq_s64(a, vqdmull_laneq_s32::<N>(b, c))
}

/// Signed saturating doubling multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2, N = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlsl_high_lane_s16<const N: i32>(a: int32x4_t, b: int16x8_t, c: int16x4_t) -> int32x4_t {
    static_assert_imm2!(N);
    vqsubq_s32(a, vqdmull_high_lane_s16::<N>(b, c))
}

/// Signed saturating doubling multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2, N = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlsl_high_laneq_s16<const N: i32>(a: int32x4_t, b: int16x8_t, c: int16x8_t) -> int32x4_t {
    static_assert_imm3!(N);
    vqsubq_s32(a, vqdmull_high_laneq_s16::<N>(b, c))
}

/// Signed saturating doubling multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2, N = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlsl_high_lane_s32<const N: i32>(a: int64x2_t, b: int32x4_t, c: int32x2_t) -> int64x2_t {
    static_assert_imm1!(N);
    vqsubq_s64(a, vqdmull_high_lane_s32::<N>(b, c))
}

/// Signed saturating doubling multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2, N = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlsl_high_laneq_s32<const N: i32>(a: int64x2_t, b: int32x4_t, c: int32x4_t) -> int64x2_t {
    static_assert_imm2!(N);
    vqsubq_s64(a, vqdmull_high_laneq_s32::<N>(b, c))
}

/// Signed saturating doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh))]
pub unsafe fn vqdmulhh_s16(a: i16, b: i16) -> i16 {
    let a: int16x4_t = vdup_n_s16(a);
    let b: int16x4_t = vdup_n_s16(b);
    simd_extract(vqdmulh_s16(a, b), 0)
}

/// Signed saturating doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh))]
pub unsafe fn vqdmulhs_s32(a: i32, b: i32) -> i32 {
    let a: int32x2_t = vdup_n_s32(a);
    let b: int32x2_t = vdup_n_s32(b);
    simd_extract(vqdmulh_s32(a, b), 0)
}

/// Signed saturating doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmulhh_lane_s16<const N: i32>(a: i16, b: int16x4_t) -> i16 {
    static_assert_imm2!(N);
    let b: i16 = simd_extract(b, N as u32);
    vqdmulhh_s16(a, b)
}

/// Signed saturating doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmulhh_laneq_s16<const N: i32>(a: i16, b: int16x8_t) -> i16 {
    static_assert_imm3!(N);
    let b: i16 = simd_extract(b, N as u32);
    vqdmulhh_s16(a, b)
}

/// Signed saturating doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmulhs_lane_s32<const N: i32>(a: i32, b: int32x2_t) -> i32 {
    static_assert_imm1!(N);
    let b: i32 = simd_extract(b, N as u32);
    vqdmulhs_s32(a, b)
}

/// Signed saturating doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmulhs_laneq_s32<const N: i32>(a: i32, b: int32x4_t) -> i32 {
    static_assert_imm2!(N);
    let b: i32 = simd_extract(b, N as u32);
    vqdmulhs_s32(a, b)
}

/// Saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtn))]
pub unsafe fn vqmovnh_s16(a: i16) -> i8 {
    simd_extract(vqmovn_s16(vdupq_n_s16(a)), 0)
}

/// Saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtn))]
pub unsafe fn vqmovns_s32(a: i32) -> i16 {
    simd_extract(vqmovn_s32(vdupq_n_s32(a)), 0)
}

/// Saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqxtn))]
pub unsafe fn vqmovnh_u16(a: u16) -> u8 {
    simd_extract(vqmovn_u16(vdupq_n_u16(a)), 0)
}

/// Saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqxtn))]
pub unsafe fn vqmovns_u32(a: u32) -> u16 {
    simd_extract(vqmovn_u32(vdupq_n_u32(a)), 0)
}

/// Saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtn))]
pub unsafe fn vqmovnd_s64(a: i64) -> i32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.scalar.sqxtn.i32.i64")]
        fn vqmovnd_s64_(a: i64) -> i32;
    }
    vqmovnd_s64_(a)
}

/// Saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqxtn))]
pub unsafe fn vqmovnd_u64(a: u64) -> u32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.scalar.uqxtn.i32.i64")]
        fn vqmovnd_u64_(a: u64) -> u32;
    }
    vqmovnd_u64_(a)
}

/// Signed saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtn2))]
pub unsafe fn vqmovn_high_s16(a: int8x8_t, b: int16x8_t) -> int8x16_t {
    simd_shuffle16!(a, vqmovn_s16(b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Signed saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtn2))]
pub unsafe fn vqmovn_high_s32(a: int16x4_t, b: int32x4_t) -> int16x8_t {
    simd_shuffle8!(a, vqmovn_s32(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Signed saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtn2))]
pub unsafe fn vqmovn_high_s64(a: int32x2_t, b: int64x2_t) -> int32x4_t {
    simd_shuffle4!(a, vqmovn_s64(b), [0, 1, 2, 3])
}

/// Signed saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqxtn2))]
pub unsafe fn vqmovn_high_u16(a: uint8x8_t, b: uint16x8_t) -> uint8x16_t {
    simd_shuffle16!(a, vqmovn_u16(b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Signed saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqxtn2))]
pub unsafe fn vqmovn_high_u32(a: uint16x4_t, b: uint32x4_t) -> uint16x8_t {
    simd_shuffle8!(a, vqmovn_u32(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Signed saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqxtn2))]
pub unsafe fn vqmovn_high_u64(a: uint32x2_t, b: uint64x2_t) -> uint32x4_t {
    simd_shuffle4!(a, vqmovn_u64(b), [0, 1, 2, 3])
}

/// Signed saturating extract unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtun))]
pub unsafe fn vqmovunh_s16(a: i16) -> u8 {
    simd_extract(vqmovun_s16(vdupq_n_s16(a)), 0)
}

/// Signed saturating extract unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtun))]
pub unsafe fn vqmovuns_s32(a: i32) -> u16 {
    simd_extract(vqmovun_s32(vdupq_n_s32(a)), 0)
}

/// Signed saturating extract unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtun))]
pub unsafe fn vqmovund_s64(a: i64) -> u32 {
    simd_extract(vqmovun_s64(vdupq_n_s64(a)), 0)
}

/// Signed saturating extract unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtun2))]
pub unsafe fn vqmovun_high_s16(a: uint8x8_t, b: int16x8_t) -> uint8x16_t {
    simd_shuffle16!(a, vqmovun_s16(b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Signed saturating extract unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtun2))]
pub unsafe fn vqmovun_high_s32(a: uint16x4_t, b: int32x4_t) -> uint16x8_t {
    simd_shuffle8!(a, vqmovun_s32(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Signed saturating extract unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtun2))]
pub unsafe fn vqmovun_high_s64(a: uint32x2_t, b: int64x2_t) -> uint32x4_t {
    simd_shuffle4!(a, vqmovun_s64(b), [0, 1, 2, 3])
}

/// Signed saturating rounding doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh))]
pub unsafe fn vqrdmulhh_s16(a: i16, b: i16) -> i16 {
    simd_extract(vqrdmulh_s16(vdup_n_s16(a), vdup_n_s16(b)), 0)
}

/// Signed saturating rounding doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh))]
pub unsafe fn vqrdmulhs_s32(a: i32, b: i32) -> i32 {
    simd_extract(vqrdmulh_s32(vdup_n_s32(a), vdup_n_s32(b)), 0)
}

/// Signed saturating rounding doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrdmulhh_lane_s16<const LANE: i32>(a: i16, b: int16x4_t) -> i16 {
    static_assert_imm2!(LANE);
    vqrdmulhh_s16(a, simd_extract(b, LANE as u32))
}

/// Signed saturating rounding doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrdmulhh_laneq_s16<const LANE: i32>(a: i16, b: int16x8_t) -> i16 {
    static_assert_imm3!(LANE);
    vqrdmulhh_s16(a, simd_extract(b, LANE as u32))
}

/// Signed saturating rounding doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrdmulhs_lane_s32<const LANE: i32>(a: i32, b: int32x2_t) -> i32 {
    static_assert_imm1!(LANE);
    vqrdmulhs_s32(a, simd_extract(b, LANE as u32))
}

/// Signed saturating rounding doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrdmulhs_laneq_s32<const LANE: i32>(a: i32, b: int32x4_t) -> i32 {
    static_assert_imm2!(LANE);
    vqrdmulhs_s32(a, simd_extract(b, LANE as u32))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh))]
pub unsafe fn vqrdmlahh_s16(a: i16, b: i16, c: i16) -> i16 {
    vqaddh_s16(a, vqrdmulhh_s16(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh))]
pub unsafe fn vqrdmlahs_s32(a: i32, b: i32, c: i32) -> i32 {
    vqadds_s32(a, vqrdmulhs_s32(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlahh_lane_s16<const LANE: i32>(a: i16, b: i16, c: int16x4_t) -> i16 {
    static_assert_imm2!(LANE);
    vqaddh_s16(a, vqrdmulhh_lane_s16::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlahh_laneq_s16<const LANE: i32>(a: i16, b: i16, c: int16x8_t) -> i16 {
    static_assert_imm3!(LANE);
    vqaddh_s16(a, vqrdmulhh_laneq_s16::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlahs_lane_s32<const LANE: i32>(a: i32, b: i32, c: int32x2_t) -> i32 {
    static_assert_imm1!(LANE);
    vqadds_s32(a, vqrdmulhs_lane_s32::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlahs_laneq_s32<const LANE: i32>(a: i32, b: i32, c: int32x4_t) -> i32 {
    static_assert_imm2!(LANE);
    vqadds_s32(a, vqrdmulhs_laneq_s32::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh))]
pub unsafe fn vqrdmlshh_s16(a: i16, b: i16, c: i16) -> i16 {
    vqsubh_s16(a, vqrdmulhh_s16(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh))]
pub unsafe fn vqrdmlshs_s32(a: i32, b: i32, c: i32) -> i32 {
    vqsubs_s32(a, vqrdmulhs_s32(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlshh_lane_s16<const LANE: i32>(a: i16, b: i16, c: int16x4_t) -> i16 {
    static_assert_imm2!(LANE);
    vqsubh_s16(a, vqrdmulhh_lane_s16::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlshh_laneq_s16<const LANE: i32>(a: i16, b: i16, c: int16x8_t) -> i16 {
    static_assert_imm3!(LANE);
    vqsubh_s16(a, vqrdmulhh_laneq_s16::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlshs_lane_s32<const LANE: i32>(a: i32, b: i32, c: int32x2_t) -> i32 {
    static_assert_imm1!(LANE);
    vqsubs_s32(a, vqrdmulhs_lane_s32::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlshs_laneq_s32<const LANE: i32>(a: i32, b: i32, c: int32x4_t) -> i32 {
    static_assert_imm2!(LANE);
    vqsubs_s32(a, vqrdmulhs_laneq_s32::<LANE>(b, c))
}

/// Signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshl))]
pub unsafe fn vqrshls_s32(a: i32, b: i32) -> i32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshl.i32")]
        fn vqrshls_s32_(a: i32, b: i32) -> i32;
    }
    vqrshls_s32_(a, b)
}

/// Signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshl))]
pub unsafe fn vqrshld_s64(a: i64, b: i64) -> i64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshl.i64")]
        fn vqrshld_s64_(a: i64, b: i64) -> i64;
    }
    vqrshld_s64_(a, b)
}

/// Signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshl))]
pub unsafe fn vqrshlb_s8(a: i8, b: i8) -> i8 {
    let a: int8x8_t = vdup_n_s8(a);
    let b: int8x8_t = vdup_n_s8(b);
    simd_extract(vqrshl_s8(a, b), 0)
}

/// Signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshl))]
pub unsafe fn vqrshlh_s16(a: i16, b: i16) -> i16 {
    let a: int16x4_t = vdup_n_s16(a);
    let b: int16x4_t = vdup_n_s16(b);
    simd_extract(vqrshl_s16(a, b), 0)
}

/// Unsigned signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshl))]
pub unsafe fn vqrshls_u32(a: u32, b: i32) -> u32 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqrshl.i32")]
        fn vqrshls_u32_(a: u32, b: i32) -> u32;
    }
    vqrshls_u32_(a, b)
}

/// Unsigned signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshl))]
pub unsafe fn vqrshld_u64(a: u64, b: i64) -> u64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqrshl.i64")]
        fn vqrshld_u64_(a: u64, b: i64) -> u64;
    }
    vqrshld_u64_(a, b)
}

/// Unsigned signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshl))]
pub unsafe fn vqrshlb_u8(a: u8, b: i8) -> u8 {
    let a: uint8x8_t = vdup_n_u8(a);
    let b: int8x8_t = vdup_n_s8(b);
    simd_extract(vqrshl_u8(a, b), 0)
}

/// Unsigned signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshl))]
pub unsafe fn vqrshlh_u16(a: u16, b: i16) -> u16 {
    let a: uint16x4_t = vdup_n_u16(a);
    let b: int16x4_t = vdup_n_s16(b);
    simd_extract(vqrshl_u16(a, b), 0)
}

/// Signed saturating rounded shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrnh_n_s16<const N: i32>(a: i16) -> i8 {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    let a: int16x8_t = vdupq_n_s16(a);
    simd_extract(vqrshrn_n_s16::<N>(a), 0)
}

/// Signed saturating rounded shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrns_n_s32<const N: i32>(a: i32) -> i16 {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    let a: int32x4_t = vdupq_n_s32(a);
    simd_extract(vqrshrn_n_s32::<N>(a), 0)
}

/// Signed saturating rounded shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrnd_n_s64<const N: i32>(a: i64) -> i32 {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    let a: int64x2_t = vdupq_n_s64(a);
    simd_extract(vqrshrn_n_s64::<N>(a), 0)
}

/// Signed saturating rounded shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrshrn_high_n_s16<const N: i32>(a: int8x8_t, b: int16x8_t) -> int8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_shuffle16!(a, vqrshrn_n_s16::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Signed saturating rounded shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrshrn_high_n_s32<const N: i32>(a: int16x4_t, b: int32x4_t) -> int16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_shuffle8!(a, vqrshrn_n_s32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Signed saturating rounded shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrshrn_high_n_s64<const N: i32>(a: int32x2_t, b: int64x2_t) -> int32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_shuffle4!(a, vqrshrn_n_s64::<N>(b), [0, 1, 2, 3])
}

/// Unsigned saturating rounded shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrnh_n_u16<const N: i32>(a: u16) -> u8 {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    let a: uint16x8_t = vdupq_n_u16(a);
    simd_extract(vqrshrn_n_u16::<N>(a), 0)
}

/// Unsigned saturating rounded shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrns_n_u32<const N: i32>(a: u32) -> u16 {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    let a: uint32x4_t = vdupq_n_u32(a);
    simd_extract(vqrshrn_n_u32::<N>(a), 0)
}

/// Unsigned saturating rounded shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrnd_n_u64<const N: i32>(a: u64) -> u32 {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    let a: uint64x2_t = vdupq_n_u64(a);
    simd_extract(vqrshrn_n_u64::<N>(a), 0)
}

/// Unsigned saturating rounded shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrshrn_high_n_u16<const N: i32>(a: uint8x8_t, b: uint16x8_t) -> uint8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_shuffle16!(a, vqrshrn_n_u16::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Unsigned saturating rounded shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrshrn_high_n_u32<const N: i32>(a: uint16x4_t, b: uint32x4_t) -> uint16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_shuffle8!(a, vqrshrn_n_u32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Unsigned saturating rounded shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrshrn_high_n_u64<const N: i32>(a: uint32x2_t, b: uint64x2_t) -> uint32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_shuffle4!(a, vqrshrn_n_u64::<N>(b), [0, 1, 2, 3])
}

/// Signed saturating rounded shift right unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrunh_n_s16<const N: i32>(a: i16) -> u8 {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    let a: int16x8_t = vdupq_n_s16(a);
    simd_extract(vqrshrun_n_s16::<N>(a), 0)
}

/// Signed saturating rounded shift right unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshruns_n_s32<const N: i32>(a: i32) -> u16 {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    let a: int32x4_t = vdupq_n_s32(a);
    simd_extract(vqrshrun_n_s32::<N>(a), 0)
}

/// Signed saturating rounded shift right unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrund_n_s64<const N: i32>(a: i64) -> u32 {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    let a: int64x2_t = vdupq_n_s64(a);
    simd_extract(vqrshrun_n_s64::<N>(a), 0)
}

/// Signed saturating rounded shift right unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrun2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrshrun_high_n_s16<const N: i32>(a: uint8x8_t, b: int16x8_t) -> uint8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_shuffle16!(a, vqrshrun_n_s16::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Signed saturating rounded shift right unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrun2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrshrun_high_n_s32<const N: i32>(a: uint16x4_t, b: int32x4_t) -> uint16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_shuffle8!(a, vqrshrun_n_s32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Signed saturating rounded shift right unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrun2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrshrun_high_n_s64<const N: i32>(a: uint32x2_t, b: int64x2_t) -> uint32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_shuffle4!(a, vqrshrun_n_s64::<N>(b), [0, 1, 2, 3])
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl))]
pub unsafe fn vqshld_s64(a: i64, b: i64) -> i64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshl.i64")]
        fn vqshld_s64_(a: i64, b: i64) -> i64;
    }
    vqshld_s64_(a, b)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl))]
pub unsafe fn vqshlb_s8(a: i8, b: i8) -> i8 {
    let c: int8x8_t = vqshl_s8(vdup_n_s8(a), vdup_n_s8(b));
    simd_extract(c, 0)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl))]
pub unsafe fn vqshlh_s16(a: i16, b: i16) -> i16 {
    let c: int16x4_t = vqshl_s16(vdup_n_s16(a), vdup_n_s16(b));
    simd_extract(c, 0)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl))]
pub unsafe fn vqshls_s32(a: i32, b: i32) -> i32 {
    let c: int32x2_t = vqshl_s32(vdup_n_s32(a), vdup_n_s32(b));
    simd_extract(c, 0)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl))]
pub unsafe fn vqshld_u64(a: u64, b: i64) -> u64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqshl.i64")]
        fn vqshld_u64_(a: u64, b: i64) -> u64;
    }
    vqshld_u64_(a, b)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl))]
pub unsafe fn vqshlb_u8(a: u8, b: i8) -> u8 {
    let c: uint8x8_t = vqshl_u8(vdup_n_u8(a), vdup_n_s8(b));
    simd_extract(c, 0)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl))]
pub unsafe fn vqshlh_u16(a: u16, b: i16) -> u16 {
    let c: uint16x4_t = vqshl_u16(vdup_n_u16(a), vdup_n_s16(b));
    simd_extract(c, 0)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl))]
pub unsafe fn vqshls_u32(a: u32, b: i32) -> u32 {
    let c: uint32x2_t = vqshl_u32(vdup_n_u32(a), vdup_n_s32(b));
    simd_extract(c, 0)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshlb_n_s8<const N: i32>(a: i8) -> i8 {
    static_assert_imm3!(N);
    simd_extract(vqshl_n_s8::<N>(vdup_n_s8(a)), 0)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshlh_n_s16<const N: i32>(a: i16) -> i16 {
    static_assert_imm4!(N);
    simd_extract(vqshl_n_s16::<N>(vdup_n_s16(a)), 0)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshls_n_s32<const N: i32>(a: i32) -> i32 {
    static_assert_imm5!(N);
    simd_extract(vqshl_n_s32::<N>(vdup_n_s32(a)), 0)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshld_n_s64<const N: i32>(a: i64) -> i64 {
    static_assert_imm6!(N);
    simd_extract(vqshl_n_s64::<N>(vdup_n_s64(a)), 0)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshlb_n_u8<const N: i32>(a: u8) -> u8 {
    static_assert_imm3!(N);
    simd_extract(vqshl_n_u8::<N>(vdup_n_u8(a)), 0)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshlh_n_u16<const N: i32>(a: u16) -> u16 {
    static_assert_imm4!(N);
    simd_extract(vqshl_n_u16::<N>(vdup_n_u16(a)), 0)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshls_n_u32<const N: i32>(a: u32) -> u32 {
    static_assert_imm5!(N);
    simd_extract(vqshl_n_u32::<N>(vdup_n_u32(a)), 0)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshld_n_u64<const N: i32>(a: u64) -> u64 {
    static_assert_imm6!(N);
    simd_extract(vqshl_n_u64::<N>(vdup_n_u64(a)), 0)
}

/// Signed saturating shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrnd_n_s64<const N: i32>(a: i64) -> i32 {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshrn.i32")]
        fn vqshrnd_n_s64_(a: i64, n: i32) -> i32;
    }
    vqshrnd_n_s64_(a, N)
}

/// Signed saturating shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrnh_n_s16<const N: i32>(a: i16) -> i8 {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_extract(vqshrn_n_s16::<N>(vdupq_n_s16(a)), 0)
}

/// Signed saturating shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrns_n_s32<const N: i32>(a: i32) -> i16 {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_extract(vqshrn_n_s32::<N>(vdupq_n_s32(a)), 0)
}

/// Signed saturating shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqshrn_high_n_s16<const N: i32>(a: int8x8_t, b: int16x8_t) -> int8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_shuffle16!(a, vqshrn_n_s16::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Signed saturating shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqshrn_high_n_s32<const N: i32>(a: int16x4_t, b: int32x4_t) -> int16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_shuffle8!(a, vqshrn_n_s32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Signed saturating shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqshrn_high_n_s64<const N: i32>(a: int32x2_t, b: int64x2_t) -> int32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_shuffle4!(a, vqshrn_n_s64::<N>(b), [0, 1, 2, 3])
}

/// Unsigned saturating shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrnd_n_u64<const N: i32>(a: u64) -> u32 {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqshrn.i32")]
        fn vqshrnd_n_u64_(a: u64, n: i32) -> u32;
    }
    vqshrnd_n_u64_(a, N)
}

/// Unsigned saturating shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrnh_n_u16<const N: i32>(a: u16) -> u8 {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_extract(vqshrn_n_u16::<N>(vdupq_n_u16(a)), 0)
}

/// Unsigned saturating shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrns_n_u32<const N: i32>(a: u32) -> u16 {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_extract(vqshrn_n_u32::<N>(vdupq_n_u32(a)), 0)
}

/// Unsigned saturating shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqshrn_high_n_u16<const N: i32>(a: uint8x8_t, b: uint16x8_t) -> uint8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_shuffle16!(a, vqshrn_n_u16::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Unsigned saturating shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqshrn_high_n_u32<const N: i32>(a: uint16x4_t, b: uint32x4_t) -> uint16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_shuffle8!(a, vqshrn_n_u32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Unsigned saturating shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqshrn_high_n_u64<const N: i32>(a: uint32x2_t, b: uint64x2_t) -> uint32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_shuffle4!(a, vqshrn_n_u64::<N>(b), [0, 1, 2, 3])
}

/// Signed saturating shift right unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrunh_n_s16<const N: i32>(a: i16) -> u8 {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_extract(vqshrun_n_s16::<N>(vdupq_n_s16(a)), 0)
}

/// Signed saturating shift right unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshruns_n_s32<const N: i32>(a: i32) -> u16 {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_extract(vqshrun_n_s32::<N>(vdupq_n_s32(a)), 0)
}

/// Signed saturating shift right unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrund_n_s64<const N: i32>(a: i64) -> u32 {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_extract(vqshrun_n_s64::<N>(vdupq_n_s64(a)), 0)
}

/// Signed saturating shift right unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrun2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqshrun_high_n_s16<const N: i32>(a: uint8x8_t, b: int16x8_t) -> uint8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_shuffle16!(a, vqshrun_n_s16::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Signed saturating shift right unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrun2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqshrun_high_n_s32<const N: i32>(a: uint16x4_t, b: int32x4_t) -> uint16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_shuffle8!(a, vqshrun_n_s32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Signed saturating shift right unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrun2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqshrun_high_n_s64<const N: i32>(a: uint32x2_t, b: int64x2_t) -> uint32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_shuffle4!(a, vqshrun_n_s64::<N>(b), [0, 1, 2, 3])
}

/// Calculates the square root of each lane.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fsqrt))]
pub unsafe fn vsqrt_f32(a: float32x2_t) -> float32x2_t {
    simd_fsqrt(a)
}

/// Calculates the square root of each lane.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fsqrt))]
pub unsafe fn vsqrtq_f32(a: float32x4_t) -> float32x4_t {
    simd_fsqrt(a)
}

/// Calculates the square root of each lane.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fsqrt))]
pub unsafe fn vsqrt_f64(a: float64x1_t) -> float64x1_t {
    simd_fsqrt(a)
}

/// Calculates the square root of each lane.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fsqrt))]
pub unsafe fn vsqrtq_f64(a: float64x2_t) -> float64x2_t {
    simd_fsqrt(a)
}

/// Reciprocal square-root estimate.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frsqrte))]
pub unsafe fn vrsqrte_f64(a: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.frsqrte.v1f64")]
        fn vrsqrte_f64_(a: float64x1_t) -> float64x1_t;
    }
    vrsqrte_f64_(a)
}

/// Reciprocal square-root estimate.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frsqrte))]
pub unsafe fn vrsqrteq_f64(a: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.frsqrte.v2f64")]
        fn vrsqrteq_f64_(a: float64x2_t) -> float64x2_t;
    }
    vrsqrteq_f64_(a)
}

/// Reciprocal estimate.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frecpe))]
pub unsafe fn vrecpe_f64(a: float64x1_t) -> float64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.frecpe.v1f64")]
        fn vrecpe_f64_(a: float64x1_t) -> float64x1_t;
    }
    vrecpe_f64_(a)
}

/// Reciprocal estimate.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frecpe))]
pub unsafe fn vrecpeq_f64(a: float64x2_t) -> float64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.frecpe.v2f64")]
        fn vrecpeq_f64_(a: float64x2_t) -> float64x2_t;
    }
    vrecpeq_f64_(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_s64_p64(a: poly64x1_t) -> int64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_u64_p64(a: poly64x1_t) -> uint64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p64_s64(a: int64x1_t) -> poly64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p64_u64(a: uint64x1_t) -> poly64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_s64_p64(a: poly64x2_t) -> int64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_u64_p64(a: poly64x2_t) -> uint64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p64_s64(a: int64x2_t) -> poly64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p64_u64(a: uint64x2_t) -> poly64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_s32_p64(a: poly64x1_t) -> int32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_u32_p64(a: poly64x1_t) -> uint32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_s32_p64(a: poly64x2_t) -> int32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_u32_p64(a: poly64x2_t) -> uint32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p64_s32(a: int32x2_t) -> poly64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p64_u32(a: uint32x2_t) -> poly64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p64_s32(a: int32x4_t) -> poly64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p64_u32(a: uint32x4_t) -> poly64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_s16_p64(a: poly64x1_t) -> int16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_u16_p64(a: poly64x1_t) -> uint16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p16_p64(a: poly64x1_t) -> poly16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_s16_p64(a: poly64x2_t) -> int16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_u16_p64(a: poly64x2_t) -> uint16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p16_p64(a: poly64x2_t) -> poly16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p64_p16(a: poly16x4_t) -> poly64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p64_s16(a: int16x4_t) -> poly64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p64_u16(a: uint16x4_t) -> poly64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p64_p16(a: poly16x8_t) -> poly64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p64_s16(a: int16x8_t) -> poly64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p64_u16(a: uint16x8_t) -> poly64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_s8_p64(a: poly64x1_t) -> int8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_u8_p64(a: poly64x1_t) -> uint8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p8_p64(a: poly64x1_t) -> poly8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_s8_p64(a: poly64x2_t) -> int8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_u8_p64(a: poly64x2_t) -> uint8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p8_p64(a: poly64x2_t) -> poly8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p64_p8(a: poly8x8_t) -> poly64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p64_s8(a: int8x8_t) -> poly64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p64_u8(a: uint8x8_t) -> poly64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p64_p8(a: poly8x16_t) -> poly64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p64_s8(a: int8x16_t) -> poly64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p64_u8(a: uint8x16_t) -> poly64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_s8_f64(a: float64x1_t) -> int8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_s16_f64(a: float64x1_t) -> int16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_s32_f64(a: float64x1_t) -> int32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_s64_f64(a: float64x1_t) -> int64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_s8_f64(a: float64x2_t) -> int8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_s16_f64(a: float64x2_t) -> int16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_s32_f64(a: float64x2_t) -> int32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_s64_f64(a: float64x2_t) -> int64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_u8_f64(a: float64x1_t) -> uint8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_u16_f64(a: float64x1_t) -> uint16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_u32_f64(a: float64x1_t) -> uint32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_u64_f64(a: float64x1_t) -> uint64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_u8_f64(a: float64x2_t) -> uint8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_u16_f64(a: float64x2_t) -> uint16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_u32_f64(a: float64x2_t) -> uint32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_u64_f64(a: float64x2_t) -> uint64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p8_f64(a: float64x1_t) -> poly8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p16_f64(a: float64x1_t) -> poly16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p64_f32(a: float32x2_t) -> poly64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_p64_f64(a: float64x1_t) -> poly64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p8_f64(a: float64x2_t) -> poly8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p16_f64(a: float64x2_t) -> poly16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p64_f32(a: float32x4_t) -> poly64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_p64_f64(a: float64x2_t) -> poly64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_f64_s8(a: int8x8_t) -> float64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_f64_s16(a: int16x4_t) -> float64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_f64_s32(a: int32x2_t) -> float64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_f64_s64(a: int64x1_t) -> float64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_f64_s8(a: int8x16_t) -> float64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_f64_s16(a: int16x8_t) -> float64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_f64_s32(a: int32x4_t) -> float64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_f64_s64(a: int64x2_t) -> float64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_f64_p8(a: poly8x8_t) -> float64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_f64_u16(a: uint16x4_t) -> float64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_f64_u32(a: uint32x2_t) -> float64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_f64_u64(a: uint64x1_t) -> float64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_f64_p8(a: poly8x16_t) -> float64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_f64_u16(a: uint16x8_t) -> float64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_f64_u32(a: uint32x4_t) -> float64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_f64_u64(a: uint64x2_t) -> float64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_f64_u8(a: uint8x8_t) -> float64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_f64_p16(a: poly16x4_t) -> float64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_f64_p64(a: poly64x1_t) -> float64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_f32_p64(a: poly64x1_t) -> float32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_f64_u8(a: uint8x16_t) -> float64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_f64_p16(a: poly16x8_t) -> float64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_f64_p64(a: poly64x2_t) -> float64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_f32_p64(a: poly64x2_t) -> float32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_f64_f32(a: float32x2_t) -> float64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpret_f32_f64(a: float64x1_t) -> float32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_f64_f32(a: float32x4_t) -> float64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(str))]
pub unsafe fn vreinterpretq_f32_f64(a: float64x2_t) -> float32x4_t {
    transmute(a)
}

/// Signed rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(srshl))]
pub unsafe fn vrshld_s64(a: i64, b: i64) -> i64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srshl.i64")]
        fn vrshld_s64_(a: i64, b: i64) -> i64;
    }
    vrshld_s64_(a, b)
}

/// Unsigned rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(urshl))]
pub unsafe fn vrshld_u64(a: u64, b: i64) -> u64 {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urshl.i64")]
        fn vrshld_u64_(a: u64, b: i64) -> u64;
    }
    vrshld_u64_(a, b)
}

/// Signed rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(srshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrd_n_s64<const N: i32>(a: i64) -> i64 {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    vrshld_s64(a, -N as i64)
}

/// Unsigned rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(urshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrd_n_u64<const N: i32>(a: u64) -> u64 {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    vrshld_u64(a, -N as i64)
}

/// Rounding shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrshrn_high_n_s16<const N: i32>(a: int8x8_t, b: int16x8_t) -> int8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_shuffle16!(a, vrshrn_n_s16::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Rounding shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrshrn_high_n_s32<const N: i32>(a: int16x4_t, b: int32x4_t) -> int16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_shuffle8!(a, vrshrn_n_s32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Rounding shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrshrn_high_n_s64<const N: i32>(a: int32x2_t, b: int64x2_t) -> int32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_shuffle4!(a, vrshrn_n_s64::<N>(b), [0, 1, 2, 3])
}

/// Rounding shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrshrn_high_n_u16<const N: i32>(a: uint8x8_t, b: uint16x8_t) -> uint8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_shuffle16!(a, vrshrn_n_u16::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Rounding shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrshrn_high_n_u32<const N: i32>(a: uint16x4_t, b: uint32x4_t) -> uint16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_shuffle8!(a, vrshrn_n_u32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Rounding shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrshrn_high_n_u64<const N: i32>(a: uint32x2_t, b: uint64x2_t) -> uint32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_shuffle4!(a, vrshrn_n_u64::<N>(b), [0, 1, 2, 3])
}

/// Signed rounding shift right and accumulate.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(srsra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsrad_n_s64<const N: i32>(a: i64, b: i64) -> i64 {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    let b: i64 = vrshrd_n_s64::<N>(b);
    a + b
}

/// Ungisned rounding shift right and accumulate.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ursra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsrad_n_u64<const N: i32>(a: u64, b: u64) -> u64 {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    let b: u64 = vrshrd_n_u64::<N>(b);
    a + b
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vset_lane_f64<const LANE: i32>(a: f64, b: float64x1_t) -> float64x1_t {
    static_assert!(LANE : i32 where LANE == 0);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsetq_lane_f64<const LANE: i32>(a: f64, b: float64x2_t) -> float64x2_t {
    static_assert_imm1!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Signed Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshl))]
pub unsafe fn vshld_s64(a: i64, b: i64) -> i64 {
    transmute(vshl_s64(transmute(a), transmute(b)))
}

/// Unsigned Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ushl))]
pub unsafe fn vshld_u64(a: u64, b: i64) -> u64 {
    transmute(vshl_u64(transmute(a), transmute(b)))
}

/// Signed shift left long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshll2, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshll_high_n_s8<const N: i32>(a: int8x16_t) -> int16x8_t {
    static_assert!(N : i32 where N >= 0 && N <= 8);
    let b: int8x8_t = simd_shuffle8!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    vshll_n_s8::<N>(b)
}

/// Signed shift left long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshll2, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshll_high_n_s16<const N: i32>(a: int16x8_t) -> int32x4_t {
    static_assert!(N : i32 where N >= 0 && N <= 16);
    let b: int16x4_t = simd_shuffle4!(a, a, [4, 5, 6, 7]);
    vshll_n_s16::<N>(b)
}

/// Signed shift left long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshll2, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshll_high_n_s32<const N: i32>(a: int32x4_t) -> int64x2_t {
    static_assert!(N : i32 where N >= 0 && N <= 32);
    let b: int32x2_t = simd_shuffle2!(a, a, [2, 3]);
    vshll_n_s32::<N>(b)
}

/// Signed shift left long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ushll2, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshll_high_n_u8<const N: i32>(a: uint8x16_t) -> uint16x8_t {
    static_assert!(N : i32 where N >= 0 && N <= 8);
    let b: uint8x8_t = simd_shuffle8!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    vshll_n_u8::<N>(b)
}

/// Signed shift left long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ushll2, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshll_high_n_u16<const N: i32>(a: uint16x8_t) -> uint32x4_t {
    static_assert!(N : i32 where N >= 0 && N <= 16);
    let b: uint16x4_t = simd_shuffle4!(a, a, [4, 5, 6, 7]);
    vshll_n_u16::<N>(b)
}

/// Signed shift left long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ushll2, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshll_high_n_u32<const N: i32>(a: uint32x4_t) -> uint64x2_t {
    static_assert!(N : i32 where N >= 0 && N <= 32);
    let b: uint32x2_t = simd_shuffle2!(a, a, [2, 3]);
    vshll_n_u32::<N>(b)
}

/// Shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(shrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vshrn_high_n_s16<const N: i32>(a: int8x8_t, b: int16x8_t) -> int8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_shuffle16!(a, vshrn_n_s16::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(shrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vshrn_high_n_s32<const N: i32>(a: int16x4_t, b: int32x4_t) -> int16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_shuffle8!(a, vshrn_n_s32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(shrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vshrn_high_n_s64<const N: i32>(a: int32x2_t, b: int64x2_t) -> int32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_shuffle4!(a, vshrn_n_s64::<N>(b), [0, 1, 2, 3])
}

/// Shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(shrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vshrn_high_n_u16<const N: i32>(a: uint8x8_t, b: uint16x8_t) -> uint8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_shuffle16!(a, vshrn_n_u16::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(shrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vshrn_high_n_u32<const N: i32>(a: uint16x4_t, b: uint32x4_t) -> uint16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_shuffle8!(a, vshrn_n_u32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(shrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vshrn_high_n_u64<const N: i32>(a: uint32x2_t, b: uint64x2_t) -> uint32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_shuffle4!(a, vshrn_n_u64::<N>(b), [0, 1, 2, 3])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_shuffle8!(a, b, [0, 8, 2, 10, 4, 12, 6, 14])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1q_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_shuffle16!(a, b, [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_shuffle4!(a, b, [0, 4, 2, 6])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1q_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_shuffle8!(a, b, [0, 8, 2, 10, 4, 12, 6, 14])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1q_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_shuffle4!(a, b, [0, 4, 2, 6])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_shuffle8!(a, b, [0, 8, 2, 10, 4, 12, 6, 14])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_shuffle16!(a, b, [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_shuffle4!(a, b, [0, 4, 2, 6])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1q_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_shuffle8!(a, b, [0, 8, 2, 10, 4, 12, 6, 14])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1q_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_shuffle4!(a, b, [0, 4, 2, 6])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1_p8(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    simd_shuffle8!(a, b, [0, 8, 2, 10, 4, 12, 6, 14])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1q_p8(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    simd_shuffle16!(a, b, [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1_p16(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    simd_shuffle4!(a, b, [0, 4, 2, 6])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1q_p16(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    simd_shuffle8!(a, b, [0, 8, 2, 10, 4, 12, 6, 14])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vtrn1_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vtrn1q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vtrn1_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vtrn1q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vtrn1q_p64(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn1))]
pub unsafe fn vtrn1q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_shuffle4!(a, b, [0, 4, 2, 6])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vtrn1_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vtrn1q_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_shuffle8!(a, b, [1, 9, 3, 11, 5, 13, 7, 15])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2q_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_shuffle16!(a, b, [1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_shuffle4!(a, b, [1, 5, 3, 7])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2q_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_shuffle8!(a, b, [1, 9, 3, 11, 5, 13, 7, 15])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2q_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_shuffle4!(a, b, [1, 5, 3, 7])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_shuffle8!(a, b, [1, 9, 3, 11, 5, 13, 7, 15])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_shuffle16!(a, b, [1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_shuffle4!(a, b, [1, 5, 3, 7])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2q_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_shuffle8!(a, b, [1, 9, 3, 11, 5, 13, 7, 15])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2q_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_shuffle4!(a, b, [1, 5, 3, 7])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2_p8(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    simd_shuffle8!(a, b, [1, 9, 3, 11, 5, 13, 7, 15])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2q_p8(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    simd_shuffle16!(a, b, [1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2_p16(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    simd_shuffle4!(a, b, [1, 5, 3, 7])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2q_p16(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    simd_shuffle8!(a, b, [1, 9, 3, 11, 5, 13, 7, 15])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vtrn2_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vtrn2q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vtrn2_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vtrn2q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vtrn2q_p64(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(trn2))]
pub unsafe fn vtrn2q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_shuffle4!(a, b, [1, 5, 3, 7])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vtrn2_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Transpose vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vtrn2q_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_shuffle8!(a, b, [0, 8, 1, 9, 2, 10, 3, 11])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1q_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_shuffle16!(a, b, [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_shuffle4!(a, b, [0, 4, 1, 5])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1q_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_shuffle8!(a, b, [0, 8, 1, 9, 2, 10, 3, 11])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1q_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_shuffle4!(a, b, [0, 4, 1, 5])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_shuffle8!(a, b, [0, 8, 1, 9, 2, 10, 3, 11])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_shuffle16!(a, b, [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_shuffle4!(a, b, [0, 4, 1, 5])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1q_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_shuffle8!(a, b, [0, 8, 1, 9, 2, 10, 3, 11])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1q_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_shuffle4!(a, b, [0, 4, 1, 5])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1_p8(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    simd_shuffle8!(a, b, [0, 8, 1, 9, 2, 10, 3, 11])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1q_p8(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    simd_shuffle16!(a, b, [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1_p16(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    simd_shuffle4!(a, b, [0, 4, 1, 5])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1q_p16(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    simd_shuffle8!(a, b, [0, 8, 1, 9, 2, 10, 3, 11])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1q_p64(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_shuffle4!(a, b, [0, 4, 1, 5])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vzip1q_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_shuffle8!(a, b, [4, 12, 5, 13, 6, 14, 7, 15])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2q_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_shuffle16!(a, b, [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_shuffle4!(a, b, [2, 6, 3, 7])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2q_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_shuffle8!(a, b, [4, 12, 5, 13, 6, 14, 7, 15])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2q_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_shuffle4!(a, b, [2, 6, 3, 7])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_shuffle8!(a, b, [4, 12, 5, 13, 6, 14, 7, 15])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_shuffle16!(a, b, [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_shuffle4!(a, b, [2, 6, 3, 7])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2q_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_shuffle8!(a, b, [4, 12, 5, 13, 6, 14, 7, 15])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2q_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_shuffle4!(a, b, [2, 6, 3, 7])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2_p8(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    simd_shuffle8!(a, b, [4, 12, 5, 13, 6, 14, 7, 15])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2q_p8(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    simd_shuffle16!(a, b, [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2_p16(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    simd_shuffle4!(a, b, [2, 6, 3, 7])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2q_p16(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    simd_shuffle8!(a, b, [4, 12, 5, 13, 6, 14, 7, 15])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2q_p64(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_shuffle4!(a, b, [2, 6, 3, 7])
}

/// Zip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vzip2q_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_shuffle8!(a, b, [0, 2, 4, 6, 8, 10, 12, 14])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1q_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_shuffle16!(a, b, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_shuffle4!(a, b, [0, 2, 4, 6])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1q_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_shuffle8!(a, b, [0, 2, 4, 6, 8, 10, 12, 14])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1q_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_shuffle4!(a, b, [0, 2, 4, 6])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_shuffle8!(a, b, [0, 2, 4, 6, 8, 10, 12, 14])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_shuffle16!(a, b, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_shuffle4!(a, b, [0, 2, 4, 6])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1q_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_shuffle8!(a, b, [0, 2, 4, 6, 8, 10, 12, 14])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1q_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_shuffle4!(a, b, [0, 2, 4, 6])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1_p8(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    simd_shuffle8!(a, b, [0, 2, 4, 6, 8, 10, 12, 14])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1q_p8(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    simd_shuffle16!(a, b, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1_p16(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    simd_shuffle4!(a, b, [0, 2, 4, 6])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1q_p16(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    simd_shuffle8!(a, b, [0, 2, 4, 6, 8, 10, 12, 14])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vuzp1_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vuzp1q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vuzp1_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vuzp1q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vuzp1q_p64(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp1))]
pub unsafe fn vuzp1q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_shuffle4!(a, b, [0, 2, 4, 6])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vuzp1_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip1))]
pub unsafe fn vuzp1q_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_shuffle2!(a, b, [0, 2])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_shuffle8!(a, b, [1, 3, 5, 7, 9, 11, 13, 15])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2q_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_shuffle16!(a, b, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_shuffle4!(a, b, [1, 3, 5, 7])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2q_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_shuffle8!(a, b, [1, 3, 5, 7, 9, 11, 13, 15])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2q_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_shuffle4!(a, b, [1, 3, 5, 7])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_shuffle8!(a, b, [1, 3, 5, 7, 9, 11, 13, 15])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_shuffle16!(a, b, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_shuffle4!(a, b, [1, 3, 5, 7])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2q_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_shuffle8!(a, b, [1, 3, 5, 7, 9, 11, 13, 15])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2q_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_shuffle4!(a, b, [1, 3, 5, 7])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2_p8(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    simd_shuffle8!(a, b, [1, 3, 5, 7, 9, 11, 13, 15])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2q_p8(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    simd_shuffle16!(a, b, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2_p16(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    simd_shuffle4!(a, b, [1, 3, 5, 7])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2q_p16(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    simd_shuffle8!(a, b, [1, 3, 5, 7, 9, 11, 13, 15])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vuzp2_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vuzp2q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vuzp2_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vuzp2q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vuzp2q_p64(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uzp2))]
pub unsafe fn vuzp2q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_shuffle4!(a, b, [1, 3, 5, 7])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vuzp2_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Unzip vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(zip2))]
pub unsafe fn vuzp2q_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_shuffle2!(a, b, [1, 3])
}

/// Unsigned Absolute difference and Accumulate Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uabal))]
pub unsafe fn vabal_high_u8(a: uint16x8_t, b: uint8x16_t, c: uint8x16_t) -> uint16x8_t {
    let d: uint8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let e: uint8x8_t = simd_shuffle8!(c, c, [8, 9, 10, 11, 12, 13, 14, 15]);
    let f: uint8x8_t = vabd_u8(d, e);
    simd_add(a, simd_cast(f))
}

/// Unsigned Absolute difference and Accumulate Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uabal))]
pub unsafe fn vabal_high_u16(a: uint32x4_t, b: uint16x8_t, c: uint16x8_t) -> uint32x4_t {
    let d: uint16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    let e: uint16x4_t = simd_shuffle4!(c, c, [4, 5, 6, 7]);
    let f: uint16x4_t = vabd_u16(d, e);
    simd_add(a, simd_cast(f))
}

/// Unsigned Absolute difference and Accumulate Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uabal))]
pub unsafe fn vabal_high_u32(a: uint64x2_t, b: uint32x4_t, c: uint32x4_t) -> uint64x2_t {
    let d: uint32x2_t = simd_shuffle2!(b, b, [2, 3]);
    let e: uint32x2_t = simd_shuffle2!(c, c, [2, 3]);
    let f: uint32x2_t = vabd_u32(d, e);
    simd_add(a, simd_cast(f))
}

/// Signed Absolute difference and Accumulate Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sabal))]
pub unsafe fn vabal_high_s8(a: int16x8_t, b: int8x16_t, c: int8x16_t) -> int16x8_t {
    let d: int8x8_t = simd_shuffle8!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let e: int8x8_t = simd_shuffle8!(c, c, [8, 9, 10, 11, 12, 13, 14, 15]);
    let f: int8x8_t = vabd_s8(d, e);
    let f: uint8x8_t = simd_cast(f);
    simd_add(a, simd_cast(f))
}

/// Signed Absolute difference and Accumulate Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sabal))]
pub unsafe fn vabal_high_s16(a: int32x4_t, b: int16x8_t, c: int16x8_t) -> int32x4_t {
    let d: int16x4_t = simd_shuffle4!(b, b, [4, 5, 6, 7]);
    let e: int16x4_t = simd_shuffle4!(c, c, [4, 5, 6, 7]);
    let f: int16x4_t = vabd_s16(d, e);
    let f: uint16x4_t = simd_cast(f);
    simd_add(a, simd_cast(f))
}

/// Signed Absolute difference and Accumulate Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sabal))]
pub unsafe fn vabal_high_s32(a: int64x2_t, b: int32x4_t, c: int32x4_t) -> int64x2_t {
    let d: int32x2_t = simd_shuffle2!(b, b, [2, 3]);
    let e: int32x2_t = simd_shuffle2!(c, c, [2, 3]);
    let f: int32x2_t = vabd_s32(d, e);
    let f: uint32x2_t = simd_cast(f);
    simd_add(a, simd_cast(f))
}

/// Singned saturating Absolute value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqabs))]
pub unsafe fn vqabs_s64(a: int64x1_t) -> int64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqabs.v1i64")]
        fn vqabs_s64_(a: int64x1_t) -> int64x1_t;
    }
    vqabs_s64_(a)
}

/// Singned saturating Absolute value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqabs))]
pub unsafe fn vqabsq_s64(a: int64x2_t) -> int64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqabs.v2i64")]
        fn vqabsq_s64_(a: int64x2_t) -> int64x2_t;
    }
    vqabsq_s64_(a)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core_arch::simd::*;
    use std::mem::transmute;
    use stdarch_test::simd_test;

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_f64() {
        let a: f64 = 1.0;
        let b: f64 = 9.0;
        let e: f64 = 8.0;
        let r: f64 = transmute(vabd_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_f64() {
        let a: f64x2 = f64x2::new(1.0, 2.0);
        let b: f64x2 = f64x2::new(9.0, 3.0);
        let e: f64x2 = f64x2::new(8.0, 1.0);
        let r: f64x2 = transmute(vabdq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdl_high_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10);
        let e: u16x8 = u16x8::new(1, 0, 1, 2, 3, 4, 5, 6);
        let r: u16x8 = transmute(vabdl_high_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdl_high_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 8, 9, 11, 12);
        let b: u16x8 = u16x8::new(10, 10, 10, 10, 10, 10, 10, 10);
        let e: u32x4 = u32x4::new(2, 1, 1, 2);
        let r: u32x4 = transmute(vabdl_high_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdl_high_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(10, 10, 10, 10);
        let e: u64x2 = u64x2::new(7, 6);
        let r: u64x2 = transmute(vabdl_high_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdl_high_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10);
        let e: i16x8 = i16x8::new(1, 0, 1, 2, 3, 4, 5, 6);
        let r: i16x8 = transmute(vabdl_high_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdl_high_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 9, 10, 11, 12);
        let b: i16x8 = i16x8::new(10, 10, 10, 10, 10, 10, 10, 10);
        let e: i32x4 = i32x4::new(1, 0, 1, 2);
        let r: i32x4 = transmute(vabdl_high_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdl_high_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(10, 10, 10, 10);
        let e: i64x2 = i64x2::new(7, 6);
        let r: i64x2 = transmute(vabdl_high_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u64() {
        let a: u64x1 = u64x1::new(0);
        let b: u64x1 = u64x1::new(0);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vceq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u64x1 = u64x1::new(0);
        let b: u64x1 = u64x1::new(0);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vceq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u64() {
        let a: u64x2 = u64x2::new(0, 0x01);
        let b: u64x2 = u64x2::new(0, 0x01);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vceqq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u64x2 = u64x2::new(0, 0);
        let b: u64x2 = u64x2::new(0, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0);
        let r: u64x2 = transmute(vceqq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s64() {
        let a: i64x1 = i64x1::new(-9223372036854775808);
        let b: i64x1 = i64x1::new(-9223372036854775808);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vceq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i64x1 = i64x1::new(-9223372036854775808);
        let b: i64x1 = i64x1::new(-9223372036854775808);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vceq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s64() {
        let a: i64x2 = i64x2::new(-9223372036854775808, 0x01);
        let b: i64x2 = i64x2::new(-9223372036854775808, 0x01);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vceqq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i64x2 = i64x2::new(-9223372036854775808, -9223372036854775808);
        let b: i64x2 = i64x2::new(-9223372036854775808, 0x7F_FF_FF_FF_FF_FF_FF_FF);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0);
        let r: u64x2 = transmute(vceqq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_p64() {
        let a: i64x1 = i64x1::new(-9223372036854775808);
        let b: i64x1 = i64x1::new(-9223372036854775808);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vceq_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i64x1 = i64x1::new(-9223372036854775808);
        let b: i64x1 = i64x1::new(-9223372036854775808);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vceq_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_p64() {
        let a: i64x2 = i64x2::new(-9223372036854775808, 0x01);
        let b: i64x2 = i64x2::new(-9223372036854775808, 0x01);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vceqq_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i64x2 = i64x2::new(-9223372036854775808, -9223372036854775808);
        let b: i64x2 = i64x2::new(-9223372036854775808, 0x7F_FF_FF_FF_FF_FF_FF_FF);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0);
        let r: u64x2 = transmute(vceqq_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_f64() {
        let a: f64 = 1.2;
        let b: f64 = 1.2;
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vceq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_f64() {
        let a: f64x2 = f64x2::new(1.2, 3.4);
        let b: f64x2 = f64x2::new(1.2, 3.4);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vceqq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqz_s8() {
        let a: i8x8 = i8x8::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u8x8 = u8x8::new(0, 0xFF, 0, 0, 0, 0, 0, 0);
        let r: u8x8 = transmute(vceqz_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqzq_s8() {
        let a: i8x16 = i8x16::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x7F);
        let e: u8x16 = u8x16::new(0, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x16 = transmute(vceqzq_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqz_s16() {
        let a: i16x4 = i16x4::new(-32768, 0x00, 0x01, 0x02);
        let e: u16x4 = u16x4::new(0, 0xFF_FF, 0, 0);
        let r: u16x4 = transmute(vceqz_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqzq_s16() {
        let a: i16x8 = i16x8::new(-32768, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u16x8 = u16x8::new(0, 0xFF_FF, 0, 0, 0, 0, 0, 0);
        let r: u16x8 = transmute(vceqzq_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqz_s32() {
        let a: i32x2 = i32x2::new(-2147483648, 0x00);
        let e: u32x2 = u32x2::new(0, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vceqz_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqzq_s32() {
        let a: i32x4 = i32x4::new(-2147483648, 0x00, 0x01, 0x02);
        let e: u32x4 = u32x4::new(0, 0xFF_FF_FF_FF, 0, 0);
        let r: u32x4 = transmute(vceqzq_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqz_s64() {
        let a: i64x1 = i64x1::new(-9223372036854775808);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vceqz_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqzq_s64() {
        let a: i64x2 = i64x2::new(-9223372036854775808, 0x00);
        let e: u64x2 = u64x2::new(0, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vceqzq_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqz_p8() {
        let a: i8x8 = i8x8::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u8x8 = u8x8::new(0, 0xFF, 0, 0, 0, 0, 0, 0);
        let r: u8x8 = transmute(vceqz_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqzq_p8() {
        let a: i8x16 = i8x16::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x7F);
        let e: u8x16 = u8x16::new(0, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x16 = transmute(vceqzq_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqz_p64() {
        let a: i64x1 = i64x1::new(-9223372036854775808);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vceqz_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqzq_p64() {
        let a: i64x2 = i64x2::new(-9223372036854775808, 0x00);
        let e: u64x2 = u64x2::new(0, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vceqzq_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqz_u8() {
        let a: u8x8 = u8x8::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0, 0, 0, 0, 0, 0);
        let r: u8x8 = transmute(vceqz_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqzq_u8() {
        let a: u8x16 = u8x16::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0xFF);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x16 = transmute(vceqzq_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqz_u16() {
        let a: u16x4 = u16x4::new(0, 0x00, 0x01, 0x02);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0, 0);
        let r: u16x4 = transmute(vceqz_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqzq_u16() {
        let a: u16x8 = u16x8::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0, 0, 0, 0, 0, 0);
        let r: u16x8 = transmute(vceqzq_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqz_u32() {
        let a: u32x2 = u32x2::new(0, 0x00);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vceqz_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqzq_u32() {
        let a: u32x4 = u32x4::new(0, 0x00, 0x01, 0x02);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0, 0);
        let r: u32x4 = transmute(vceqzq_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqz_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vceqz_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqzq_u64() {
        let a: u64x2 = u64x2::new(0, 0x00);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vceqzq_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqz_f32() {
        let a: f32x2 = f32x2::new(0.0, 1.2);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0);
        let r: u32x2 = transmute(vceqz_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqzq_f32() {
        let a: f32x4 = f32x4::new(0.0, 1.2, 3.4, 5.6);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0, 0, 0);
        let r: u32x4 = transmute(vceqzq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqz_f64() {
        let a: f64 = 0.0;
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vceqz_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqzq_f64() {
        let a: f64x2 = f64x2::new(0.0, 1.2);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0);
        let r: u64x2 = transmute(vceqzq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_s64() {
        let a: i64x1 = i64x1::new(-9223372036854775808);
        let b: i64x1 = i64x1::new(-9223372036854775808);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vtst_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_s64() {
        let a: i64x2 = i64x2::new(-9223372036854775808, 0x00);
        let b: i64x2 = i64x2::new(-9223372036854775808, 0x00);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0);
        let r: u64x2 = transmute(vtstq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_p64() {
        let a: i64x1 = i64x1::new(-9223372036854775808);
        let b: i64x1 = i64x1::new(-9223372036854775808);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vtst_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_p64() {
        let a: i64x2 = i64x2::new(-9223372036854775808, 0x00);
        let b: i64x2 = i64x2::new(-9223372036854775808, 0x00);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0);
        let r: u64x2 = transmute(vtstq_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_u64() {
        let a: u64x1 = u64x1::new(0);
        let b: u64x1 = u64x1::new(0);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vtst_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_u64() {
        let a: u64x2 = u64x2::new(0, 0x00);
        let b: u64x2 = u64x2::new(0, 0x00);
        let e: u64x2 = u64x2::new(0, 0);
        let r: u64x2 = transmute(vtstq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabs_f64() {
        let a: f64 = -0.1;
        let e: f64 = 0.1;
        let r: f64 = transmute(vabs_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabsq_f64() {
        let a: f64x2 = f64x2::new(-0.1, -2.2);
        let e: f64x2 = f64x2::new(0.1, 2.2);
        let r: f64x2 = transmute(vabsq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s64() {
        let a: i64x1 = i64x1::new(1);
        let b: i64x1 = i64x1::new(0);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcgt_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s64() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i64x2 = i64x2::new(0, 1);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcgtq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u64() {
        let a: u64x1 = u64x1::new(1);
        let b: u64x1 = u64x1::new(0);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcgt_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u64() {
        let a: u64x2 = u64x2::new(1, 2);
        let b: u64x2 = u64x2::new(0, 1);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcgtq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_f64() {
        let a: f64 = 1.2;
        let b: f64 = 0.1;
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcgt_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_f64() {
        let a: f64x2 = f64x2::new(1.2, 2.3);
        let b: f64x2 = f64x2::new(0.1, 1.2);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcgtq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s64() {
        let a: i64x1 = i64x1::new(0);
        let b: i64x1 = i64x1::new(1);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vclt_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let b: i64x2 = i64x2::new(1, 2);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcltq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u64() {
        let a: u64x1 = u64x1::new(0);
        let b: u64x1 = u64x1::new(1);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vclt_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let b: u64x2 = u64x2::new(1, 2);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcltq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_f64() {
        let a: f64 = 0.1;
        let b: f64 = 1.2;
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vclt_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_f64() {
        let a: f64x2 = f64x2::new(0.1, 1.2);
        let b: f64x2 = f64x2::new(1.2, 2.3);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcltq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s64() {
        let a: i64x1 = i64x1::new(0);
        let b: i64x1 = i64x1::new(1);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcle_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let b: i64x2 = i64x2::new(1, 2);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcleq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u64() {
        let a: u64x1 = u64x1::new(0);
        let b: u64x1 = u64x1::new(1);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcle_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let b: u64x2 = u64x2::new(1, 2);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcleq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_f64() {
        let a: f64 = 0.1;
        let b: f64 = 1.2;
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcle_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_f64() {
        let a: f64x2 = f64x2::new(0.1, 1.2);
        let b: f64x2 = f64x2::new(1.2, 2.3);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcleq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s64() {
        let a: i64x1 = i64x1::new(1);
        let b: i64x1 = i64x1::new(0);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcge_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s64() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i64x2 = i64x2::new(0, 1);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcgeq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u64() {
        let a: u64x1 = u64x1::new(1);
        let b: u64x1 = u64x1::new(0);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcge_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u64() {
        let a: u64x2 = u64x2::new(1, 2);
        let b: u64x2 = u64x2::new(0, 1);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcgeq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_f64() {
        let a: f64 = 1.2;
        let b: f64 = 0.1;
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcge_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_f64() {
        let a: f64x2 = f64x2::new(1.2, 2.3);
        let b: f64x2 = f64x2::new(0.1, 1.2);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcgeq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgez_s8() {
        let a: i8x8 = i8x8::new(-128, -1, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05);
        let e: u8x8 = u8x8::new(0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcgez_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgezq_s8() {
        let a: i8x16 = i8x16::new(-128, -1, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x7F);
        let e: u8x16 = u8x16::new(0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcgezq_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgez_s16() {
        let a: i16x4 = i16x4::new(-32768, -1, 0x00, 0x01);
        let e: u16x4 = u16x4::new(0, 0, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcgez_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgezq_s16() {
        let a: i16x8 = i16x8::new(-32768, -1, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05);
        let e: u16x8 = u16x8::new(0, 0, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcgezq_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgez_s32() {
        let a: i32x2 = i32x2::new(-2147483648, -1);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vcgez_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgezq_s32() {
        let a: i32x4 = i32x4::new(-2147483648, -1, 0x00, 0x01);
        let e: u32x4 = u32x4::new(0, 0, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgezq_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgez_s64() {
        let a: i64x1 = i64x1::new(-9223372036854775808);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vcgez_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgezq_s64() {
        let a: i64x2 = i64x2::new(-9223372036854775808, -1);
        let e: u64x2 = u64x2::new(0, 0);
        let r: u64x2 = transmute(vcgezq_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgez_f32() {
        let a: f32x2 = f32x2::new(-1.2, 0.0);
        let e: u32x2 = u32x2::new(0, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcgez_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgezq_f32() {
        let a: f32x4 = f32x4::new(-1.2, 0.0, 1.2, 2.3);
        let e: u32x4 = u32x4::new(0, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgezq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgez_f64() {
        let a: f64 = -1.2;
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vcgez_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgezq_f64() {
        let a: f64x2 = f64x2::new(-1.2, 0.0);
        let e: u64x2 = u64x2::new(0, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcgezq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtz_s8() {
        let a: i8x8 = i8x8::new(-128, -1, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05);
        let e: u8x8 = u8x8::new(0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcgtz_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtzq_s8() {
        let a: i8x16 = i8x16::new(-128, -1, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x7F);
        let e: u8x16 = u8x16::new(0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcgtzq_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtz_s16() {
        let a: i16x4 = i16x4::new(-32768, -1, 0x00, 0x01);
        let e: u16x4 = u16x4::new(0, 0, 0, 0xFF_FF);
        let r: u16x4 = transmute(vcgtz_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtzq_s16() {
        let a: i16x8 = i16x8::new(-32768, -1, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05);
        let e: u16x8 = u16x8::new(0, 0, 0, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcgtzq_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtz_s32() {
        let a: i32x2 = i32x2::new(-2147483648, -1);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vcgtz_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtzq_s32() {
        let a: i32x4 = i32x4::new(-2147483648, -1, 0x00, 0x01);
        let e: u32x4 = u32x4::new(0, 0, 0, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgtzq_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtz_s64() {
        let a: i64x1 = i64x1::new(-9223372036854775808);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vcgtz_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtzq_s64() {
        let a: i64x2 = i64x2::new(-9223372036854775808, -1);
        let e: u64x2 = u64x2::new(0, 0);
        let r: u64x2 = transmute(vcgtzq_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtz_f32() {
        let a: f32x2 = f32x2::new(-1.2, 0.0);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vcgtz_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtzq_f32() {
        let a: f32x4 = f32x4::new(-1.2, 0.0, 1.2, 2.3);
        let e: u32x4 = u32x4::new(0, 0, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgtzq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtz_f64() {
        let a: f64 = -1.2;
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vcgtz_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtzq_f64() {
        let a: f64x2 = f64x2::new(-1.2, 0.0);
        let e: u64x2 = u64x2::new(0, 0);
        let r: u64x2 = transmute(vcgtzq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclez_s8() {
        let a: i8x8 = i8x8::new(-128, -1, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0);
        let r: u8x8 = transmute(vclez_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclezq_s8() {
        let a: i8x16 = i8x16::new(-128, -1, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x7F);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x16 = transmute(vclezq_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclez_s16() {
        let a: i16x4 = i16x4::new(-32768, -1, 0x00, 0x01);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0);
        let r: u16x4 = transmute(vclez_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclezq_s16() {
        let a: i16x8 = i16x8::new(-32768, -1, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0, 0, 0, 0, 0);
        let r: u16x8 = transmute(vclezq_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclez_s32() {
        let a: i32x2 = i32x2::new(-2147483648, -1);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vclez_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclezq_s32() {
        let a: i32x4 = i32x4::new(-2147483648, -1, 0x00, 0x01);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0);
        let r: u32x4 = transmute(vclezq_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclez_s64() {
        let a: i64x1 = i64x1::new(-9223372036854775808);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vclez_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclezq_s64() {
        let a: i64x2 = i64x2::new(-9223372036854775808, -1);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vclezq_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclez_f32() {
        let a: f32x2 = f32x2::new(-1.2, 0.0);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vclez_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclezq_f32() {
        let a: f32x4 = f32x4::new(-1.2, 0.0, 1.2, 2.3);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0, 0);
        let r: u32x4 = transmute(vclezq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclez_f64() {
        let a: f64 = -1.2;
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vclez_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclezq_f64() {
        let a: f64x2 = f64x2::new(-1.2, 0.0);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vclezq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltz_s8() {
        let a: i8x8 = i8x8::new(-128, -1, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0, 0, 0, 0, 0, 0);
        let r: u8x8 = transmute(vcltz_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltzq_s8() {
        let a: i8x16 = i8x16::new(-128, -1, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x7F);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x16 = transmute(vcltzq_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltz_s16() {
        let a: i16x4 = i16x4::new(-32768, -1, 0x00, 0x01);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0, 0);
        let r: u16x4 = transmute(vcltz_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltzq_s16() {
        let a: i16x8 = i16x8::new(-32768, -1, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0, 0, 0, 0, 0, 0);
        let r: u16x8 = transmute(vcltzq_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltz_s32() {
        let a: i32x2 = i32x2::new(-2147483648, -1);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcltz_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltzq_s32() {
        let a: i32x4 = i32x4::new(-2147483648, -1, 0x00, 0x01);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0, 0);
        let r: u32x4 = transmute(vcltzq_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltz_s64() {
        let a: i64x1 = i64x1::new(-9223372036854775808);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcltz_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltzq_s64() {
        let a: i64x2 = i64x2::new(-9223372036854775808, -1);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcltzq_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltz_f32() {
        let a: f32x2 = f32x2::new(-1.2, 0.0);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0);
        let r: u32x2 = transmute(vcltz_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltzq_f32() {
        let a: f32x4 = f32x4::new(-1.2, 0.0, 1.2, 2.3);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0, 0, 0);
        let r: u32x4 = transmute(vcltzq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltz_f64() {
        let a: f64 = -1.2;
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcltz_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltzq_f64() {
        let a: f64x2 = f64x2::new(-1.2, 0.0);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0);
        let r: u64x2 = transmute(vcltzq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcagt_f64() {
        let a: f64 = -1.2;
        let b: f64 = -1.1;
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcagt_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcagtq_f64() {
        let a: f64x2 = f64x2::new(-1.2, 0.0);
        let b: f64x2 = f64x2::new(-1.1, 0.0);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0);
        let r: u64x2 = transmute(vcagtq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcage_f64() {
        let a: f64 = -1.2;
        let b: f64 = -1.1;
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcage_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcageq_f64() {
        let a: f64x2 = f64x2::new(-1.2, 0.0);
        let b: f64x2 = f64x2::new(-1.1, 0.0);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcageq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcalt_f64() {
        let a: f64 = -1.2;
        let b: f64 = -1.1;
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vcalt_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcaltq_f64() {
        let a: f64x2 = f64x2::new(-1.2, 0.0);
        let b: f64x2 = f64x2::new(-1.1, 0.0);
        let e: u64x2 = u64x2::new(0, 0);
        let r: u64x2 = transmute(vcaltq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcale_f64() {
        let a: f64 = -1.2;
        let b: f64 = -1.1;
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vcale_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcaleq_f64() {
        let a: f64x2 = f64x2::new(-1.2, 0.0);
        let b: f64x2 = f64x2::new(-1.1, 0.0);
        let e: u64x2 = u64x2::new(0, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcaleq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_lane_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(0, 0x7F, 0, 0, 0, 0, 0, 0);
        let e: i8x8 = i8x8::new(0x7F, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vcopy_lane_s8::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_laneq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(0, 0x7F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let e: i8x16 = i8x16::new(0x7F, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: i8x16 = transmute(vcopyq_laneq_s8::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_lane_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(0, 0x7F_FF, 0, 0);
        let e: i16x4 = i16x4::new(0x7F_FF, 2, 3, 4);
        let r: i16x4 = transmute(vcopy_lane_s16::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_laneq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(0, 0x7F_FF, 0, 0, 0, 0, 0, 0);
        let e: i16x8 = i16x8::new(0x7F_FF, 2, 3, 4, 5, 6, 7, 8);
        let r: i16x8 = transmute(vcopyq_laneq_s16::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_lane_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(0, 0x7F_FF_FF_FF);
        let e: i32x2 = i32x2::new(0x7F_FF_FF_FF, 2);
        let r: i32x2 = transmute(vcopy_lane_s32::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_laneq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(0, 0x7F_FF_FF_FF, 0, 0);
        let e: i32x4 = i32x4::new(0x7F_FF_FF_FF, 2, 3, 4);
        let r: i32x4 = transmute(vcopyq_laneq_s32::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_laneq_s64() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i64x2 = i64x2::new(0, 0x7F_FF_FF_FF_FF_FF_FF_FF);
        let e: i64x2 = i64x2::new(0x7F_FF_FF_FF_FF_FF_FF_FF, 2);
        let r: i64x2 = transmute(vcopyq_laneq_s64::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_lane_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(0, 0xFF, 0, 0, 0, 0, 0, 0);
        let e: u8x8 = u8x8::new(0xFF, 2, 3, 4, 5, 6, 7, 8);
        let r: u8x8 = transmute(vcopy_lane_u8::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_laneq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(0, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let e: u8x16 = u8x16::new(0xFF, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: u8x16 = transmute(vcopyq_laneq_u8::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_lane_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(0, 0xFF_FF, 0, 0);
        let e: u16x4 = u16x4::new(0xFF_FF, 2, 3, 4);
        let r: u16x4 = transmute(vcopy_lane_u16::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_laneq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(0, 0xFF_FF, 0, 0, 0, 0, 0, 0);
        let e: u16x8 = u16x8::new(0xFF_FF, 2, 3, 4, 5, 6, 7, 8);
        let r: u16x8 = transmute(vcopyq_laneq_u16::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_lane_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(0, 0xFF_FF_FF_FF);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 2);
        let r: u32x2 = transmute(vcopy_lane_u32::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_laneq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(0, 0xFF_FF_FF_FF, 0, 0);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 2, 3, 4);
        let r: u32x4 = transmute(vcopyq_laneq_u32::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_laneq_u64() {
        let a: u64x2 = u64x2::new(1, 2);
        let b: u64x2 = u64x2::new(0, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let e: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 2);
        let r: u64x2 = transmute(vcopyq_laneq_u64::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_lane_p8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(0, 0x7F, 0, 0, 0, 0, 0, 0);
        let e: i8x8 = i8x8::new(0x7F, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vcopy_lane_p8::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_laneq_p8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(0, 0x7F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let e: i8x16 = i8x16::new(0x7F, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: i8x16 = transmute(vcopyq_laneq_p8::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_lane_p16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(0, 0x7F_FF, 0, 0);
        let e: i16x4 = i16x4::new(0x7F_FF, 2, 3, 4);
        let r: i16x4 = transmute(vcopy_lane_p16::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_laneq_p16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(0, 0x7F_FF, 0, 0, 0, 0, 0, 0);
        let e: i16x8 = i16x8::new(0x7F_FF, 2, 3, 4, 5, 6, 7, 8);
        let r: i16x8 = transmute(vcopyq_laneq_p16::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_laneq_p64() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i64x2 = i64x2::new(0, 0x7F_FF_FF_FF_FF_FF_FF_FF);
        let e: i64x2 = i64x2::new(0x7F_FF_FF_FF_FF_FF_FF_FF, 2);
        let r: i64x2 = transmute(vcopyq_laneq_p64::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_lane_f32() {
        let a: f32x2 = f32x2::new(1., 2.);
        let b: f32x2 = f32x2::new(0., 0.5);
        let e: f32x2 = f32x2::new(0.5, 2.);
        let r: f32x2 = transmute(vcopy_lane_f32::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_laneq_f32() {
        let a: f32x4 = f32x4::new(1., 2., 3., 4.);
        let b: f32x4 = f32x4::new(0., 0.5, 0., 0.);
        let e: f32x4 = f32x4::new(0.5, 2., 3., 4.);
        let r: f32x4 = transmute(vcopyq_laneq_f32::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_laneq_f64() {
        let a: f64x2 = f64x2::new(1., 2.);
        let b: f64x2 = f64x2::new(0., 0.5);
        let e: f64x2 = f64x2::new(0.5, 2.);
        let r: f64x2 = transmute(vcopyq_laneq_f64::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_laneq_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x16 = i8x16::new(0, 0x7F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let e: i8x8 = i8x8::new(0x7F, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vcopy_laneq_s8::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_laneq_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x8 = i16x8::new(0, 0x7F_FF, 0, 0, 0, 0, 0, 0);
        let e: i16x4 = i16x4::new(0x7F_FF, 2, 3, 4);
        let r: i16x4 = transmute(vcopy_laneq_s16::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_laneq_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x4 = i32x4::new(0, 0x7F_FF_FF_FF, 0, 0);
        let e: i32x2 = i32x2::new(0x7F_FF_FF_FF, 2);
        let r: i32x2 = transmute(vcopy_laneq_s32::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_laneq_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x16 = u8x16::new(0, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let e: u8x8 = u8x8::new(0xFF, 2, 3, 4, 5, 6, 7, 8);
        let r: u8x8 = transmute(vcopy_laneq_u8::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_laneq_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x8 = u16x8::new(0, 0xFF_FF, 0, 0, 0, 0, 0, 0);
        let e: u16x4 = u16x4::new(0xFF_FF, 2, 3, 4);
        let r: u16x4 = transmute(vcopy_laneq_u16::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_laneq_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x4 = u32x4::new(0, 0xFF_FF_FF_FF, 0, 0);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 2);
        let r: u32x2 = transmute(vcopy_laneq_u32::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_laneq_p8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x16 = i8x16::new(0, 0x7F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let e: i8x8 = i8x8::new(0x7F, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vcopy_laneq_p8::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_laneq_p16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x8 = i16x8::new(0, 0x7F_FF, 0, 0, 0, 0, 0, 0);
        let e: i16x4 = i16x4::new(0x7F_FF, 2, 3, 4);
        let r: i16x4 = transmute(vcopy_laneq_p16::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_laneq_f32() {
        let a: f32x2 = f32x2::new(1., 2.);
        let b: f32x4 = f32x4::new(0., 0.5, 0., 0.);
        let e: f32x2 = f32x2::new(0.5, 2.);
        let r: f32x2 = transmute(vcopy_laneq_f32::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_lane_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x8 = i8x8::new(0, 0x7F, 0, 0, 0, 0, 0, 0);
        let e: i8x16 = i8x16::new(0x7F, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: i8x16 = transmute(vcopyq_lane_s8::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_lane_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x4 = i16x4::new(0, 0x7F_FF, 0, 0);
        let e: i16x8 = i16x8::new(0x7F_FF, 2, 3, 4, 5, 6, 7, 8);
        let r: i16x8 = transmute(vcopyq_lane_s16::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_lane_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x2 = i32x2::new(0, 0x7F_FF_FF_FF);
        let e: i32x4 = i32x4::new(0x7F_FF_FF_FF, 2, 3, 4);
        let r: i32x4 = transmute(vcopyq_lane_s32::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_lane_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x8 = u8x8::new(0, 0xFF, 0, 0, 0, 0, 0, 0);
        let e: u8x16 = u8x16::new(0xFF, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: u8x16 = transmute(vcopyq_lane_u8::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_lane_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x4 = u16x4::new(0, 0xFF_FF, 0, 0);
        let e: u16x8 = u16x8::new(0xFF_FF, 2, 3, 4, 5, 6, 7, 8);
        let r: u16x8 = transmute(vcopyq_lane_u16::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_lane_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x2 = u32x2::new(0, 0xFF_FF_FF_FF);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 2, 3, 4);
        let r: u32x4 = transmute(vcopyq_lane_u32::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_lane_p8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x8 = i8x8::new(0, 0x7F, 0, 0, 0, 0, 0, 0);
        let e: i8x16 = i8x16::new(0x7F, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: i8x16 = transmute(vcopyq_lane_p8::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_lane_p16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x4 = i16x4::new(0, 0x7F_FF, 0, 0);
        let e: i16x8 = i16x8::new(0x7F_FF, 2, 3, 4, 5, 6, 7, 8);
        let r: i16x8 = transmute(vcopyq_lane_p16::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_lane_s64() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i64x1 = i64x1::new(0x7F_FF_FF_FF_FF_FF_FF_FF);
        let e: i64x2 = i64x2::new(1, 0x7F_FF_FF_FF_FF_FF_FF_FF);
        let r: i64x2 = transmute(vcopyq_lane_s64::<1, 0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_lane_u64() {
        let a: u64x2 = u64x2::new(1, 2);
        let b: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let e: u64x2 = u64x2::new(1, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x2 = transmute(vcopyq_lane_u64::<1, 0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_lane_p64() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i64x1 = i64x1::new(0x7F_FF_FF_FF_FF_FF_FF_FF);
        let e: i64x2 = i64x2::new(1, 0x7F_FF_FF_FF_FF_FF_FF_FF);
        let r: i64x2 = transmute(vcopyq_lane_p64::<1, 0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_lane_f32() {
        let a: f32x4 = f32x4::new(1., 2., 3., 4.);
        let b: f32x2 = f32x2::new(0.5, 0.);
        let e: f32x4 = f32x4::new(1., 0.5, 3., 4.);
        let r: f32x4 = transmute(vcopyq_lane_f32::<1, 0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopyq_lane_f64() {
        let a: f64x2 = f64x2::new(1., 2.);
        let b: f64 = 0.5;
        let e: f64x2 = f64x2::new(1., 0.5);
        let r: f64x2 = transmute(vcopyq_lane_f64::<1, 0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcreate_f64() {
        let a: u64 = 0;
        let e: f64 = 0.;
        let r: f64 = transmute(vcreate_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_f64_s64() {
        let a: i64x1 = i64x1::new(1);
        let e: f64 = 1.;
        let r: f64 = transmute(vcvt_f64_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_f64_s64() {
        let a: i64x2 = i64x2::new(1, 2);
        let e: f64x2 = f64x2::new(1., 2.);
        let r: f64x2 = transmute(vcvtq_f64_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_f64_u64() {
        let a: u64x1 = u64x1::new(1);
        let e: f64 = 1.;
        let r: f64 = transmute(vcvt_f64_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_f64_u64() {
        let a: u64x2 = u64x2::new(1, 2);
        let e: f64x2 = f64x2::new(1., 2.);
        let r: f64x2 = transmute(vcvtq_f64_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_f64_f32() {
        let a: f32x2 = f32x2::new(-1.2, 1.2);
        let e: f64x2 = f64x2::new(-1.2f32 as f64, 1.2f32 as f64);
        let r: f64x2 = transmute(vcvt_f64_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_high_f64_f32() {
        let a: f32x4 = f32x4::new(-1.2, 1.2, 2.3, 3.4);
        let e: f64x2 = f64x2::new(2.3f32 as f64, 3.4f32 as f64);
        let r: f64x2 = transmute(vcvt_high_f64_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_f32_f64() {
        let a: f64x2 = f64x2::new(-1.2, 1.2);
        let e: f32x2 = f32x2::new(-1.2f64 as f32, 1.2f64 as f32);
        let r: f32x2 = transmute(vcvt_f32_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_high_f32_f64() {
        let a: f32x2 = f32x2::new(-1.2, 1.2);
        let b: f64x2 = f64x2::new(-2.3, 3.4);
        let e: f32x4 = f32x4::new(-1.2, 1.2, -2.3f64 as f32, 3.4f64 as f32);
        let r: f32x4 = transmute(vcvt_high_f32_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtx_f32_f64() {
        let a: f64x2 = f64x2::new(-1.0, 2.0);
        let e: f32x2 = f32x2::new(-1.0, 2.0);
        let r: f32x2 = transmute(vcvtx_f32_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtx_high_f32_f64() {
        let a: f32x2 = f32x2::new(-1.0, 2.0);
        let b: f64x2 = f64x2::new(-3.0, 4.0);
        let e: f32x4 = f32x4::new(-1.0, 2.0, -3.0, 4.0);
        let r: f32x4 = transmute(vcvtx_high_f32_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_n_f64_s64() {
        let a: i64x1 = i64x1::new(1);
        let e: f64 = 0.25;
        let r: f64 = transmute(vcvt_n_f64_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_n_f64_s64() {
        let a: i64x2 = i64x2::new(1, 2);
        let e: f64x2 = f64x2::new(0.25, 0.5);
        let r: f64x2 = transmute(vcvtq_n_f64_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvts_n_f32_s32() {
        let a: i32 = 1;
        let e: f32 = 0.25;
        let r: f32 = transmute(vcvts_n_f32_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtd_n_f64_s64() {
        let a: i64 = 1;
        let e: f64 = 0.25;
        let r: f64 = transmute(vcvtd_n_f64_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_n_f64_u64() {
        let a: u64x1 = u64x1::new(1);
        let e: f64 = 0.25;
        let r: f64 = transmute(vcvt_n_f64_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_n_f64_u64() {
        let a: u64x2 = u64x2::new(1, 2);
        let e: f64x2 = f64x2::new(0.25, 0.5);
        let r: f64x2 = transmute(vcvtq_n_f64_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvts_n_f32_u32() {
        let a: u32 = 1;
        let e: f32 = 0.25;
        let r: f32 = transmute(vcvts_n_f32_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtd_n_f64_u64() {
        let a: u64 = 1;
        let e: f64 = 0.25;
        let r: f64 = transmute(vcvtd_n_f64_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_n_s64_f64() {
        let a: f64 = 0.25;
        let e: i64x1 = i64x1::new(1);
        let r: i64x1 = transmute(vcvt_n_s64_f64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_n_s64_f64() {
        let a: f64x2 = f64x2::new(0.25, 0.5);
        let e: i64x2 = i64x2::new(1, 2);
        let r: i64x2 = transmute(vcvtq_n_s64_f64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvts_n_s32_f32() {
        let a: f32 = 0.25;
        let e: i32 = 1;
        let r: i32 = transmute(vcvts_n_s32_f32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtd_n_s64_f64() {
        let a: f64 = 0.25;
        let e: i64 = 1;
        let r: i64 = transmute(vcvtd_n_s64_f64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_n_u64_f64() {
        let a: f64 = 0.25;
        let e: u64x1 = u64x1::new(1);
        let r: u64x1 = transmute(vcvt_n_u64_f64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_n_u64_f64() {
        let a: f64x2 = f64x2::new(0.25, 0.5);
        let e: u64x2 = u64x2::new(1, 2);
        let r: u64x2 = transmute(vcvtq_n_u64_f64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvts_n_u32_f32() {
        let a: f32 = 0.25;
        let e: u32 = 1;
        let r: u32 = transmute(vcvts_n_u32_f32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtd_n_u64_f64() {
        let a: f64 = 0.25;
        let e: u64 = 1;
        let r: u64 = transmute(vcvtd_n_u64_f64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvts_f32_s32() {
        let a: i32 = 1;
        let e: f32 = 1.;
        let r: f32 = transmute(vcvts_f32_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtd_f64_s64() {
        let a: i64 = 1;
        let e: f64 = 1.;
        let r: f64 = transmute(vcvtd_f64_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvts_f32_u32() {
        let a: u32 = 1;
        let e: f32 = 1.;
        let r: f32 = transmute(vcvts_f32_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtd_f64_u64() {
        let a: u64 = 1;
        let e: f64 = 1.;
        let r: f64 = transmute(vcvtd_f64_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvts_s32_f32() {
        let a: f32 = 1.;
        let e: i32 = 1;
        let r: i32 = transmute(vcvts_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtd_s64_f64() {
        let a: f64 = 1.;
        let e: i64 = 1;
        let r: i64 = transmute(vcvtd_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvts_u32_f32() {
        let a: f32 = 1.;
        let e: u32 = 1;
        let r: u32 = transmute(vcvts_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtd_u64_f64() {
        let a: f64 = 1.;
        let e: u64 = 1;
        let r: u64 = transmute(vcvtd_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_s64_f64() {
        let a: f64 = -1.1;
        let e: i64x1 = i64x1::new(-1);
        let r: i64x1 = transmute(vcvt_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_s64_f64() {
        let a: f64x2 = f64x2::new(-1.1, 2.1);
        let e: i64x2 = i64x2::new(-1, 2);
        let r: i64x2 = transmute(vcvtq_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_u64_f64() {
        let a: f64 = 1.1;
        let e: u64x1 = u64x1::new(1);
        let r: u64x1 = transmute(vcvt_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_u64_f64() {
        let a: f64x2 = f64x2::new(1.1, 2.1);
        let e: u64x2 = u64x2::new(1, 2);
        let r: u64x2 = transmute(vcvtq_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvta_s32_f32() {
        let a: f32x2 = f32x2::new(-1.1, 2.1);
        let e: i32x2 = i32x2::new(-1, 2);
        let r: i32x2 = transmute(vcvta_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtaq_s32_f32() {
        let a: f32x4 = f32x4::new(-1.1, 2.1, -2.9, 3.9);
        let e: i32x4 = i32x4::new(-1, 2, -3, 4);
        let r: i32x4 = transmute(vcvtaq_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvta_s64_f64() {
        let a: f64 = -1.1;
        let e: i64x1 = i64x1::new(-1);
        let r: i64x1 = transmute(vcvta_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtaq_s64_f64() {
        let a: f64x2 = f64x2::new(-1.1, 2.1);
        let e: i64x2 = i64x2::new(-1, 2);
        let r: i64x2 = transmute(vcvtaq_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtas_s32_f32() {
        let a: f32 = 2.9;
        let e: i32 = 3;
        let r: i32 = transmute(vcvtas_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtad_s64_f64() {
        let a: f64 = 2.9;
        let e: i64 = 3;
        let r: i64 = transmute(vcvtad_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtas_u32_f32() {
        let a: f32 = 2.9;
        let e: u32 = 3;
        let r: u32 = transmute(vcvtas_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtad_u64_f64() {
        let a: f64 = 2.9;
        let e: u64 = 3;
        let r: u64 = transmute(vcvtad_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtn_s32_f32() {
        let a: f32x2 = f32x2::new(-1.5, 2.1);
        let e: i32x2 = i32x2::new(-2, 2);
        let r: i32x2 = transmute(vcvtn_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtnq_s32_f32() {
        let a: f32x4 = f32x4::new(-1.5, 2.1, -2.9, 3.9);
        let e: i32x4 = i32x4::new(-2, 2, -3, 4);
        let r: i32x4 = transmute(vcvtnq_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtn_s64_f64() {
        let a: f64 = -1.5;
        let e: i64x1 = i64x1::new(-2);
        let r: i64x1 = transmute(vcvtn_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtnq_s64_f64() {
        let a: f64x2 = f64x2::new(-1.5, 2.1);
        let e: i64x2 = i64x2::new(-2, 2);
        let r: i64x2 = transmute(vcvtnq_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtns_s32_f32() {
        let a: f32 = -1.5;
        let e: i32 = -2;
        let r: i32 = transmute(vcvtns_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtnd_s64_f64() {
        let a: f64 = -1.5;
        let e: i64 = -2;
        let r: i64 = transmute(vcvtnd_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtm_s32_f32() {
        let a: f32x2 = f32x2::new(-1.1, 2.1);
        let e: i32x2 = i32x2::new(-2, 2);
        let r: i32x2 = transmute(vcvtm_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtmq_s32_f32() {
        let a: f32x4 = f32x4::new(-1.1, 2.1, -2.9, 3.9);
        let e: i32x4 = i32x4::new(-2, 2, -3, 3);
        let r: i32x4 = transmute(vcvtmq_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtm_s64_f64() {
        let a: f64 = -1.1;
        let e: i64x1 = i64x1::new(-2);
        let r: i64x1 = transmute(vcvtm_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtmq_s64_f64() {
        let a: f64x2 = f64x2::new(-1.1, 2.1);
        let e: i64x2 = i64x2::new(-2, 2);
        let r: i64x2 = transmute(vcvtmq_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtms_s32_f32() {
        let a: f32 = -1.1;
        let e: i32 = -2;
        let r: i32 = transmute(vcvtms_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtmd_s64_f64() {
        let a: f64 = -1.1;
        let e: i64 = -2;
        let r: i64 = transmute(vcvtmd_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtp_s32_f32() {
        let a: f32x2 = f32x2::new(-1.1, 2.1);
        let e: i32x2 = i32x2::new(-1, 3);
        let r: i32x2 = transmute(vcvtp_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtpq_s32_f32() {
        let a: f32x4 = f32x4::new(-1.1, 2.1, -2.9, 3.9);
        let e: i32x4 = i32x4::new(-1, 3, -2, 4);
        let r: i32x4 = transmute(vcvtpq_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtp_s64_f64() {
        let a: f64 = -1.1;
        let e: i64x1 = i64x1::new(-1);
        let r: i64x1 = transmute(vcvtp_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtpq_s64_f64() {
        let a: f64x2 = f64x2::new(-1.1, 2.1);
        let e: i64x2 = i64x2::new(-1, 3);
        let r: i64x2 = transmute(vcvtpq_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtps_s32_f32() {
        let a: f32 = -1.1;
        let e: i32 = -1;
        let r: i32 = transmute(vcvtps_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtpd_s64_f64() {
        let a: f64 = -1.1;
        let e: i64 = -1;
        let r: i64 = transmute(vcvtpd_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvta_u32_f32() {
        let a: f32x2 = f32x2::new(1.1, 2.1);
        let e: u32x2 = u32x2::new(1, 2);
        let r: u32x2 = transmute(vcvta_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtaq_u32_f32() {
        let a: f32x4 = f32x4::new(1.1, 2.1, 2.9, 3.9);
        let e: u32x4 = u32x4::new(1, 2, 3, 4);
        let r: u32x4 = transmute(vcvtaq_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvta_u64_f64() {
        let a: f64 = 1.1;
        let e: u64x1 = u64x1::new(1);
        let r: u64x1 = transmute(vcvta_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtaq_u64_f64() {
        let a: f64x2 = f64x2::new(1.1, 2.1);
        let e: u64x2 = u64x2::new(1, 2);
        let r: u64x2 = transmute(vcvtaq_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtn_u32_f32() {
        let a: f32x2 = f32x2::new(1.5, 2.1);
        let e: u32x2 = u32x2::new(2, 2);
        let r: u32x2 = transmute(vcvtn_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtnq_u32_f32() {
        let a: f32x4 = f32x4::new(1.5, 2.1, 2.9, 3.9);
        let e: u32x4 = u32x4::new(2, 2, 3, 4);
        let r: u32x4 = transmute(vcvtnq_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtn_u64_f64() {
        let a: f64 = 1.5;
        let e: u64x1 = u64x1::new(2);
        let r: u64x1 = transmute(vcvtn_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtnq_u64_f64() {
        let a: f64x2 = f64x2::new(1.5, 2.1);
        let e: u64x2 = u64x2::new(2, 2);
        let r: u64x2 = transmute(vcvtnq_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtns_u32_f32() {
        let a: f32 = 1.5;
        let e: u32 = 2;
        let r: u32 = transmute(vcvtns_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtnd_u64_f64() {
        let a: f64 = 1.5;
        let e: u64 = 2;
        let r: u64 = transmute(vcvtnd_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtm_u32_f32() {
        let a: f32x2 = f32x2::new(1.1, 2.1);
        let e: u32x2 = u32x2::new(1, 2);
        let r: u32x2 = transmute(vcvtm_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtmq_u32_f32() {
        let a: f32x4 = f32x4::new(1.1, 2.1, 2.9, 3.9);
        let e: u32x4 = u32x4::new(1, 2, 2, 3);
        let r: u32x4 = transmute(vcvtmq_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtm_u64_f64() {
        let a: f64 = 1.1;
        let e: u64x1 = u64x1::new(1);
        let r: u64x1 = transmute(vcvtm_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtmq_u64_f64() {
        let a: f64x2 = f64x2::new(1.1, 2.1);
        let e: u64x2 = u64x2::new(1, 2);
        let r: u64x2 = transmute(vcvtmq_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtms_u32_f32() {
        let a: f32 = 1.1;
        let e: u32 = 1;
        let r: u32 = transmute(vcvtms_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtmd_u64_f64() {
        let a: f64 = 1.1;
        let e: u64 = 1;
        let r: u64 = transmute(vcvtmd_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtp_u32_f32() {
        let a: f32x2 = f32x2::new(1.1, 2.1);
        let e: u32x2 = u32x2::new(2, 3);
        let r: u32x2 = transmute(vcvtp_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtpq_u32_f32() {
        let a: f32x4 = f32x4::new(1.1, 2.1, 2.9, 3.9);
        let e: u32x4 = u32x4::new(2, 3, 3, 4);
        let r: u32x4 = transmute(vcvtpq_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtp_u64_f64() {
        let a: f64 = 1.1;
        let e: u64x1 = u64x1::new(2);
        let r: u64x1 = transmute(vcvtp_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtpq_u64_f64() {
        let a: f64x2 = f64x2::new(1.1, 2.1);
        let e: u64x2 = u64x2::new(2, 3);
        let r: u64x2 = transmute(vcvtpq_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtps_u32_f32() {
        let a: f32 = 1.1;
        let e: u32 = 2;
        let r: u32 = transmute(vcvtps_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtpd_u64_f64() {
        let a: f64 = 1.1;
        let e: u64 = 2;
        let r: u64 = transmute(vcvtpd_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_laneq_p64() {
        let a: i64x2 = i64x2::new(1, 1);
        let e: i64x2 = i64x2::new(1, 1);
        let r: i64x2 = transmute(vdupq_laneq_p64::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_lane_p64() {
        let a: i64x1 = i64x1::new(1);
        let e: i64x2 = i64x2::new(1, 1);
        let r: i64x2 = transmute(vdupq_lane_p64::<0>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_laneq_f64() {
        let a: f64x2 = f64x2::new(1., 1.);
        let e: f64x2 = f64x2::new(1., 1.);
        let r: f64x2 = transmute(vdupq_laneq_f64::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_lane_f64() {
        let a: f64 = 1.;
        let e: f64x2 = f64x2::new(1., 1.);
        let r: f64x2 = transmute(vdupq_lane_f64::<0>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_lane_p64() {
        let a: i64x1 = i64x1::new(0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vdup_lane_p64::<0>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_lane_f64() {
        let a: f64 = 0.;
        let e: f64 = 0.;
        let r: f64 = transmute(vdup_lane_f64::<0>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_laneq_p64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i64x1 = i64x1::new(1);
        let r: i64x1 = transmute(vdup_laneq_p64::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_laneq_f64() {
        let a: f64x2 = f64x2::new(0., 1.);
        let e: f64 = 1.;
        let r: f64 = transmute(vdup_laneq_f64::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupb_lane_s8() {
        let a: i8x8 = i8x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: i8 = 1;
        let r: i8 = transmute(vdupb_lane_s8::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupb_laneq_s8() {
        let a: i8x16 = i8x16::new(1, 1, 1, 4, 1, 6, 7, 8, 1, 10, 11, 12, 13, 14, 15, 16);
        let e: i8 = 1;
        let r: i8 = transmute(vdupb_laneq_s8::<8>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vduph_lane_s16() {
        let a: i16x4 = i16x4::new(1, 1, 1, 4);
        let e: i16 = 1;
        let r: i16 = transmute(vduph_lane_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vduph_laneq_s16() {
        let a: i16x8 = i16x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: i16 = 1;
        let r: i16 = transmute(vduph_laneq_s16::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdups_lane_s32() {
        let a: i32x2 = i32x2::new(1, 1);
        let e: i32 = 1;
        let r: i32 = transmute(vdups_lane_s32::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdups_laneq_s32() {
        let a: i32x4 = i32x4::new(1, 1, 1, 4);
        let e: i32 = 1;
        let r: i32 = transmute(vdups_laneq_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupd_lane_s64() {
        let a: i64x1 = i64x1::new(1);
        let e: i64 = 1;
        let r: i64 = transmute(vdupd_lane_s64::<0>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupd_laneq_s64() {
        let a: i64x2 = i64x2::new(1, 1);
        let e: i64 = 1;
        let r: i64 = transmute(vdupd_laneq_s64::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupb_lane_u8() {
        let a: u8x8 = u8x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: u8 = 1;
        let r: u8 = transmute(vdupb_lane_u8::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupb_laneq_u8() {
        let a: u8x16 = u8x16::new(1, 1, 1, 4, 1, 6, 7, 8, 1, 10, 11, 12, 13, 14, 15, 16);
        let e: u8 = 1;
        let r: u8 = transmute(vdupb_laneq_u8::<8>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vduph_lane_u16() {
        let a: u16x4 = u16x4::new(1, 1, 1, 4);
        let e: u16 = 1;
        let r: u16 = transmute(vduph_lane_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vduph_laneq_u16() {
        let a: u16x8 = u16x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: u16 = 1;
        let r: u16 = transmute(vduph_laneq_u16::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdups_lane_u32() {
        let a: u32x2 = u32x2::new(1, 1);
        let e: u32 = 1;
        let r: u32 = transmute(vdups_lane_u32::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdups_laneq_u32() {
        let a: u32x4 = u32x4::new(1, 1, 1, 4);
        let e: u32 = 1;
        let r: u32 = transmute(vdups_laneq_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupd_lane_u64() {
        let a: u64x1 = u64x1::new(1);
        let e: u64 = 1;
        let r: u64 = transmute(vdupd_lane_u64::<0>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupd_laneq_u64() {
        let a: u64x2 = u64x2::new(1, 1);
        let e: u64 = 1;
        let r: u64 = transmute(vdupd_laneq_u64::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupb_lane_p8() {
        let a: i8x8 = i8x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: p8 = 1;
        let r: p8 = transmute(vdupb_lane_p8::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupb_laneq_p8() {
        let a: i8x16 = i8x16::new(1, 1, 1, 4, 1, 6, 7, 8, 1, 10, 11, 12, 13, 14, 15, 16);
        let e: p8 = 1;
        let r: p8 = transmute(vdupb_laneq_p8::<8>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vduph_lane_p16() {
        let a: i16x4 = i16x4::new(1, 1, 1, 4);
        let e: p16 = 1;
        let r: p16 = transmute(vduph_lane_p16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vduph_laneq_p16() {
        let a: i16x8 = i16x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: p16 = 1;
        let r: p16 = transmute(vduph_laneq_p16::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdups_lane_f32() {
        let a: f32x2 = f32x2::new(1., 1.);
        let e: f32 = 1.;
        let r: f32 = transmute(vdups_lane_f32::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdups_laneq_f32() {
        let a: f32x4 = f32x4::new(1., 1., 1., 4.);
        let e: f32 = 1.;
        let r: f32 = transmute(vdups_laneq_f32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupd_lane_f64() {
        let a: f64 = 1.;
        let e: f64 = 1.;
        let r: f64 = transmute(vdupd_lane_f64::<0>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupd_laneq_f64() {
        let a: f64x2 = f64x2::new(1., 1.);
        let e: f64 = 1.;
        let r: f64 = transmute(vdupd_laneq_f64::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_p64() {
        let a: i64x2 = i64x2::new(0, 8);
        let b: i64x2 = i64x2::new(9, 11);
        let e: i64x2 = i64x2::new(8, 9);
        let r: i64x2 = transmute(vextq_p64::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_f64() {
        let a: f64x2 = f64x2::new(0., 2.);
        let b: f64x2 = f64x2::new(3., 4.);
        let e: f64x2 = f64x2::new(2., 3.);
        let r: f64x2 = transmute(vextq_f64::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_f64() {
        let a: f64 = 0.;
        let b: f64 = 2.;
        let c: f64 = 3.;
        let e: f64 = 6.;
        let r: f64 = transmute(vmla_f64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_f64() {
        let a: f64x2 = f64x2::new(0., 1.);
        let b: f64x2 = f64x2::new(2., 2.);
        let c: f64x2 = f64x2::new(3., 3.);
        let e: f64x2 = f64x2::new(6., 7.);
        let r: f64x2 = transmute(vmlaq_f64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_s8() {
        let a: i16x8 = i16x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let b: i8x16 = i8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let c: i8x16 = i8x16::new(3, 3, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7);
        let e: i16x8 = i16x8::new(8, 9, 10, 11, 12, 13, 14, 15);
        let r: i16x8 = transmute(vmlal_high_s8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_s16() {
        let a: i32x4 = i32x4::new(8, 7, 6, 5);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: i16x8 = i16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let e: i32x4 = i32x4::new(8, 9, 10, 11);
        let r: i32x4 = transmute(vmlal_high_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_s32() {
        let a: i64x2 = i64x2::new(8, 7);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let c: i32x4 = i32x4::new(3, 3, 0, 1);
        let e: i64x2 = i64x2::new(8, 9);
        let r: i64x2 = transmute(vmlal_high_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_u8() {
        let a: u16x8 = u16x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let b: u8x16 = u8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let c: u8x16 = u8x16::new(3, 3, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(8, 9, 10, 11, 12, 13, 14, 15);
        let r: u16x8 = transmute(vmlal_high_u8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_u16() {
        let a: u32x4 = u32x4::new(8, 7, 6, 5);
        let b: u16x8 = u16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: u16x8 = u16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let e: u32x4 = u32x4::new(8, 9, 10, 11);
        let r: u32x4 = transmute(vmlal_high_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_u32() {
        let a: u64x2 = u64x2::new(8, 7);
        let b: u32x4 = u32x4::new(2, 2, 2, 2);
        let c: u32x4 = u32x4::new(3, 3, 0, 1);
        let e: u64x2 = u64x2::new(8, 9);
        let r: u64x2 = transmute(vmlal_high_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_n_s16() {
        let a: i32x4 = i32x4::new(8, 7, 6, 5);
        let b: i16x8 = i16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let c: i16 = 2;
        let e: i32x4 = i32x4::new(8, 9, 10, 11);
        let r: i32x4 = transmute(vmlal_high_n_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_n_s32() {
        let a: i64x2 = i64x2::new(8, 7);
        let b: i32x4 = i32x4::new(3, 3, 0, 1);
        let c: i32 = 2;
        let e: i64x2 = i64x2::new(8, 9);
        let r: i64x2 = transmute(vmlal_high_n_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_n_u16() {
        let a: u32x4 = u32x4::new(8, 7, 6, 5);
        let b: u16x8 = u16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let c: u16 = 2;
        let e: u32x4 = u32x4::new(8, 9, 10, 11);
        let r: u32x4 = transmute(vmlal_high_n_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_n_u32() {
        let a: u64x2 = u64x2::new(8, 7);
        let b: u32x4 = u32x4::new(3, 3, 0, 1);
        let c: u32 = 2;
        let e: u64x2 = u64x2::new(8, 9);
        let r: u64x2 = transmute(vmlal_high_n_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_lane_s16() {
        let a: i32x4 = i32x4::new(8, 7, 6, 5);
        let b: i16x8 = i16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let c: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i32x4 = i32x4::new(8, 9, 10, 11);
        let r: i32x4 = transmute(vmlal_high_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_laneq_s16() {
        let a: i32x4 = i32x4::new(8, 7, 6, 5);
        let b: i16x8 = i16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let c: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i32x4 = i32x4::new(8, 9, 10, 11);
        let r: i32x4 = transmute(vmlal_high_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_lane_s32() {
        let a: i64x2 = i64x2::new(8, 7);
        let b: i32x4 = i32x4::new(3, 3, 0, 1);
        let c: i32x2 = i32x2::new(0, 2);
        let e: i64x2 = i64x2::new(8, 9);
        let r: i64x2 = transmute(vmlal_high_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_laneq_s32() {
        let a: i64x2 = i64x2::new(8, 7);
        let b: i32x4 = i32x4::new(3, 3, 0, 1);
        let c: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i64x2 = i64x2::new(8, 9);
        let r: i64x2 = transmute(vmlal_high_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_lane_u16() {
        let a: u32x4 = u32x4::new(8, 7, 6, 5);
        let b: u16x8 = u16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let c: u16x4 = u16x4::new(0, 2, 0, 0);
        let e: u32x4 = u32x4::new(8, 9, 10, 11);
        let r: u32x4 = transmute(vmlal_high_lane_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_laneq_u16() {
        let a: u32x4 = u32x4::new(8, 7, 6, 5);
        let b: u16x8 = u16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let c: u16x8 = u16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: u32x4 = u32x4::new(8, 9, 10, 11);
        let r: u32x4 = transmute(vmlal_high_laneq_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_lane_u32() {
        let a: u64x2 = u64x2::new(8, 7);
        let b: u32x4 = u32x4::new(3, 3, 0, 1);
        let c: u32x2 = u32x2::new(0, 2);
        let e: u64x2 = u64x2::new(8, 9);
        let r: u64x2 = transmute(vmlal_high_lane_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_high_laneq_u32() {
        let a: u64x2 = u64x2::new(8, 7);
        let b: u32x4 = u32x4::new(3, 3, 0, 1);
        let c: u32x4 = u32x4::new(0, 2, 0, 0);
        let e: u64x2 = u64x2::new(8, 9);
        let r: u64x2 = transmute(vmlal_high_laneq_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_f64() {
        let a: f64 = 6.;
        let b: f64 = 2.;
        let c: f64 = 3.;
        let e: f64 = 0.;
        let r: f64 = transmute(vmls_f64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_f64() {
        let a: f64x2 = f64x2::new(6., 7.);
        let b: f64x2 = f64x2::new(2., 2.);
        let c: f64x2 = f64x2::new(3., 3.);
        let e: f64x2 = f64x2::new(0., 1.);
        let r: f64x2 = transmute(vmlsq_f64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_s8() {
        let a: i16x8 = i16x8::new(14, 15, 16, 17, 18, 19, 20, 21);
        let b: i8x16 = i8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let c: i8x16 = i8x16::new(3, 3, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7);
        let e: i16x8 = i16x8::new(14, 13, 12, 11, 10, 9, 8, 7);
        let r: i16x8 = transmute(vmlsl_high_s8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_s16() {
        let a: i32x4 = i32x4::new(14, 15, 16, 17);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: i16x8 = i16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let e: i32x4 = i32x4::new(14, 13, 12, 11);
        let r: i32x4 = transmute(vmlsl_high_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_s32() {
        let a: i64x2 = i64x2::new(14, 15);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let c: i32x4 = i32x4::new(3, 3, 0, 1);
        let e: i64x2 = i64x2::new(14, 13);
        let r: i64x2 = transmute(vmlsl_high_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_u8() {
        let a: u16x8 = u16x8::new(14, 15, 16, 17, 18, 19, 20, 21);
        let b: u8x16 = u8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let c: u8x16 = u8x16::new(3, 3, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(14, 13, 12, 11, 10, 9, 8, 7);
        let r: u16x8 = transmute(vmlsl_high_u8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_u16() {
        let a: u32x4 = u32x4::new(14, 15, 16, 17);
        let b: u16x8 = u16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: u16x8 = u16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let e: u32x4 = u32x4::new(14, 13, 12, 11);
        let r: u32x4 = transmute(vmlsl_high_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_u32() {
        let a: u64x2 = u64x2::new(14, 15);
        let b: u32x4 = u32x4::new(2, 2, 2, 2);
        let c: u32x4 = u32x4::new(3, 3, 0, 1);
        let e: u64x2 = u64x2::new(14, 13);
        let r: u64x2 = transmute(vmlsl_high_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_n_s16() {
        let a: i32x4 = i32x4::new(14, 15, 16, 17);
        let b: i16x8 = i16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let c: i16 = 2;
        let e: i32x4 = i32x4::new(14, 13, 12, 11);
        let r: i32x4 = transmute(vmlsl_high_n_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_n_s32() {
        let a: i64x2 = i64x2::new(14, 15);
        let b: i32x4 = i32x4::new(3, 3, 0, 1);
        let c: i32 = 2;
        let e: i64x2 = i64x2::new(14, 13);
        let r: i64x2 = transmute(vmlsl_high_n_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_n_u16() {
        let a: u32x4 = u32x4::new(14, 15, 16, 17);
        let b: u16x8 = u16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let c: u16 = 2;
        let e: u32x4 = u32x4::new(14, 13, 12, 11);
        let r: u32x4 = transmute(vmlsl_high_n_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_n_u32() {
        let a: u64x2 = u64x2::new(14, 15);
        let b: u32x4 = u32x4::new(3, 3, 0, 1);
        let c: u32 = 2;
        let e: u64x2 = u64x2::new(14, 13);
        let r: u64x2 = transmute(vmlsl_high_n_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_lane_s16() {
        let a: i32x4 = i32x4::new(14, 15, 16, 17);
        let b: i16x8 = i16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let c: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i32x4 = i32x4::new(14, 13, 12, 11);
        let r: i32x4 = transmute(vmlsl_high_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_laneq_s16() {
        let a: i32x4 = i32x4::new(14, 15, 16, 17);
        let b: i16x8 = i16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let c: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i32x4 = i32x4::new(14, 13, 12, 11);
        let r: i32x4 = transmute(vmlsl_high_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_lane_s32() {
        let a: i64x2 = i64x2::new(14, 15);
        let b: i32x4 = i32x4::new(3, 3, 0, 1);
        let c: i32x2 = i32x2::new(0, 2);
        let e: i64x2 = i64x2::new(14, 13);
        let r: i64x2 = transmute(vmlsl_high_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_laneq_s32() {
        let a: i64x2 = i64x2::new(14, 15);
        let b: i32x4 = i32x4::new(3, 3, 0, 1);
        let c: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i64x2 = i64x2::new(14, 13);
        let r: i64x2 = transmute(vmlsl_high_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_lane_u16() {
        let a: u32x4 = u32x4::new(14, 15, 16, 17);
        let b: u16x8 = u16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let c: u16x4 = u16x4::new(0, 2, 0, 0);
        let e: u32x4 = u32x4::new(14, 13, 12, 11);
        let r: u32x4 = transmute(vmlsl_high_lane_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_laneq_u16() {
        let a: u32x4 = u32x4::new(14, 15, 16, 17);
        let b: u16x8 = u16x8::new(3, 3, 0, 1, 0, 1, 2, 3);
        let c: u16x8 = u16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: u32x4 = u32x4::new(14, 13, 12, 11);
        let r: u32x4 = transmute(vmlsl_high_laneq_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_lane_u32() {
        let a: u64x2 = u64x2::new(14, 15);
        let b: u32x4 = u32x4::new(3, 3, 0, 1);
        let c: u32x2 = u32x2::new(0, 2);
        let e: u64x2 = u64x2::new(14, 13);
        let r: u64x2 = transmute(vmlsl_high_lane_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_high_laneq_u32() {
        let a: u64x2 = u64x2::new(14, 15);
        let b: u32x4 = u32x4::new(3, 3, 0, 1);
        let c: u32x4 = u32x4::new(0, 2, 0, 0);
        let e: u64x2 = u64x2::new(14, 13);
        let r: u64x2 = transmute(vmlsl_high_laneq_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_high_s16() {
        let a: i8x8 = i8x8::new(0, 1, 2, 3, 2, 3, 4, 5);
        let b: i16x8 = i16x8::new(2, 3, 4, 5, 12, 13, 14, 15);
        let e: i8x16 = i8x16::new(0, 1, 2, 3, 2, 3, 4, 5, 2, 3, 4, 5, 12, 13, 14, 15);
        let r: i8x16 = transmute(vmovn_high_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_high_s32() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let b: i32x4 = i32x4::new(2, 3, 4, 5);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 2, 3, 4, 5);
        let r: i16x8 = transmute(vmovn_high_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_high_s64() {
        let a: i32x2 = i32x2::new(0, 1);
        let b: i64x2 = i64x2::new(2, 3);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vmovn_high_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_high_u16() {
        let a: u8x8 = u8x8::new(0, 1, 2, 3, 2, 3, 4, 5);
        let b: u16x8 = u16x8::new(2, 3, 4, 5, 12, 13, 14, 15);
        let e: u8x16 = u8x16::new(0, 1, 2, 3, 2, 3, 4, 5, 2, 3, 4, 5, 12, 13, 14, 15);
        let r: u8x16 = transmute(vmovn_high_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_high_u32() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let b: u32x4 = u32x4::new(2, 3, 4, 5);
        let e: u16x8 = u16x8::new(0, 1, 2, 3, 2, 3, 4, 5);
        let r: u16x8 = transmute(vmovn_high_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_high_u64() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: u64x2 = u64x2::new(2, 3);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vmovn_high_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vneg_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vneg_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vnegq_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i64x2 = i64x2::new(0, -1);
        let r: i64x2 = transmute(vnegq_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vneg_f64() {
        let a: f64 = 0.;
        let e: f64 = 0.;
        let r: f64 = transmute(vneg_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vnegq_f64() {
        let a: f64x2 = f64x2::new(0., 1.);
        let e: f64x2 = f64x2::new(0., -1.);
        let r: f64x2 = transmute(vnegq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqneg_s64() {
        let a: i64x1 = i64x1::new(-9223372036854775808);
        let e: i64x1 = i64x1::new(0x7F_FF_FF_FF_FF_FF_FF_FF);
        let r: i64x1 = transmute(vqneg_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqnegq_s64() {
        let a: i64x2 = i64x2::new(-9223372036854775808, 0);
        let e: i64x2 = i64x2::new(0x7F_FF_FF_FF_FF_FF_FF_FF, 0);
        let r: i64x2 = transmute(vqnegq_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubb_s8() {
        let a: i8 = 42;
        let b: i8 = 1;
        let e: i8 = 41;
        let r: i8 = transmute(vqsubb_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubh_s16() {
        let a: i16 = 42;
        let b: i16 = 1;
        let e: i16 = 41;
        let r: i16 = transmute(vqsubh_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubb_u8() {
        let a: u8 = 42;
        let b: u8 = 1;
        let e: u8 = 41;
        let r: u8 = transmute(vqsubb_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubh_u16() {
        let a: u16 = 42;
        let b: u16 = 1;
        let e: u16 = 41;
        let r: u16 = transmute(vqsubh_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubs_u32() {
        let a: u32 = 42;
        let b: u32 = 1;
        let e: u32 = 41;
        let r: u32 = transmute(vqsubs_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubd_u64() {
        let a: u64 = 42;
        let b: u64 = 1;
        let e: u64 = 41;
        let r: u64 = transmute(vqsubd_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubs_s32() {
        let a: i32 = 42;
        let b: i32 = 1;
        let e: i32 = 41;
        let r: i32 = transmute(vqsubs_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubd_s64() {
        let a: i64 = 42;
        let b: i64 = 1;
        let e: i64 = 41;
        let r: i64 = transmute(vqsubd_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrbit_s8() {
        let a: i8x8 = i8x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let e: i8x8 = i8x8::new(0, 64, 32, 96, 16, 80, 48, 112);
        let r: i8x8 = transmute(vrbit_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrbitq_s8() {
        let a: i8x16 = i8x16::new(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let e: i8x16 = i8x16::new(0, 64, 32, 96, 16, 80, 48, 112, 8, 72, 40, 104, 24, 88, 56, 120);
        let r: i8x16 = transmute(vrbitq_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrbit_u8() {
        let a: u8x8 = u8x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let e: u8x8 = u8x8::new(0, 64, 32, 96, 16, 80, 48, 112);
        let r: u8x8 = transmute(vrbit_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrbitq_u8() {
        let a: u8x16 = u8x16::new(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let e: u8x16 = u8x16::new(0, 64, 32, 96, 16, 80, 48, 112, 8, 72, 40, 104, 24, 88, 56, 120);
        let r: u8x16 = transmute(vrbitq_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrbit_p8() {
        let a: i8x8 = i8x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let e: i8x8 = i8x8::new(0, 64, 32, 96, 16, 80, 48, 112);
        let r: i8x8 = transmute(vrbit_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrbitq_p8() {
        let a: i8x16 = i8x16::new(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let e: i8x16 = i8x16::new(0, 64, 32, 96, 16, 80, 48, 112, 8, 72, 40, 104, 24, 88, 56, 120);
        let r: i8x16 = transmute(vrbitq_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndx_f32() {
        let a: f32x2 = f32x2::new(-1.5, 0.5);
        let e: f32x2 = f32x2::new(-2.0, 0.0);
        let r: f32x2 = transmute(vrndx_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndxq_f32() {
        let a: f32x4 = f32x4::new(-1.5, 0.5, 1.5, 2.5);
        let e: f32x4 = f32x4::new(-2.0, 0.0, 2.0, 2.0);
        let r: f32x4 = transmute(vrndxq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndx_f64() {
        let a: f64 = -1.5;
        let e: f64 = -2.0;
        let r: f64 = transmute(vrndx_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndxq_f64() {
        let a: f64x2 = f64x2::new(-1.5, 0.5);
        let e: f64x2 = f64x2::new(-2.0, 0.0);
        let r: f64x2 = transmute(vrndxq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrnda_f32() {
        let a: f32x2 = f32x2::new(-1.5, 0.5);
        let e: f32x2 = f32x2::new(-2.0, 1.0);
        let r: f32x2 = transmute(vrnda_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndaq_f32() {
        let a: f32x4 = f32x4::new(-1.5, 0.5, 1.5, 2.5);
        let e: f32x4 = f32x4::new(-2.0, 1.0, 2.0, 3.0);
        let r: f32x4 = transmute(vrndaq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrnda_f64() {
        let a: f64 = -1.5;
        let e: f64 = -2.0;
        let r: f64 = transmute(vrnda_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndaq_f64() {
        let a: f64x2 = f64x2::new(-1.5, 0.5);
        let e: f64x2 = f64x2::new(-2.0, 1.0);
        let r: f64x2 = transmute(vrndaq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndn_f64() {
        let a: f64 = -1.5;
        let e: f64 = -2.0;
        let r: f64 = transmute(vrndn_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndnq_f64() {
        let a: f64x2 = f64x2::new(-1.5, 0.5);
        let e: f64x2 = f64x2::new(-2.0, 0.0);
        let r: f64x2 = transmute(vrndnq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndm_f32() {
        let a: f32x2 = f32x2::new(-1.5, 0.5);
        let e: f32x2 = f32x2::new(-2.0, 0.0);
        let r: f32x2 = transmute(vrndm_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndmq_f32() {
        let a: f32x4 = f32x4::new(-1.5, 0.5, 1.5, 2.5);
        let e: f32x4 = f32x4::new(-2.0, 0.0, 1.0, 2.0);
        let r: f32x4 = transmute(vrndmq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndm_f64() {
        let a: f64 = -1.5;
        let e: f64 = -2.0;
        let r: f64 = transmute(vrndm_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndmq_f64() {
        let a: f64x2 = f64x2::new(-1.5, 0.5);
        let e: f64x2 = f64x2::new(-2.0, 0.0);
        let r: f64x2 = transmute(vrndmq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndp_f32() {
        let a: f32x2 = f32x2::new(-1.5, 0.5);
        let e: f32x2 = f32x2::new(-1.0, 1.0);
        let r: f32x2 = transmute(vrndp_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndpq_f32() {
        let a: f32x4 = f32x4::new(-1.5, 0.5, 1.5, 2.5);
        let e: f32x4 = f32x4::new(-1.0, 1.0, 2.0, 3.0);
        let r: f32x4 = transmute(vrndpq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndp_f64() {
        let a: f64 = -1.5;
        let e: f64 = -1.0;
        let r: f64 = transmute(vrndp_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndpq_f64() {
        let a: f64x2 = f64x2::new(-1.5, 0.5);
        let e: f64x2 = f64x2::new(-1.0, 1.0);
        let r: f64x2 = transmute(vrndpq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrnd_f32() {
        let a: f32x2 = f32x2::new(-1.5, 0.5);
        let e: f32x2 = f32x2::new(-1.0, 0.0);
        let r: f32x2 = transmute(vrnd_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndq_f32() {
        let a: f32x4 = f32x4::new(-1.5, 0.5, 1.5, 2.5);
        let e: f32x4 = f32x4::new(-1.0, 0.0, 1.0, 2.0);
        let r: f32x4 = transmute(vrndq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrnd_f64() {
        let a: f64 = -1.5;
        let e: f64 = -1.0;
        let r: f64 = transmute(vrnd_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndq_f64() {
        let a: f64x2 = f64x2::new(-1.5, 0.5);
        let e: f64x2 = f64x2::new(-1.0, 0.0);
        let r: f64x2 = transmute(vrndq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndi_f32() {
        let a: f32x2 = f32x2::new(-1.5, 0.5);
        let e: f32x2 = f32x2::new(-2.0, 0.0);
        let r: f32x2 = transmute(vrndi_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndiq_f32() {
        let a: f32x4 = f32x4::new(-1.5, 0.5, 1.5, 2.5);
        let e: f32x4 = f32x4::new(-2.0, 0.0, 2.0, 2.0);
        let r: f32x4 = transmute(vrndiq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndi_f64() {
        let a: f64 = -1.5;
        let e: f64 = -2.0;
        let r: f64 = transmute(vrndi_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndiq_f64() {
        let a: f64x2 = f64x2::new(-1.5, 0.5);
        let e: f64x2 = f64x2::new(-2.0, 0.0);
        let r: f64x2 = transmute(vrndiq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddb_s8() {
        let a: i8 = 42;
        let b: i8 = 1;
        let e: i8 = 43;
        let r: i8 = transmute(vqaddb_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddh_s16() {
        let a: i16 = 42;
        let b: i16 = 1;
        let e: i16 = 43;
        let r: i16 = transmute(vqaddh_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddb_u8() {
        let a: u8 = 42;
        let b: u8 = 1;
        let e: u8 = 43;
        let r: u8 = transmute(vqaddb_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddh_u16() {
        let a: u16 = 42;
        let b: u16 = 1;
        let e: u16 = 43;
        let r: u16 = transmute(vqaddh_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadds_u32() {
        let a: u32 = 42;
        let b: u32 = 1;
        let e: u32 = 43;
        let r: u32 = transmute(vqadds_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddd_u64() {
        let a: u64 = 42;
        let b: u64 = 1;
        let e: u64 = 43;
        let r: u64 = transmute(vqaddd_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadds_s32() {
        let a: i32 = 42;
        let b: i32 = 1;
        let e: i32 = 43;
        let r: i32 = transmute(vqadds_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddd_s64() {
        let a: i64 = 42;
        let b: i64 = 1;
        let e: i64 = 43;
        let r: i64 = transmute(vqaddd_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_f64() {
        let a: f64 = 1.0;
        let b: f64 = 2.0;
        let e: f64 = 2.0;
        let r: f64 = transmute(vmul_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_f64() {
        let a: f64x2 = f64x2::new(1.0, 2.0);
        let b: f64x2 = f64x2::new(2.0, 3.0);
        let e: f64x2 = f64x2::new(2.0, 6.0);
        let r: f64x2 = transmute(vmulq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_n_f64() {
        let a: f64 = 1.;
        let b: f64 = 2.;
        let e: f64 = 2.;
        let r: f64 = transmute(vmul_n_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_n_f64() {
        let a: f64x2 = f64x2::new(1., 2.);
        let b: f64 = 2.;
        let e: f64x2 = f64x2::new(2., 4.);
        let r: f64x2 = transmute(vmulq_n_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_lane_f64() {
        let a: f64 = 1.;
        let b: f64 = 2.;
        let e: f64 = 2.;
        let r: f64 = transmute(vmul_lane_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_laneq_f64() {
        let a: f64 = 1.;
        let b: f64x2 = f64x2::new(2., 0.);
        let e: f64 = 2.;
        let r: f64 = transmute(vmul_laneq_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_lane_f64() {
        let a: f64x2 = f64x2::new(1., 2.);
        let b: f64 = 2.;
        let e: f64x2 = f64x2::new(2., 4.);
        let r: f64x2 = transmute(vmulq_lane_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_laneq_f64() {
        let a: f64x2 = f64x2::new(1., 2.);
        let b: f64x2 = f64x2::new(2., 0.);
        let e: f64x2 = f64x2::new(2., 4.);
        let r: f64x2 = transmute(vmulq_laneq_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmuls_lane_f32() {
        let a: f32 = 1.;
        let b: f32x2 = f32x2::new(2., 0.);
        let e: f32 = 2.;
        let r: f32 = transmute(vmuls_lane_f32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmuls_laneq_f32() {
        let a: f32 = 1.;
        let b: f32x4 = f32x4::new(2., 0., 0., 0.);
        let e: f32 = 2.;
        let r: f32 = transmute(vmuls_laneq_f32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmuld_lane_f64() {
        let a: f64 = 1.;
        let b: f64 = 2.;
        let e: f64 = 2.;
        let r: f64 = transmute(vmuld_lane_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmuld_laneq_f64() {
        let a: f64 = 1.;
        let b: f64x2 = f64x2::new(2., 0.);
        let e: f64 = 2.;
        let r: f64 = transmute(vmuld_laneq_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_s8() {
        let a: i8x16 = i8x16::new(1, 2, 9, 10, 9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        let e: i16x8 = i16x8::new(9, 20, 11, 24, 13, 28, 15, 32);
        let r: i16x8 = transmute(vmull_high_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_s16() {
        let a: i16x8 = i16x8::new(1, 2, 9, 10, 9, 10, 11, 12);
        let b: i16x8 = i16x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: i32x4 = i32x4::new(9, 20, 11, 24);
        let r: i32x4 = transmute(vmull_high_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_s32() {
        let a: i32x4 = i32x4::new(1, 2, 9, 10);
        let b: i32x4 = i32x4::new(1, 2, 1, 2);
        let e: i64x2 = i64x2::new(9, 20);
        let r: i64x2 = transmute(vmull_high_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_u8() {
        let a: u8x16 = u8x16::new(1, 2, 9, 10, 9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        let e: u16x8 = u16x8::new(9, 20, 11, 24, 13, 28, 15, 32);
        let r: u16x8 = transmute(vmull_high_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_u16() {
        let a: u16x8 = u16x8::new(1, 2, 9, 10, 9, 10, 11, 12);
        let b: u16x8 = u16x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: u32x4 = u32x4::new(9, 20, 11, 24);
        let r: u32x4 = transmute(vmull_high_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_u32() {
        let a: u32x4 = u32x4::new(1, 2, 9, 10);
        let b: u32x4 = u32x4::new(1, 2, 1, 2);
        let e: u64x2 = u64x2::new(9, 20);
        let r: u64x2 = transmute(vmull_high_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_p64() {
        let a: p64 = 15;
        let b: p64 = 3;
        let e: p128 = 17;
        let r: p128 = transmute(vmull_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_p8() {
        let a: i8x16 = i8x16::new(1, 2, 9, 10, 9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3);
        let e: i16x8 = i16x8::new(9, 30, 11, 20, 13, 18, 15, 48);
        let r: i16x8 = transmute(vmull_high_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_p64() {
        let a: i64x2 = i64x2::new(1, 15);
        let b: i64x2 = i64x2::new(1, 3);
        let e: p128 = 17;
        let r: p128 = transmute(vmull_high_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_n_s16() {
        let a: i16x8 = i16x8::new(1, 2, 9, 10, 9, 10, 11, 12);
        let b: i16 = 2;
        let e: i32x4 = i32x4::new(18, 20, 22, 24);
        let r: i32x4 = transmute(vmull_high_n_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_n_s32() {
        let a: i32x4 = i32x4::new(1, 2, 9, 10);
        let b: i32 = 2;
        let e: i64x2 = i64x2::new(18, 20);
        let r: i64x2 = transmute(vmull_high_n_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_n_u16() {
        let a: u16x8 = u16x8::new(1, 2, 9, 10, 9, 10, 11, 12);
        let b: u16 = 2;
        let e: u32x4 = u32x4::new(18, 20, 22, 24);
        let r: u32x4 = transmute(vmull_high_n_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_n_u32() {
        let a: u32x4 = u32x4::new(1, 2, 9, 10);
        let b: u32 = 2;
        let e: u64x2 = u64x2::new(18, 20);
        let r: u64x2 = transmute(vmull_high_n_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_lane_s16() {
        let a: i16x8 = i16x8::new(1, 2, 9, 10, 9, 10, 11, 12);
        let b: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i32x4 = i32x4::new(18, 20, 22, 24);
        let r: i32x4 = transmute(vmull_high_lane_s16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_laneq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 9, 10, 9, 10, 11, 12);
        let b: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i32x4 = i32x4::new(18, 20, 22, 24);
        let r: i32x4 = transmute(vmull_high_laneq_s16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_lane_s32() {
        let a: i32x4 = i32x4::new(1, 2, 9, 10);
        let b: i32x2 = i32x2::new(0, 2);
        let e: i64x2 = i64x2::new(18, 20);
        let r: i64x2 = transmute(vmull_high_lane_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_laneq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 9, 10);
        let b: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i64x2 = i64x2::new(18, 20);
        let r: i64x2 = transmute(vmull_high_laneq_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_lane_u16() {
        let a: u16x8 = u16x8::new(1, 2, 9, 10, 9, 10, 11, 12);
        let b: u16x4 = u16x4::new(0, 2, 0, 0);
        let e: u32x4 = u32x4::new(18, 20, 22, 24);
        let r: u32x4 = transmute(vmull_high_lane_u16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_laneq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 9, 10, 9, 10, 11, 12);
        let b: u16x8 = u16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: u32x4 = u32x4::new(18, 20, 22, 24);
        let r: u32x4 = transmute(vmull_high_laneq_u16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_lane_u32() {
        let a: u32x4 = u32x4::new(1, 2, 9, 10);
        let b: u32x2 = u32x2::new(0, 2);
        let e: u64x2 = u64x2::new(18, 20);
        let r: u64x2 = transmute(vmull_high_lane_u32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_high_laneq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 9, 10);
        let b: u32x4 = u32x4::new(0, 2, 0, 0);
        let e: u64x2 = u64x2::new(18, 20);
        let r: u64x2 = transmute(vmull_high_laneq_u32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulx_f32() {
        let a: f32x2 = f32x2::new(1., 2.);
        let b: f32x2 = f32x2::new(2., 2.);
        let e: f32x2 = f32x2::new(2., 4.);
        let r: f32x2 = transmute(vmulx_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulxq_f32() {
        let a: f32x4 = f32x4::new(1., 2., 3., 4.);
        let b: f32x4 = f32x4::new(2., 2., 2., 2.);
        let e: f32x4 = f32x4::new(2., 4., 6., 8.);
        let r: f32x4 = transmute(vmulxq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulx_f64() {
        let a: f64 = 1.;
        let b: f64 = 2.;
        let e: f64 = 2.;
        let r: f64 = transmute(vmulx_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulxq_f64() {
        let a: f64x2 = f64x2::new(1., 2.);
        let b: f64x2 = f64x2::new(2., 2.);
        let e: f64x2 = f64x2::new(2., 4.);
        let r: f64x2 = transmute(vmulxq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulx_lane_f64() {
        let a: f64 = 1.;
        let b: f64 = 2.;
        let e: f64 = 2.;
        let r: f64 = transmute(vmulx_lane_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulx_laneq_f64() {
        let a: f64 = 1.;
        let b: f64x2 = f64x2::new(2., 0.);
        let e: f64 = 2.;
        let r: f64 = transmute(vmulx_laneq_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulx_lane_f32() {
        let a: f32x2 = f32x2::new(1., 2.);
        let b: f32x2 = f32x2::new(2., 0.);
        let e: f32x2 = f32x2::new(2., 4.);
        let r: f32x2 = transmute(vmulx_lane_f32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulx_laneq_f32() {
        let a: f32x2 = f32x2::new(1., 2.);
        let b: f32x4 = f32x4::new(2., 0., 0., 0.);
        let e: f32x2 = f32x2::new(2., 4.);
        let r: f32x2 = transmute(vmulx_laneq_f32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulxq_lane_f32() {
        let a: f32x4 = f32x4::new(1., 2., 3., 4.);
        let b: f32x2 = f32x2::new(2., 0.);
        let e: f32x4 = f32x4::new(2., 4., 6., 8.);
        let r: f32x4 = transmute(vmulxq_lane_f32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulxq_laneq_f32() {
        let a: f32x4 = f32x4::new(1., 2., 3., 4.);
        let b: f32x4 = f32x4::new(2., 0., 0., 0.);
        let e: f32x4 = f32x4::new(2., 4., 6., 8.);
        let r: f32x4 = transmute(vmulxq_laneq_f32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulxq_lane_f64() {
        let a: f64x2 = f64x2::new(1., 2.);
        let b: f64 = 2.;
        let e: f64x2 = f64x2::new(2., 4.);
        let r: f64x2 = transmute(vmulxq_lane_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulxq_laneq_f64() {
        let a: f64x2 = f64x2::new(1., 2.);
        let b: f64x2 = f64x2::new(2., 0.);
        let e: f64x2 = f64x2::new(2., 4.);
        let r: f64x2 = transmute(vmulxq_laneq_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulxs_f32() {
        let a: f32 = 2.;
        let b: f32 = 3.;
        let e: f32 = 6.;
        let r: f32 = transmute(vmulxs_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulxd_f64() {
        let a: f64 = 2.;
        let b: f64 = 3.;
        let e: f64 = 6.;
        let r: f64 = transmute(vmulxd_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulxs_lane_f32() {
        let a: f32 = 2.;
        let b: f32x2 = f32x2::new(3., 0.);
        let e: f32 = 6.;
        let r: f32 = transmute(vmulxs_lane_f32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulxs_laneq_f32() {
        let a: f32 = 2.;
        let b: f32x4 = f32x4::new(3., 0., 0., 0.);
        let e: f32 = 6.;
        let r: f32 = transmute(vmulxs_laneq_f32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulxd_lane_f64() {
        let a: f64 = 2.;
        let b: f64 = 3.;
        let e: f64 = 6.;
        let r: f64 = transmute(vmulxd_lane_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulxd_laneq_f64() {
        let a: f64 = 2.;
        let b: f64x2 = f64x2::new(3., 0.);
        let e: f64 = 6.;
        let r: f64 = transmute(vmulxd_laneq_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfma_f64() {
        let a: f64 = 8.0;
        let b: f64 = 6.0;
        let c: f64 = 2.0;
        let e: f64 = 20.0;
        let r: f64 = transmute(vfma_f64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmaq_f64() {
        let a: f64x2 = f64x2::new(8.0, 18.0);
        let b: f64x2 = f64x2::new(6.0, 4.0);
        let c: f64x2 = f64x2::new(2.0, 3.0);
        let e: f64x2 = f64x2::new(20.0, 30.0);
        let r: f64x2 = transmute(vfmaq_f64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfma_n_f64() {
        let a: f64 = 2.0;
        let b: f64 = 6.0;
        let c: f64 = 8.0;
        let e: f64 = 50.0;
        let r: f64 = transmute(vfma_n_f64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmaq_n_f64() {
        let a: f64x2 = f64x2::new(2.0, 3.0);
        let b: f64x2 = f64x2::new(6.0, 4.0);
        let c: f64 = 8.0;
        let e: f64x2 = f64x2::new(50.0, 35.0);
        let r: f64x2 = transmute(vfmaq_n_f64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfma_lane_f32() {
        let a: f32x2 = f32x2::new(2., 3.);
        let b: f32x2 = f32x2::new(6., 4.);
        let c: f32x2 = f32x2::new(2., 0.);
        let e: f32x2 = f32x2::new(14., 11.);
        let r: f32x2 = transmute(vfma_lane_f32::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfma_laneq_f32() {
        let a: f32x2 = f32x2::new(2., 3.);
        let b: f32x2 = f32x2::new(6., 4.);
        let c: f32x4 = f32x4::new(2., 0., 0., 0.);
        let e: f32x2 = f32x2::new(14., 11.);
        let r: f32x2 = transmute(vfma_laneq_f32::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmaq_lane_f32() {
        let a: f32x4 = f32x4::new(2., 3., 4., 5.);
        let b: f32x4 = f32x4::new(6., 4., 7., 8.);
        let c: f32x2 = f32x2::new(2., 0.);
        let e: f32x4 = f32x4::new(14., 11., 18., 21.);
        let r: f32x4 = transmute(vfmaq_lane_f32::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmaq_laneq_f32() {
        let a: f32x4 = f32x4::new(2., 3., 4., 5.);
        let b: f32x4 = f32x4::new(6., 4., 7., 8.);
        let c: f32x4 = f32x4::new(2., 0., 0., 0.);
        let e: f32x4 = f32x4::new(14., 11., 18., 21.);
        let r: f32x4 = transmute(vfmaq_laneq_f32::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfma_lane_f64() {
        let a: f64 = 2.;
        let b: f64 = 6.;
        let c: f64 = 2.;
        let e: f64 = 14.;
        let r: f64 = transmute(vfma_lane_f64::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfma_laneq_f64() {
        let a: f64 = 2.;
        let b: f64 = 6.;
        let c: f64x2 = f64x2::new(2., 0.);
        let e: f64 = 14.;
        let r: f64 = transmute(vfma_laneq_f64::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmaq_lane_f64() {
        let a: f64x2 = f64x2::new(2., 3.);
        let b: f64x2 = f64x2::new(6., 4.);
        let c: f64 = 2.;
        let e: f64x2 = f64x2::new(14., 11.);
        let r: f64x2 = transmute(vfmaq_lane_f64::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmaq_laneq_f64() {
        let a: f64x2 = f64x2::new(2., 3.);
        let b: f64x2 = f64x2::new(6., 4.);
        let c: f64x2 = f64x2::new(2., 0.);
        let e: f64x2 = f64x2::new(14., 11.);
        let r: f64x2 = transmute(vfmaq_laneq_f64::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmas_lane_f32() {
        let a: f32 = 2.;
        let b: f32 = 6.;
        let c: f32x2 = f32x2::new(3., 0.);
        let e: f32 = 20.;
        let r: f32 = transmute(vfmas_lane_f32::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmas_laneq_f32() {
        let a: f32 = 2.;
        let b: f32 = 6.;
        let c: f32x4 = f32x4::new(3., 0., 0., 0.);
        let e: f32 = 20.;
        let r: f32 = transmute(vfmas_laneq_f32::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmad_lane_f64() {
        let a: f64 = 2.;
        let b: f64 = 6.;
        let c: f64 = 3.;
        let e: f64 = 20.;
        let r: f64 = transmute(vfmad_lane_f64::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmad_laneq_f64() {
        let a: f64 = 2.;
        let b: f64 = 6.;
        let c: f64x2 = f64x2::new(3., 0.);
        let e: f64 = 20.;
        let r: f64 = transmute(vfmad_laneq_f64::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfms_f64() {
        let a: f64 = 20.0;
        let b: f64 = 6.0;
        let c: f64 = 2.0;
        let e: f64 = 8.0;
        let r: f64 = transmute(vfms_f64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmsq_f64() {
        let a: f64x2 = f64x2::new(20.0, 30.0);
        let b: f64x2 = f64x2::new(6.0, 4.0);
        let c: f64x2 = f64x2::new(2.0, 3.0);
        let e: f64x2 = f64x2::new(8.0, 18.0);
        let r: f64x2 = transmute(vfmsq_f64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfms_n_f64() {
        let a: f64 = 50.0;
        let b: f64 = 6.0;
        let c: f64 = 8.0;
        let e: f64 = 2.0;
        let r: f64 = transmute(vfms_n_f64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmsq_n_f64() {
        let a: f64x2 = f64x2::new(50.0, 35.0);
        let b: f64x2 = f64x2::new(6.0, 4.0);
        let c: f64 = 8.0;
        let e: f64x2 = f64x2::new(2.0, 3.0);
        let r: f64x2 = transmute(vfmsq_n_f64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfms_lane_f32() {
        let a: f32x2 = f32x2::new(14., 11.);
        let b: f32x2 = f32x2::new(6., 4.);
        let c: f32x2 = f32x2::new(2., 0.);
        let e: f32x2 = f32x2::new(2., 3.);
        let r: f32x2 = transmute(vfms_lane_f32::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfms_laneq_f32() {
        let a: f32x2 = f32x2::new(14., 11.);
        let b: f32x2 = f32x2::new(6., 4.);
        let c: f32x4 = f32x4::new(2., 0., 0., 0.);
        let e: f32x2 = f32x2::new(2., 3.);
        let r: f32x2 = transmute(vfms_laneq_f32::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmsq_lane_f32() {
        let a: f32x4 = f32x4::new(14., 11., 18., 21.);
        let b: f32x4 = f32x4::new(6., 4., 7., 8.);
        let c: f32x2 = f32x2::new(2., 0.);
        let e: f32x4 = f32x4::new(2., 3., 4., 5.);
        let r: f32x4 = transmute(vfmsq_lane_f32::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmsq_laneq_f32() {
        let a: f32x4 = f32x4::new(14., 11., 18., 21.);
        let b: f32x4 = f32x4::new(6., 4., 7., 8.);
        let c: f32x4 = f32x4::new(2., 0., 0., 0.);
        let e: f32x4 = f32x4::new(2., 3., 4., 5.);
        let r: f32x4 = transmute(vfmsq_laneq_f32::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfms_lane_f64() {
        let a: f64 = 14.;
        let b: f64 = 6.;
        let c: f64 = 2.;
        let e: f64 = 2.;
        let r: f64 = transmute(vfms_lane_f64::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfms_laneq_f64() {
        let a: f64 = 14.;
        let b: f64 = 6.;
        let c: f64x2 = f64x2::new(2., 0.);
        let e: f64 = 2.;
        let r: f64 = transmute(vfms_laneq_f64::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmsq_lane_f64() {
        let a: f64x2 = f64x2::new(14., 11.);
        let b: f64x2 = f64x2::new(6., 4.);
        let c: f64 = 2.;
        let e: f64x2 = f64x2::new(2., 3.);
        let r: f64x2 = transmute(vfmsq_lane_f64::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmsq_laneq_f64() {
        let a: f64x2 = f64x2::new(14., 11.);
        let b: f64x2 = f64x2::new(6., 4.);
        let c: f64x2 = f64x2::new(2., 0.);
        let e: f64x2 = f64x2::new(2., 3.);
        let r: f64x2 = transmute(vfmsq_laneq_f64::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmss_lane_f32() {
        let a: f32 = 14.;
        let b: f32 = 6.;
        let c: f32x2 = f32x2::new(2., 0.);
        let e: f32 = 2.;
        let r: f32 = transmute(vfmss_lane_f32::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmss_laneq_f32() {
        let a: f32 = 14.;
        let b: f32 = 6.;
        let c: f32x4 = f32x4::new(2., 0., 0., 0.);
        let e: f32 = 2.;
        let r: f32 = transmute(vfmss_laneq_f32::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmsd_lane_f64() {
        let a: f64 = 14.;
        let b: f64 = 6.;
        let c: f64 = 2.;
        let e: f64 = 2.;
        let r: f64 = transmute(vfmsd_lane_f64::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmsd_laneq_f64() {
        let a: f64 = 14.;
        let b: f64 = 6.;
        let c: f64x2 = f64x2::new(2., 0.);
        let e: f64 = 2.;
        let r: f64 = transmute(vfmsd_laneq_f64::<0>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdiv_f32() {
        let a: f32x2 = f32x2::new(2.0, 6.0);
        let b: f32x2 = f32x2::new(1.0, 2.0);
        let e: f32x2 = f32x2::new(2.0, 3.0);
        let r: f32x2 = transmute(vdiv_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdivq_f32() {
        let a: f32x4 = f32x4::new(2.0, 6.0, 4.0, 10.0);
        let b: f32x4 = f32x4::new(1.0, 2.0, 1.0, 2.0);
        let e: f32x4 = f32x4::new(2.0, 3.0, 4.0, 5.0);
        let r: f32x4 = transmute(vdivq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdiv_f64() {
        let a: f64 = 2.0;
        let b: f64 = 1.0;
        let e: f64 = 2.0;
        let r: f64 = transmute(vdiv_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdivq_f64() {
        let a: f64x2 = f64x2::new(2.0, 6.0);
        let b: f64x2 = f64x2::new(1.0, 2.0);
        let e: f64x2 = f64x2::new(2.0, 3.0);
        let r: f64x2 = transmute(vdivq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_f64() {
        let a: f64 = 1.0;
        let b: f64 = 1.0;
        let e: f64 = 0.0;
        let r: f64 = transmute(vsub_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_f64() {
        let a: f64x2 = f64x2::new(1.0, 4.0);
        let b: f64x2 = f64x2::new(1.0, 2.0);
        let e: f64x2 = f64x2::new(0.0, 2.0);
        let r: f64x2 = transmute(vsubq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddlv_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: i32 = 10;
        let r: i32 = transmute(vaddlv_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddlvq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i32 = 36;
        let r: i32 = transmute(vaddlvq_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddlv_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let e: i64 = 3;
        let r: i64 = transmute(vaddlv_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddlvq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: i64 = 10;
        let r: i64 = transmute(vaddlvq_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddlv_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u32 = 10;
        let r: u32 = transmute(vaddlv_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddlvq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u32 = 36;
        let r: u32 = transmute(vaddlvq_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddlv_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let e: u64 = 3;
        let r: u64 = transmute(vaddlv_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddlvq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u64 = 10;
        let r: u64 = transmute(vaddlvq_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubw_high_s8() {
        let a: i16x8 = i16x8::new(8, 9, 10, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16);
        let e: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i16x8 = transmute(vsubw_high_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubw_high_s16() {
        let a: i32x4 = i32x4::new(8, 9, 10, 11);
        let b: i16x8 = i16x8::new(0, 1, 2, 3, 8, 9, 10, 11);
        let e: i32x4 = i32x4::new(0, 0, 0, 0);
        let r: i32x4 = transmute(vsubw_high_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubw_high_s32() {
        let a: i64x2 = i64x2::new(8, 9);
        let b: i32x4 = i32x4::new(6, 7, 8, 9);
        let e: i64x2 = i64x2::new(0, 0);
        let r: i64x2 = transmute(vsubw_high_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubw_high_u8() {
        let a: u16x8 = u16x8::new(8, 9, 10, 11, 12, 13, 14, 15);
        let b: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u16x8 = u16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: u16x8 = transmute(vsubw_high_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubw_high_u16() {
        let a: u32x4 = u32x4::new(8, 9, 10, 11);
        let b: u16x8 = u16x8::new(0, 1, 2, 3, 8, 9, 10, 11);
        let e: u32x4 = u32x4::new(0, 0, 0, 0);
        let r: u32x4 = transmute(vsubw_high_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubw_high_u32() {
        let a: u64x2 = u64x2::new(8, 9);
        let b: u32x4 = u32x4::new(6, 7, 8, 9);
        let e: u64x2 = u64x2::new(0, 0);
        let r: u64x2 = transmute(vsubw_high_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubl_high_s8() {
        let a: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: i8x16 = i8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);
        let e: i16x8 = i16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let r: i16x8 = transmute(vsubl_high_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubl_high_s16() {
        let a: i16x8 = i16x8::new(8, 9, 10, 11, 12, 13, 14, 15);
        let b: i16x8 = i16x8::new(6, 6, 6, 6, 8, 8, 8, 8);
        let e: i32x4 = i32x4::new(4, 5, 6, 7);
        let r: i32x4 = transmute(vsubl_high_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubl_high_s32() {
        let a: i32x4 = i32x4::new(12, 13, 14, 15);
        let b: i32x4 = i32x4::new(6, 6, 8, 8);
        let e: i64x2 = i64x2::new(6, 7);
        let r: i64x2 = transmute(vsubl_high_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubl_high_u8() {
        let a: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: u8x16 = u8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);
        let e: u16x8 = u16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let r: u16x8 = transmute(vsubl_high_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubl_high_u16() {
        let a: u16x8 = u16x8::new(8, 9, 10, 11, 12, 13, 14, 15);
        let b: u16x8 = u16x8::new(6, 6, 6, 6, 8, 8, 8, 8);
        let e: u32x4 = u32x4::new(4, 5, 6, 7);
        let r: u32x4 = transmute(vsubl_high_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubl_high_u32() {
        let a: u32x4 = u32x4::new(12, 13, 14, 15);
        let b: u32x4 = u32x4::new(6, 6, 8, 8);
        let e: u64x2 = u64x2::new(6, 7);
        let r: u64x2 = transmute(vsubl_high_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_f64() {
        let a: f64 = 1.0;
        let b: f64 = 0.0;
        let e: f64 = 1.0;
        let r: f64 = transmute(vmax_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_f64() {
        let a: f64x2 = f64x2::new(1.0, -2.0);
        let b: f64x2 = f64x2::new(0.0, 3.0);
        let e: f64x2 = f64x2::new(1.0, 3.0);
        let r: f64x2 = transmute(vmaxq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxnm_f64() {
        let a: f64 = 1.0;
        let b: f64 = 8.0;
        let e: f64 = 8.0;
        let r: f64 = transmute(vmaxnm_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxnmq_f64() {
        let a: f64x2 = f64x2::new(1.0, 2.0);
        let b: f64x2 = f64x2::new(8.0, 16.0);
        let e: f64x2 = f64x2::new(8.0, 16.0);
        let r: f64x2 = transmute(vmaxnmq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmaxnm_f32() {
        let a: f32x2 = f32x2::new(1.0, 2.0);
        let b: f32x2 = f32x2::new(6.0, -3.0);
        let e: f32x2 = f32x2::new(2.0, 6.0);
        let r: f32x2 = transmute(vpmaxnm_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmaxnmq_f64() {
        let a: f64x2 = f64x2::new(1.0, 2.0);
        let b: f64x2 = f64x2::new(6.0, -3.0);
        let e: f64x2 = f64x2::new(2.0, 6.0);
        let r: f64x2 = transmute(vpmaxnmq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpmaxnmq_f32() {
        let a: f32x4 = f32x4::new(1.0, 2.0, 3.0, -4.0);
        let b: f32x4 = f32x4::new(8.0, 16.0, -1.0, 6.0);
        let e: f32x4 = f32x4::new(2.0, 3.0, 16.0, 6.0);
        let r: f32x4 = transmute(vpmaxnmq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_f64() {
        let a: f64 = 1.0;
        let b: f64 = 0.0;
        let e: f64 = 0.0;
        let r: f64 = transmute(vmin_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_f64() {
        let a: f64x2 = f64x2::new(1.0, -2.0);
        let b: f64x2 = f64x2::new(0.0, 3.0);
        let e: f64x2 = f64x2::new(0.0, -2.0);
        let r: f64x2 = transmute(vminq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminnm_f64() {
        let a: f64 = 1.0;
        let b: f64 = 8.0;
        let e: f64 = 1.0;
        let r: f64 = transmute(vminnm_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminnmq_f64() {
        let a: f64x2 = f64x2::new(1.0, 2.0);
        let b: f64x2 = f64x2::new(8.0, 16.0);
        let e: f64x2 = f64x2::new(1.0, 2.0);
        let r: f64x2 = transmute(vminnmq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpminnm_f32() {
        let a: f32x2 = f32x2::new(1.0, 2.0);
        let b: f32x2 = f32x2::new(6.0, -3.0);
        let e: f32x2 = f32x2::new(1.0, -3.0);
        let r: f32x2 = transmute(vpminnm_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpminnmq_f64() {
        let a: f64x2 = f64x2::new(1.0, 2.0);
        let b: f64x2 = f64x2::new(6.0, -3.0);
        let e: f64x2 = f64x2::new(1.0, -3.0);
        let r: f64x2 = transmute(vpminnmq_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vpminnmq_f32() {
        let a: f32x4 = f32x4::new(1.0, 2.0, 3.0, -4.0);
        let b: f32x4 = f32x4::new(8.0, 16.0, -1.0, 6.0);
        let e: f32x4 = f32x4::new(1.0, -4.0, 8.0, -1.0);
        let r: f32x4 = transmute(vpminnmq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmullh_s16() {
        let a: i16 = 2;
        let b: i16 = 3;
        let e: i32 = 12;
        let r: i32 = transmute(vqdmullh_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulls_s32() {
        let a: i32 = 2;
        let b: i32 = 3;
        let e: i64 = 12;
        let r: i64 = transmute(vqdmulls_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_high_s16() {
        let a: i16x8 = i16x8::new(0, 1, 4, 5, 4, 5, 6, 7);
        let b: i16x8 = i16x8::new(1, 2, 5, 6, 5, 6, 7, 8);
        let e: i32x4 = i32x4::new(40, 60, 84, 112);
        let r: i32x4 = transmute(vqdmull_high_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_high_s32() {
        let a: i32x4 = i32x4::new(0, 1, 4, 5);
        let b: i32x4 = i32x4::new(1, 2, 5, 6);
        let e: i64x2 = i64x2::new(40, 60);
        let r: i64x2 = transmute(vqdmull_high_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_high_n_s16() {
        let a: i16x8 = i16x8::new(0, 2, 8, 10, 8, 10, 12, 14);
        let b: i16 = 2;
        let e: i32x4 = i32x4::new(32, 40, 48, 56);
        let r: i32x4 = transmute(vqdmull_high_n_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_high_n_s32() {
        let a: i32x4 = i32x4::new(0, 2, 8, 10);
        let b: i32 = 2;
        let e: i64x2 = i64x2::new(32, 40);
        let r: i64x2 = transmute(vqdmull_high_n_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_laneq_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x8 = i16x8::new(0, 2, 2, 0, 2, 0, 0, 0);
        let e: i32x4 = i32x4::new(4, 8, 12, 16);
        let r: i32x4 = transmute(vqdmull_laneq_s16::<4>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_laneq_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x4 = i32x4::new(0, 2, 2, 0);
        let e: i64x2 = i64x2::new(4, 8);
        let r: i64x2 = transmute(vqdmull_laneq_s32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmullh_lane_s16() {
        let a: i16 = 2;
        let b: i16x4 = i16x4::new(0, 2, 2, 0);
        let e: i32 = 8;
        let r: i32 = transmute(vqdmullh_lane_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmullh_laneq_s16() {
        let a: i16 = 2;
        let b: i16x8 = i16x8::new(0, 2, 2, 0, 2, 0, 0, 0);
        let e: i32 = 8;
        let r: i32 = transmute(vqdmullh_laneq_s16::<4>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulls_lane_s32() {
        let a: i32 = 2;
        let b: i32x2 = i32x2::new(0, 2);
        let e: i64 = 8;
        let r: i64 = transmute(vqdmulls_lane_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulls_laneq_s32() {
        let a: i32 = 2;
        let b: i32x4 = i32x4::new(0, 2, 2, 0);
        let e: i64 = 8;
        let r: i64 = transmute(vqdmulls_laneq_s32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_high_lane_s16() {
        let a: i16x8 = i16x8::new(0, 1, 4, 5, 4, 5, 6, 7);
        let b: i16x4 = i16x4::new(0, 2, 2, 0);
        let e: i32x4 = i32x4::new(16, 20, 24, 28);
        let r: i32x4 = transmute(vqdmull_high_lane_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_high_lane_s32() {
        let a: i32x4 = i32x4::new(0, 1, 4, 5);
        let b: i32x2 = i32x2::new(0, 2);
        let e: i64x2 = i64x2::new(16, 20);
        let r: i64x2 = transmute(vqdmull_high_lane_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_high_laneq_s16() {
        let a: i16x8 = i16x8::new(0, 1, 4, 5, 4, 5, 6, 7);
        let b: i16x8 = i16x8::new(0, 2, 2, 0, 2, 0, 0, 0);
        let e: i32x4 = i32x4::new(16, 20, 24, 28);
        let r: i32x4 = transmute(vqdmull_high_laneq_s16::<4>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_high_laneq_s32() {
        let a: i32x4 = i32x4::new(0, 1, 4, 5);
        let b: i32x4 = i32x4::new(0, 2, 2, 0);
        let e: i64x2 = i64x2::new(16, 20);
        let r: i64x2 = transmute(vqdmull_high_laneq_s32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_high_s16() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i16x8 = i16x8::new(0, 1, 4, 5, 4, 5, 6, 7);
        let c: i16x8 = i16x8::new(1, 2, 5, 6, 5, 6, 7, 8);
        let e: i32x4 = i32x4::new(41, 62, 87, 116);
        let r: i32x4 = transmute(vqdmlal_high_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_high_s32() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i32x4 = i32x4::new(0, 1, 4, 5);
        let c: i32x4 = i32x4::new(1, 2, 5, 6);
        let e: i64x2 = i64x2::new(41, 62);
        let r: i64x2 = transmute(vqdmlal_high_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_high_n_s16() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i16x8 = i16x8::new(0, 2, 8, 10, 8, 10, 12, 14);
        let c: i16 = 2;
        let e: i32x4 = i32x4::new(33, 42, 51, 60);
        let r: i32x4 = transmute(vqdmlal_high_n_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_high_n_s32() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i32x4 = i32x4::new(0, 2, 8, 10);
        let c: i32 = 2;
        let e: i64x2 = i64x2::new(33, 42);
        let r: i64x2 = transmute(vqdmlal_high_n_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_laneq_s16() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let c: i16x8 = i16x8::new(0, 2, 2, 0, 2, 0, 0, 0);
        let e: i32x4 = i32x4::new(5, 10, 15, 20);
        let r: i32x4 = transmute(vqdmlal_laneq_s16::<2>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_laneq_s32() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i32x2 = i32x2::new(1, 2);
        let c: i32x4 = i32x4::new(0, 2, 2, 0);
        let e: i64x2 = i64x2::new(5, 10);
        let r: i64x2 = transmute(vqdmlal_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_high_lane_s16() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i16x8 = i16x8::new(0, 1, 4, 5, 4, 5, 6, 7);
        let c: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i32x4 = i32x4::new(17, 22, 27, 32);
        let r: i32x4 = transmute(vqdmlal_high_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_high_laneq_s16() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i16x8 = i16x8::new(0, 1, 4, 5, 4, 5, 6, 7);
        let c: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i32x4 = i32x4::new(17, 22, 27, 32);
        let r: i32x4 = transmute(vqdmlal_high_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_high_lane_s32() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i32x4 = i32x4::new(0, 1, 4, 5);
        let c: i32x2 = i32x2::new(0, 2);
        let e: i64x2 = i64x2::new(17, 22);
        let r: i64x2 = transmute(vqdmlal_high_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_high_laneq_s32() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i32x4 = i32x4::new(0, 1, 4, 5);
        let c: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i64x2 = i64x2::new(17, 22);
        let r: i64x2 = transmute(vqdmlal_high_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_high_s16() {
        let a: i32x4 = i32x4::new(39, 58, 81, 108);
        let b: i16x8 = i16x8::new(0, 1, 4, 5, 4, 5, 6, 7);
        let c: i16x8 = i16x8::new(1, 2, 5, 6, 5, 6, 7, 8);
        let e: i32x4 = i32x4::new(-1, -2, -3, -4);
        let r: i32x4 = transmute(vqdmlsl_high_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_high_s32() {
        let a: i64x2 = i64x2::new(39, 58);
        let b: i32x4 = i32x4::new(0, 1, 4, 5);
        let c: i32x4 = i32x4::new(1, 2, 5, 6);
        let e: i64x2 = i64x2::new(-1, -2);
        let r: i64x2 = transmute(vqdmlsl_high_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_high_n_s16() {
        let a: i32x4 = i32x4::new(31, 38, 45, 52);
        let b: i16x8 = i16x8::new(0, 2, 8, 10, 8, 10, 12, 14);
        let c: i16 = 2;
        let e: i32x4 = i32x4::new(-1, -2, -3, -4);
        let r: i32x4 = transmute(vqdmlsl_high_n_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_high_n_s32() {
        let a: i64x2 = i64x2::new(31, 38);
        let b: i32x4 = i32x4::new(0, 2, 8, 10);
        let c: i32 = 2;
        let e: i64x2 = i64x2::new(-1, -2);
        let r: i64x2 = transmute(vqdmlsl_high_n_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_laneq_s16() {
        let a: i32x4 = i32x4::new(3, 6, 9, 12);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let c: i16x8 = i16x8::new(0, 2, 2, 0, 2, 0, 0, 0);
        let e: i32x4 = i32x4::new(-1, -2, -3, -4);
        let r: i32x4 = transmute(vqdmlsl_laneq_s16::<2>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_laneq_s32() {
        let a: i64x2 = i64x2::new(3, 6);
        let b: i32x2 = i32x2::new(1, 2);
        let c: i32x4 = i32x4::new(0, 2, 2, 0);
        let e: i64x2 = i64x2::new(-1, -2);
        let r: i64x2 = transmute(vqdmlsl_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_high_lane_s16() {
        let a: i32x4 = i32x4::new(15, 18, 21, 24);
        let b: i16x8 = i16x8::new(0, 1, 4, 5, 4, 5, 6, 7);
        let c: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i32x4 = i32x4::new(-1, -2, -3, -4);
        let r: i32x4 = transmute(vqdmlsl_high_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_high_laneq_s16() {
        let a: i32x4 = i32x4::new(15, 18, 21, 24);
        let b: i16x8 = i16x8::new(0, 1, 4, 5, 4, 5, 6, 7);
        let c: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i32x4 = i32x4::new(-1, -2, -3, -4);
        let r: i32x4 = transmute(vqdmlsl_high_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_high_lane_s32() {
        let a: i64x2 = i64x2::new(15, 18);
        let b: i32x4 = i32x4::new(0, 1, 4, 5);
        let c: i32x2 = i32x2::new(0, 2);
        let e: i64x2 = i64x2::new(-1, -2);
        let r: i64x2 = transmute(vqdmlsl_high_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_high_laneq_s32() {
        let a: i64x2 = i64x2::new(15, 18);
        let b: i32x4 = i32x4::new(0, 1, 4, 5);
        let c: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i64x2 = i64x2::new(-1, -2);
        let r: i64x2 = transmute(vqdmlsl_high_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulhh_s16() {
        let a: i16 = 1;
        let b: i16 = 2;
        let e: i16 = 0;
        let r: i16 = transmute(vqdmulhh_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulhs_s32() {
        let a: i32 = 1;
        let b: i32 = 2;
        let e: i32 = 0;
        let r: i32 = transmute(vqdmulhs_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulhh_lane_s16() {
        let a: i16 = 2;
        let b: i16x4 = i16x4::new(0, 0, 0x7F_FF, 0);
        let e: i16 = 1;
        let r: i16 = transmute(vqdmulhh_lane_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulhh_laneq_s16() {
        let a: i16 = 2;
        let b: i16x8 = i16x8::new(0, 0, 0x7F_FF, 0, 0, 0, 0, 0);
        let e: i16 = 1;
        let r: i16 = transmute(vqdmulhh_laneq_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulhs_lane_s32() {
        let a: i32 = 2;
        let b: i32x2 = i32x2::new(0, 0x7F_FF_FF_FF);
        let e: i32 = 1;
        let r: i32 = transmute(vqdmulhs_lane_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulhs_laneq_s32() {
        let a: i32 = 2;
        let b: i32x4 = i32x4::new(0, 0x7F_FF_FF_FF, 0, 0);
        let e: i32 = 1;
        let r: i32 = transmute(vqdmulhs_laneq_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovnh_s16() {
        let a: i16 = 1;
        let e: i8 = 1;
        let r: i8 = transmute(vqmovnh_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovns_s32() {
        let a: i32 = 1;
        let e: i16 = 1;
        let r: i16 = transmute(vqmovns_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovnh_u16() {
        let a: u16 = 1;
        let e: u8 = 1;
        let r: u8 = transmute(vqmovnh_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovns_u32() {
        let a: u32 = 1;
        let e: u16 = 1;
        let r: u16 = transmute(vqmovns_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovnd_s64() {
        let a: i64 = 1;
        let e: i32 = 1;
        let r: i32 = transmute(vqmovnd_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovnd_u64() {
        let a: u64 = 1;
        let e: u32 = 1;
        let r: u32 = transmute(vqmovnd_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovn_high_s16() {
        let a: i8x8 = i8x8::new(0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F);
        let b: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let e: i8x16 = i8x16::new(0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F);
        let r: i8x16 = transmute(vqmovn_high_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovn_high_s32() {
        let a: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let b: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let e: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let r: i16x8 = transmute(vqmovn_high_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovn_high_s64() {
        let a: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let b: i64x2 = i64x2::new(0x7F_FF_FF_FF_FF_FF_FF_FF, 0x7F_FF_FF_FF_FF_FF_FF_FF);
        let e: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let r: i32x4 = transmute(vqmovn_high_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovn_high_u16() {
        let a: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let b: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vqmovn_high_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovn_high_u32() {
        let a: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let b: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vqmovn_high_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovn_high_u64() {
        let a: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let b: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vqmovn_high_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovunh_s16() {
        let a: i16 = 1;
        let e: u8 = 1;
        let r: u8 = transmute(vqmovunh_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovuns_s32() {
        let a: i32 = 1;
        let e: u16 = 1;
        let r: u16 = transmute(vqmovuns_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovund_s64() {
        let a: i64 = 1;
        let e: u32 = 1;
        let r: u32 = transmute(vqmovund_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovun_high_s16() {
        let a: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let b: i16x8 = i16x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let e: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x16 = transmute(vqmovun_high_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovun_high_s32() {
        let a: u16x4 = u16x4::new(0, 0, 0, 0);
        let b: i32x4 = i32x4::new(-1, -1, -1, -1);
        let e: u16x8 = u16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: u16x8 = transmute(vqmovun_high_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovun_high_s64() {
        let a: u32x2 = u32x2::new(0, 0);
        let b: i64x2 = i64x2::new(-1, -1);
        let e: u32x4 = u32x4::new(0, 0, 0, 0);
        let r: u32x4 = transmute(vqmovun_high_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulhh_s16() {
        let a: i16 = 1;
        let b: i16 = 2;
        let e: i16 = 0;
        let r: i16 = transmute(vqrdmulhh_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulhs_s32() {
        let a: i32 = 1;
        let b: i32 = 2;
        let e: i32 = 0;
        let r: i32 = transmute(vqrdmulhs_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulhh_lane_s16() {
        let a: i16 = 1;
        let b: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i16 = 0;
        let r: i16 = transmute(vqrdmulhh_lane_s16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulhh_laneq_s16() {
        let a: i16 = 1;
        let b: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i16 = 0;
        let r: i16 = transmute(vqrdmulhh_laneq_s16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulhs_lane_s32() {
        let a: i32 = 1;
        let b: i32x2 = i32x2::new(0, 2);
        let e: i32 = 0;
        let r: i32 = transmute(vqrdmulhs_lane_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulhs_laneq_s32() {
        let a: i32 = 1;
        let b: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i32 = 0;
        let r: i32 = transmute(vqrdmulhs_laneq_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlahh_s16() {
        let a: i16 = 1;
        let b: i16 = 1;
        let c: i16 = 2;
        let e: i16 = 1;
        let r: i16 = transmute(vqrdmlahh_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlahs_s32() {
        let a: i32 = 1;
        let b: i32 = 1;
        let c: i32 = 2;
        let e: i32 = 1;
        let r: i32 = transmute(vqrdmlahs_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlahh_lane_s16() {
        let a: i16 = 1;
        let b: i16 = 1;
        let c: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i16 = 1;
        let r: i16 = transmute(vqrdmlahh_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlahh_laneq_s16() {
        let a: i16 = 1;
        let b: i16 = 1;
        let c: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i16 = 1;
        let r: i16 = transmute(vqrdmlahh_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlahs_lane_s32() {
        let a: i32 = 1;
        let b: i32 = 1;
        let c: i32x2 = i32x2::new(0, 2);
        let e: i32 = 1;
        let r: i32 = transmute(vqrdmlahs_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlahs_laneq_s32() {
        let a: i32 = 1;
        let b: i32 = 1;
        let c: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i32 = 1;
        let r: i32 = transmute(vqrdmlahs_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlshh_s16() {
        let a: i16 = 1;
        let b: i16 = 1;
        let c: i16 = 2;
        let e: i16 = 1;
        let r: i16 = transmute(vqrdmlshh_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlshs_s32() {
        let a: i32 = 1;
        let b: i32 = 1;
        let c: i32 = 2;
        let e: i32 = 1;
        let r: i32 = transmute(vqrdmlshs_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlshh_lane_s16() {
        let a: i16 = 1;
        let b: i16 = 1;
        let c: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i16 = 1;
        let r: i16 = transmute(vqrdmlshh_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlshh_laneq_s16() {
        let a: i16 = 1;
        let b: i16 = 1;
        let c: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i16 = 1;
        let r: i16 = transmute(vqrdmlshh_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlshs_lane_s32() {
        let a: i32 = 1;
        let b: i32 = 1;
        let c: i32x2 = i32x2::new(0, 2);
        let e: i32 = 1;
        let r: i32 = transmute(vqrdmlshs_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlshs_laneq_s32() {
        let a: i32 = 1;
        let b: i32 = 1;
        let c: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i32 = 1;
        let r: i32 = transmute(vqrdmlshs_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshls_s32() {
        let a: i32 = 2;
        let b: i32 = 2;
        let e: i32 = 8;
        let r: i32 = transmute(vqrshls_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshld_s64() {
        let a: i64 = 2;
        let b: i64 = 2;
        let e: i64 = 8;
        let r: i64 = transmute(vqrshld_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshlb_s8() {
        let a: i8 = 1;
        let b: i8 = 2;
        let e: i8 = 4;
        let r: i8 = transmute(vqrshlb_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshlh_s16() {
        let a: i16 = 1;
        let b: i16 = 2;
        let e: i16 = 4;
        let r: i16 = transmute(vqrshlh_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshls_u32() {
        let a: u32 = 2;
        let b: i32 = 2;
        let e: u32 = 8;
        let r: u32 = transmute(vqrshls_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshld_u64() {
        let a: u64 = 2;
        let b: i64 = 2;
        let e: u64 = 8;
        let r: u64 = transmute(vqrshld_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshlb_u8() {
        let a: u8 = 1;
        let b: i8 = 2;
        let e: u8 = 4;
        let r: u8 = transmute(vqrshlb_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshlh_u16() {
        let a: u16 = 1;
        let b: i16 = 2;
        let e: u16 = 4;
        let r: u16 = transmute(vqrshlh_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrnh_n_s16() {
        let a: i16 = 4;
        let e: i8 = 1;
        let r: i8 = transmute(vqrshrnh_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrns_n_s32() {
        let a: i32 = 4;
        let e: i16 = 1;
        let r: i16 = transmute(vqrshrns_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrnd_n_s64() {
        let a: i64 = 4;
        let e: i32 = 1;
        let r: i32 = transmute(vqrshrnd_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrn_high_n_s16() {
        let a: i8x8 = i8x8::new(0, 1, 2, 3, 2, 3, 6, 7);
        let b: i16x8 = i16x8::new(8, 12, 24, 28, 48, 52, 56, 60);
        let e: i8x16 = i8x16::new(0, 1, 2, 3, 2, 3, 6, 7, 2, 3, 6, 7, 12, 13, 14, 15);
        let r: i8x16 = transmute(vqrshrn_high_n_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrn_high_n_s32() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let b: i32x4 = i32x4::new(8, 12, 24, 28);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 2, 3, 6, 7);
        let r: i16x8 = transmute(vqrshrn_high_n_s32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrn_high_n_s64() {
        let a: i32x2 = i32x2::new(0, 1);
        let b: i64x2 = i64x2::new(8, 12);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vqrshrn_high_n_s64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrnh_n_u16() {
        let a: u16 = 4;
        let e: u8 = 1;
        let r: u8 = transmute(vqrshrnh_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrns_n_u32() {
        let a: u32 = 4;
        let e: u16 = 1;
        let r: u16 = transmute(vqrshrns_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrnd_n_u64() {
        let a: u64 = 4;
        let e: u32 = 1;
        let r: u32 = transmute(vqrshrnd_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrn_high_n_u16() {
        let a: u8x8 = u8x8::new(0, 1, 2, 3, 2, 3, 6, 7);
        let b: u16x8 = u16x8::new(8, 12, 24, 28, 48, 52, 56, 60);
        let e: u8x16 = u8x16::new(0, 1, 2, 3, 2, 3, 6, 7, 2, 3, 6, 7, 12, 13, 14, 15);
        let r: u8x16 = transmute(vqrshrn_high_n_u16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrn_high_n_u32() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let b: u32x4 = u32x4::new(8, 12, 24, 28);
        let e: u16x8 = u16x8::new(0, 1, 2, 3, 2, 3, 6, 7);
        let r: u16x8 = transmute(vqrshrn_high_n_u32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrn_high_n_u64() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: u64x2 = u64x2::new(8, 12);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vqrshrn_high_n_u64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrunh_n_s16() {
        let a: i16 = 4;
        let e: u8 = 1;
        let r: u8 = transmute(vqrshrunh_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshruns_n_s32() {
        let a: i32 = 4;
        let e: u16 = 1;
        let r: u16 = transmute(vqrshruns_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrund_n_s64() {
        let a: i64 = 4;
        let e: u32 = 1;
        let r: u32 = transmute(vqrshrund_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrun_high_n_s16() {
        let a: u8x8 = u8x8::new(0, 1, 2, 3, 2, 3, 6, 7);
        let b: i16x8 = i16x8::new(8, 12, 24, 28, 48, 52, 56, 60);
        let e: u8x16 = u8x16::new(0, 1, 2, 3, 2, 3, 6, 7, 2, 3, 6, 7, 12, 13, 14, 15);
        let r: u8x16 = transmute(vqrshrun_high_n_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrun_high_n_s32() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let b: i32x4 = i32x4::new(8, 12, 24, 28);
        let e: u16x8 = u16x8::new(0, 1, 2, 3, 2, 3, 6, 7);
        let r: u16x8 = transmute(vqrshrun_high_n_s32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrun_high_n_s64() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: i64x2 = i64x2::new(8, 12);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vqrshrun_high_n_s64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshld_s64() {
        let a: i64 = 0;
        let b: i64 = 2;
        let e: i64 = 0;
        let r: i64 = transmute(vqshld_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlb_s8() {
        let a: i8 = 1;
        let b: i8 = 2;
        let e: i8 = 4;
        let r: i8 = transmute(vqshlb_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlh_s16() {
        let a: i16 = 1;
        let b: i16 = 2;
        let e: i16 = 4;
        let r: i16 = transmute(vqshlh_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshls_s32() {
        let a: i32 = 1;
        let b: i32 = 2;
        let e: i32 = 4;
        let r: i32 = transmute(vqshls_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshld_u64() {
        let a: u64 = 0;
        let b: i64 = 2;
        let e: u64 = 0;
        let r: u64 = transmute(vqshld_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlb_u8() {
        let a: u8 = 1;
        let b: i8 = 2;
        let e: u8 = 4;
        let r: u8 = transmute(vqshlb_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlh_u16() {
        let a: u16 = 1;
        let b: i16 = 2;
        let e: u16 = 4;
        let r: u16 = transmute(vqshlh_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshls_u32() {
        let a: u32 = 1;
        let b: i32 = 2;
        let e: u32 = 4;
        let r: u32 = transmute(vqshls_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlb_n_s8() {
        let a: i8 = 1;
        let e: i8 = 4;
        let r: i8 = transmute(vqshlb_n_s8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlh_n_s16() {
        let a: i16 = 1;
        let e: i16 = 4;
        let r: i16 = transmute(vqshlh_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshls_n_s32() {
        let a: i32 = 1;
        let e: i32 = 4;
        let r: i32 = transmute(vqshls_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshld_n_s64() {
        let a: i64 = 1;
        let e: i64 = 4;
        let r: i64 = transmute(vqshld_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlb_n_u8() {
        let a: u8 = 1;
        let e: u8 = 4;
        let r: u8 = transmute(vqshlb_n_u8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlh_n_u16() {
        let a: u16 = 1;
        let e: u16 = 4;
        let r: u16 = transmute(vqshlh_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshls_n_u32() {
        let a: u32 = 1;
        let e: u32 = 4;
        let r: u32 = transmute(vqshls_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshld_n_u64() {
        let a: u64 = 1;
        let e: u64 = 4;
        let r: u64 = transmute(vqshld_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrnd_n_s64() {
        let a: i64 = 0;
        let e: i32 = 0;
        let r: i32 = transmute(vqshrnd_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrnh_n_s16() {
        let a: i16 = 4;
        let e: i8 = 1;
        let r: i8 = transmute(vqshrnh_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrns_n_s32() {
        let a: i32 = 4;
        let e: i16 = 1;
        let r: i16 = transmute(vqshrns_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrn_high_n_s16() {
        let a: i8x8 = i8x8::new(0, 1, 8, 9, 8, 9, 10, 11);
        let b: i16x8 = i16x8::new(32, 36, 40, 44, 48, 52, 56, 60);
        let e: i8x16 = i8x16::new(0, 1, 8, 9, 8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: i8x16 = transmute(vqshrn_high_n_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrn_high_n_s32() {
        let a: i16x4 = i16x4::new(0, 1, 8, 9);
        let b: i32x4 = i32x4::new(32, 36, 40, 44);
        let e: i16x8 = i16x8::new(0, 1, 8, 9, 8, 9, 10, 11);
        let r: i16x8 = transmute(vqshrn_high_n_s32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrn_high_n_s64() {
        let a: i32x2 = i32x2::new(0, 1);
        let b: i64x2 = i64x2::new(32, 36);
        let e: i32x4 = i32x4::new(0, 1, 8, 9);
        let r: i32x4 = transmute(vqshrn_high_n_s64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrnd_n_u64() {
        let a: u64 = 0;
        let e: u32 = 0;
        let r: u32 = transmute(vqshrnd_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrnh_n_u16() {
        let a: u16 = 4;
        let e: u8 = 1;
        let r: u8 = transmute(vqshrnh_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrns_n_u32() {
        let a: u32 = 4;
        let e: u16 = 1;
        let r: u16 = transmute(vqshrns_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrn_high_n_u16() {
        let a: u8x8 = u8x8::new(0, 1, 8, 9, 8, 9, 10, 11);
        let b: u16x8 = u16x8::new(32, 36, 40, 44, 48, 52, 56, 60);
        let e: u8x16 = u8x16::new(0, 1, 8, 9, 8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: u8x16 = transmute(vqshrn_high_n_u16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrn_high_n_u32() {
        let a: u16x4 = u16x4::new(0, 1, 8, 9);
        let b: u32x4 = u32x4::new(32, 36, 40, 44);
        let e: u16x8 = u16x8::new(0, 1, 8, 9, 8, 9, 10, 11);
        let r: u16x8 = transmute(vqshrn_high_n_u32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrn_high_n_u64() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: u64x2 = u64x2::new(32, 36);
        let e: u32x4 = u32x4::new(0, 1, 8, 9);
        let r: u32x4 = transmute(vqshrn_high_n_u64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrunh_n_s16() {
        let a: i16 = 4;
        let e: u8 = 1;
        let r: u8 = transmute(vqshrunh_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshruns_n_s32() {
        let a: i32 = 4;
        let e: u16 = 1;
        let r: u16 = transmute(vqshruns_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrund_n_s64() {
        let a: i64 = 4;
        let e: u32 = 1;
        let r: u32 = transmute(vqshrund_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrun_high_n_s16() {
        let a: u8x8 = u8x8::new(0, 1, 8, 9, 8, 9, 10, 11);
        let b: i16x8 = i16x8::new(32, 36, 40, 44, 48, 52, 56, 60);
        let e: u8x16 = u8x16::new(0, 1, 8, 9, 8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: u8x16 = transmute(vqshrun_high_n_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrun_high_n_s32() {
        let a: u16x4 = u16x4::new(0, 1, 8, 9);
        let b: i32x4 = i32x4::new(32, 36, 40, 44);
        let e: u16x8 = u16x8::new(0, 1, 8, 9, 8, 9, 10, 11);
        let r: u16x8 = transmute(vqshrun_high_n_s32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrun_high_n_s64() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: i64x2 = i64x2::new(32, 36);
        let e: u32x4 = u32x4::new(0, 1, 8, 9);
        let r: u32x4 = transmute(vqshrun_high_n_s64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsqrt_f32() {
        let a: f32x2 = f32x2::new(4.0, 9.0);
        let e: f32x2 = f32x2::new(2.0, 3.0);
        let r: f32x2 = transmute(vsqrt_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsqrtq_f32() {
        let a: f32x4 = f32x4::new(4.0, 9.0, 16.0, 25.0);
        let e: f32x4 = f32x4::new(2.0, 3.0, 4.0, 5.0);
        let r: f32x4 = transmute(vsqrtq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsqrt_f64() {
        let a: f64 = 4.0;
        let e: f64 = 2.0;
        let r: f64 = transmute(vsqrt_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsqrtq_f64() {
        let a: f64x2 = f64x2::new(4.0, 9.0);
        let e: f64x2 = f64x2::new(2.0, 3.0);
        let r: f64x2 = transmute(vsqrtq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsqrte_f64() {
        let a: f64 = 1.0;
        let e: f64 = 0.998046875;
        let r: f64 = transmute(vrsqrte_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsqrteq_f64() {
        let a: f64x2 = f64x2::new(1.0, 2.0);
        let e: f64x2 = f64x2::new(0.998046875, 0.705078125);
        let r: f64x2 = transmute(vrsqrteq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrecpe_f64() {
        let a: f64 = 4.0;
        let e: f64 = 0.24951171875;
        let r: f64 = transmute(vrecpe_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrecpeq_f64() {
        let a: f64x2 = f64x2::new(4.0, 3.0);
        let e: f64x2 = f64x2::new(0.24951171875, 0.3330078125);
        let r: f64x2 = transmute(vrecpeq_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s64_p64() {
        let a: i64x1 = i64x1::new(0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_s64_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u64_p64() {
        let a: i64x1 = i64x1::new(0);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vreinterpret_u64_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p64_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_p64_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p64_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_p64_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s64_p64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_s64_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u64_p64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vreinterpretq_u64_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p64_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_p64_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p64_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_p64_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s32_p64() {
        let a: i64x1 = i64x1::new(0);
        let e: i32x2 = i32x2::new(0, 0);
        let r: i32x2 = transmute(vreinterpret_s32_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u32_p64() {
        let a: i64x1 = i64x1::new(0);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vreinterpret_u32_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s32_p64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i32x4 = i32x4::new(0, 0, 1, 0);
        let r: i32x4 = transmute(vreinterpretq_s32_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u32_p64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: u32x4 = u32x4::new(0, 0, 1, 0);
        let r: u32x4 = transmute(vreinterpretq_u32_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p64_s32() {
        let a: i32x2 = i32x2::new(0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_p64_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p64_u32() {
        let a: u32x2 = u32x2::new(0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_p64_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p64_s32() {
        let a: i32x4 = i32x4::new(0, 0, 1, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_p64_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p64_u32() {
        let a: u32x4 = u32x4::new(0, 0, 1, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_p64_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s16_p64() {
        let a: i64x1 = i64x1::new(0);
        let e: i16x4 = i16x4::new(0, 0, 0, 0);
        let r: i16x4 = transmute(vreinterpret_s16_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u16_p64() {
        let a: i64x1 = i64x1::new(0);
        let e: u16x4 = u16x4::new(0, 0, 0, 0);
        let r: u16x4 = transmute(vreinterpret_u16_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p16_p64() {
        let a: i64x1 = i64x1::new(0);
        let e: i16x4 = i16x4::new(0, 0, 0, 0);
        let r: i16x4 = transmute(vreinterpret_p16_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s16_p64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i16x8 = i16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: i16x8 = transmute(vreinterpretq_s16_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u16_p64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: u16x8 = u16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: u16x8 = transmute(vreinterpretq_u16_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p16_p64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i16x8 = i16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: i16x8 = transmute(vreinterpretq_p16_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p64_p16() {
        let a: i16x4 = i16x4::new(0, 0, 0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_p64_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p64_s16() {
        let a: i16x4 = i16x4::new(0, 0, 0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_p64_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p64_u16() {
        let a: u16x4 = u16x4::new(0, 0, 0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_p64_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p64_p16() {
        let a: i16x8 = i16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_p64_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p64_s16() {
        let a: i16x8 = i16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_p64_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p64_u16() {
        let a: u16x8 = u16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_p64_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s8_p64() {
        let a: i64x1 = i64x1::new(0);
        let e: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x8 = transmute(vreinterpret_s8_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u8_p64() {
        let a: i64x1 = i64x1::new(0);
        let e: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x8 = transmute(vreinterpret_u8_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p8_p64() {
        let a: i64x1 = i64x1::new(0);
        let e: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x8 = transmute(vreinterpret_p8_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s8_p64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x16 = transmute(vreinterpretq_s8_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u8_p64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x16 = transmute(vreinterpretq_u8_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p8_p64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x16 = transmute(vreinterpretq_p8_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p64_p8() {
        let a: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_p64_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p64_s8() {
        let a: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_p64_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p64_u8() {
        let a: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_p64_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p64_p8() {
        let a: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_p64_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p64_s8() {
        let a: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_p64_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p64_u8() {
        let a: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_p64_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s8_f64() {
        let a: f64 = 0.;
        let e: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x8 = transmute(vreinterpret_s8_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s16_f64() {
        let a: f64 = 0.;
        let e: i16x4 = i16x4::new(0, 0, 0, 0);
        let r: i16x4 = transmute(vreinterpret_s16_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s32_f64() {
        let a: f64 = 0.;
        let e: i32x2 = i32x2::new(0, 0);
        let r: i32x2 = transmute(vreinterpret_s32_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s64_f64() {
        let a: f64 = 0.;
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s8_f64() {
        let a: f64x2 = f64x2::new(0., 0.);
        let e: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x16 = transmute(vreinterpretq_s8_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s16_f64() {
        let a: f64x2 = f64x2::new(0., 0.);
        let e: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i16x8 = transmute(vreinterpretq_s16_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s32_f64() {
        let a: f64x2 = f64x2::new(0., 0.);
        let e: i32x4 = i32x4::new(0, 0, 0, 0);
        let r: i32x4 = transmute(vreinterpretq_s32_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s64_f64() {
        let a: f64x2 = f64x2::new(0., 0.);
        let e: i64x2 = i64x2::new(0, 0);
        let r: i64x2 = transmute(vreinterpretq_s64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u8_f64() {
        let a: f64 = 0.;
        let e: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x8 = transmute(vreinterpret_u8_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u16_f64() {
        let a: f64 = 0.;
        let e: u16x4 = u16x4::new(0, 0, 0, 0);
        let r: u16x4 = transmute(vreinterpret_u16_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u32_f64() {
        let a: f64 = 0.;
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vreinterpret_u32_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u64_f64() {
        let a: f64 = 0.;
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vreinterpret_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u8_f64() {
        let a: f64x2 = f64x2::new(0., 0.);
        let e: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x16 = transmute(vreinterpretq_u8_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u16_f64() {
        let a: f64x2 = f64x2::new(0., 0.);
        let e: u16x8 = u16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: u16x8 = transmute(vreinterpretq_u16_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u32_f64() {
        let a: f64x2 = f64x2::new(0., 0.);
        let e: u32x4 = u32x4::new(0, 0, 0, 0);
        let r: u32x4 = transmute(vreinterpretq_u32_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u64_f64() {
        let a: f64x2 = f64x2::new(0., 0.);
        let e: u64x2 = u64x2::new(0, 0);
        let r: u64x2 = transmute(vreinterpretq_u64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p8_f64() {
        let a: f64 = 0.;
        let e: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x8 = transmute(vreinterpret_p8_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p16_f64() {
        let a: f64 = 0.;
        let e: i16x4 = i16x4::new(0, 0, 0, 0);
        let r: i16x4 = transmute(vreinterpret_p16_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p64_f32() {
        let a: f32x2 = f32x2::new(0., 0.);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_p64_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p64_f64() {
        let a: f64 = 0.;
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_p64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p8_f64() {
        let a: f64x2 = f64x2::new(0., 0.);
        let e: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x16 = transmute(vreinterpretq_p8_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p16_f64() {
        let a: f64x2 = f64x2::new(0., 0.);
        let e: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i16x8 = transmute(vreinterpretq_p16_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p64_f32() {
        let a: f32x4 = f32x4::new(0., 0., 0., 0.);
        let e: i64x2 = i64x2::new(0, 0);
        let r: i64x2 = transmute(vreinterpretq_p64_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p64_f64() {
        let a: f64x2 = f64x2::new(0., 0.);
        let e: i64x2 = i64x2::new(0, 0);
        let r: i64x2 = transmute(vreinterpretq_p64_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f64_s8() {
        let a: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: f64 = 0.;
        let r: f64 = transmute(vreinterpret_f64_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f64_s16() {
        let a: i16x4 = i16x4::new(0, 0, 0, 0);
        let e: f64 = 0.;
        let r: f64 = transmute(vreinterpret_f64_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f64_s32() {
        let a: i32x2 = i32x2::new(0, 0);
        let e: f64 = 0.;
        let r: f64 = transmute(vreinterpret_f64_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f64_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: f64 = 0.;
        let r: f64 = transmute(vreinterpret_f64_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f64_s8() {
        let a: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let e: f64x2 = f64x2::new(0., 0.);
        let r: f64x2 = transmute(vreinterpretq_f64_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f64_s16() {
        let a: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: f64x2 = f64x2::new(0., 0.);
        let r: f64x2 = transmute(vreinterpretq_f64_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f64_s32() {
        let a: i32x4 = i32x4::new(0, 0, 0, 0);
        let e: f64x2 = f64x2::new(0., 0.);
        let r: f64x2 = transmute(vreinterpretq_f64_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f64_s64() {
        let a: i64x2 = i64x2::new(0, 0);
        let e: f64x2 = f64x2::new(0., 0.);
        let r: f64x2 = transmute(vreinterpretq_f64_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f64_p8() {
        let a: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: f64 = 0.;
        let r: f64 = transmute(vreinterpret_f64_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f64_u16() {
        let a: u16x4 = u16x4::new(0, 0, 0, 0);
        let e: f64 = 0.;
        let r: f64 = transmute(vreinterpret_f64_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f64_u32() {
        let a: u32x2 = u32x2::new(0, 0);
        let e: f64 = 0.;
        let r: f64 = transmute(vreinterpret_f64_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f64_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: f64 = 0.;
        let r: f64 = transmute(vreinterpret_f64_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f64_p8() {
        let a: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let e: f64x2 = f64x2::new(0., 0.);
        let r: f64x2 = transmute(vreinterpretq_f64_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f64_u16() {
        let a: u16x8 = u16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: f64x2 = f64x2::new(0., 0.);
        let r: f64x2 = transmute(vreinterpretq_f64_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f64_u32() {
        let a: u32x4 = u32x4::new(0, 0, 0, 0);
        let e: f64x2 = f64x2::new(0., 0.);
        let r: f64x2 = transmute(vreinterpretq_f64_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f64_u64() {
        let a: u64x2 = u64x2::new(0, 0);
        let e: f64x2 = f64x2::new(0., 0.);
        let r: f64x2 = transmute(vreinterpretq_f64_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f64_u8() {
        let a: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: f64 = 0.;
        let r: f64 = transmute(vreinterpret_f64_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f64_p16() {
        let a: i16x4 = i16x4::new(0, 0, 0, 0);
        let e: f64 = 0.;
        let r: f64 = transmute(vreinterpret_f64_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f64_p64() {
        let a: i64x1 = i64x1::new(0);
        let e: f64 = 0.;
        let r: f64 = transmute(vreinterpret_f64_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f32_p64() {
        let a: i64x1 = i64x1::new(0);
        let e: f32x2 = f32x2::new(0., 0.);
        let r: f32x2 = transmute(vreinterpret_f32_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f64_u8() {
        let a: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let e: f64x2 = f64x2::new(0., 0.);
        let r: f64x2 = transmute(vreinterpretq_f64_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f64_p16() {
        let a: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: f64x2 = f64x2::new(0., 0.);
        let r: f64x2 = transmute(vreinterpretq_f64_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f64_p64() {
        let a: i64x2 = i64x2::new(0, 0);
        let e: f64x2 = f64x2::new(0., 0.);
        let r: f64x2 = transmute(vreinterpretq_f64_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f32_p64() {
        let a: i64x2 = i64x2::new(0, 0);
        let e: f32x4 = f32x4::new(0., 0., 0., 0.);
        let r: f32x4 = transmute(vreinterpretq_f32_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f64_f32() {
        let a: f32x2 = f32x2::new(0., 0.);
        let e: f64 = 0.;
        let r: f64 = transmute(vreinterpret_f64_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f32_f64() {
        let a: f64 = 0.;
        let e: f32x2 = f32x2::new(0., 0.);
        let r: f32x2 = transmute(vreinterpret_f32_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f64_f32() {
        let a: f32x4 = f32x4::new(0., 0., 0., 0.);
        let e: f64x2 = f64x2::new(0., 0.);
        let r: f64x2 = transmute(vreinterpretq_f64_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f32_f64() {
        let a: f64x2 = f64x2::new(0., 0.);
        let e: f32x4 = f32x4::new(0., 0., 0., 0.);
        let r: f32x4 = transmute(vreinterpretq_f32_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshld_s64() {
        let a: i64 = 1;
        let b: i64 = 2;
        let e: i64 = 4;
        let r: i64 = transmute(vrshld_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshld_u64() {
        let a: u64 = 1;
        let b: i64 = 2;
        let e: u64 = 4;
        let r: u64 = transmute(vrshld_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrd_n_s64() {
        let a: i64 = 4;
        let e: i64 = 1;
        let r: i64 = transmute(vrshrd_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrd_n_u64() {
        let a: u64 = 4;
        let e: u64 = 1;
        let r: u64 = transmute(vrshrd_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrn_high_n_s16() {
        let a: i8x8 = i8x8::new(0, 1, 8, 9, 8, 9, 10, 11);
        let b: i16x8 = i16x8::new(32, 36, 40, 44, 48, 52, 56, 60);
        let e: i8x16 = i8x16::new(0, 1, 8, 9, 8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: i8x16 = transmute(vrshrn_high_n_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrn_high_n_s32() {
        let a: i16x4 = i16x4::new(0, 1, 8, 9);
        let b: i32x4 = i32x4::new(32, 36, 40, 44);
        let e: i16x8 = i16x8::new(0, 1, 8, 9, 8, 9, 10, 11);
        let r: i16x8 = transmute(vrshrn_high_n_s32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrn_high_n_s64() {
        let a: i32x2 = i32x2::new(0, 1);
        let b: i64x2 = i64x2::new(32, 36);
        let e: i32x4 = i32x4::new(0, 1, 8, 9);
        let r: i32x4 = transmute(vrshrn_high_n_s64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrn_high_n_u16() {
        let a: u8x8 = u8x8::new(0, 1, 8, 9, 8, 9, 10, 11);
        let b: u16x8 = u16x8::new(32, 36, 40, 44, 48, 52, 56, 60);
        let e: u8x16 = u8x16::new(0, 1, 8, 9, 8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: u8x16 = transmute(vrshrn_high_n_u16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrn_high_n_u32() {
        let a: u16x4 = u16x4::new(0, 1, 8, 9);
        let b: u32x4 = u32x4::new(32, 36, 40, 44);
        let e: u16x8 = u16x8::new(0, 1, 8, 9, 8, 9, 10, 11);
        let r: u16x8 = transmute(vrshrn_high_n_u32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrn_high_n_u64() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: u64x2 = u64x2::new(32, 36);
        let e: u32x4 = u32x4::new(0, 1, 8, 9);
        let r: u32x4 = transmute(vrshrn_high_n_u64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsrad_n_s64() {
        let a: i64 = 1;
        let b: i64 = 4;
        let e: i64 = 2;
        let r: i64 = transmute(vrsrad_n_s64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsrad_n_u64() {
        let a: u64 = 1;
        let b: u64 = 4;
        let e: u64 = 2;
        let r: u64 = transmute(vrsrad_n_u64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vset_lane_f64() {
        let a: f64 = 1.;
        let b: f64 = 0.;
        let e: f64 = 1.;
        let r: f64 = transmute(vset_lane_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsetq_lane_f64() {
        let a: f64 = 1.;
        let b: f64x2 = f64x2::new(0., 2.);
        let e: f64x2 = f64x2::new(1., 2.);
        let r: f64x2 = transmute(vsetq_lane_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshld_s64() {
        let a: i64 = 1;
        let b: i64 = 2;
        let e: i64 = 4;
        let r: i64 = transmute(vshld_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshld_u64() {
        let a: u64 = 1;
        let b: i64 = 2;
        let e: u64 = 4;
        let r: u64 = transmute(vshld_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshll_high_n_s8() {
        let a: i8x16 = i8x16::new(0, 0, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: i16x8 = transmute(vshll_high_n_s8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshll_high_n_s16() {
        let a: i16x8 = i16x8::new(0, 0, 1, 2, 1, 2, 3, 4);
        let e: i32x4 = i32x4::new(4, 8, 12, 16);
        let r: i32x4 = transmute(vshll_high_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshll_high_n_s32() {
        let a: i32x4 = i32x4::new(0, 0, 1, 2);
        let e: i64x2 = i64x2::new(4, 8);
        let r: i64x2 = transmute(vshll_high_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshll_high_n_u8() {
        let a: u8x16 = u8x16::new(0, 0, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: u16x8 = transmute(vshll_high_n_u8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshll_high_n_u16() {
        let a: u16x8 = u16x8::new(0, 0, 1, 2, 1, 2, 3, 4);
        let e: u32x4 = u32x4::new(4, 8, 12, 16);
        let r: u32x4 = transmute(vshll_high_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshll_high_n_u32() {
        let a: u32x4 = u32x4::new(0, 0, 1, 2);
        let e: u64x2 = u64x2::new(4, 8);
        let r: u64x2 = transmute(vshll_high_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrn_high_n_s16() {
        let a: i8x8 = i8x8::new(1, 2, 5, 6, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(20, 24, 28, 32, 52, 56, 60, 64);
        let e: i8x16 = i8x16::new(1, 2, 5, 6, 5, 6, 7, 8, 5, 6, 7, 8, 13, 14, 15, 16);
        let r: i8x16 = transmute(vshrn_high_n_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrn_high_n_s32() {
        let a: i16x4 = i16x4::new(1, 2, 5, 6);
        let b: i32x4 = i32x4::new(20, 24, 28, 32);
        let e: i16x8 = i16x8::new(1, 2, 5, 6, 5, 6, 7, 8);
        let r: i16x8 = transmute(vshrn_high_n_s32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrn_high_n_s64() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i64x2 = i64x2::new(20, 24);
        let e: i32x4 = i32x4::new(1, 2, 5, 6);
        let r: i32x4 = transmute(vshrn_high_n_s64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrn_high_n_u16() {
        let a: u8x8 = u8x8::new(1, 2, 5, 6, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(20, 24, 28, 32, 52, 56, 60, 64);
        let e: u8x16 = u8x16::new(1, 2, 5, 6, 5, 6, 7, 8, 5, 6, 7, 8, 13, 14, 15, 16);
        let r: u8x16 = transmute(vshrn_high_n_u16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrn_high_n_u32() {
        let a: u16x4 = u16x4::new(1, 2, 5, 6);
        let b: u32x4 = u32x4::new(20, 24, 28, 32);
        let e: u16x8 = u16x8::new(1, 2, 5, 6, 5, 6, 7, 8);
        let r: u16x8 = transmute(vshrn_high_n_u32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrn_high_n_u64() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u64x2 = u64x2::new(20, 24);
        let e: u32x4 = u32x4::new(1, 2, 5, 6);
        let r: u32x4 = transmute(vshrn_high_n_u64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1_s8() {
        let a: i8x8 = i8x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: i8x8 = i8x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: i8x8 = i8x8::new(0, 1, 4, 5, 8, 9, 12, 13);
        let r: i8x8 = transmute(vtrn1_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1q_s8() {
        let a: i8x16 = i8x16::new(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let b: i8x16 = i8x16::new(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        let e: i8x16 = i8x16::new(0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29);
        let r: i8x16 = transmute(vtrn1q_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1_s16() {
        let a: i16x4 = i16x4::new(0, 2, 4, 6);
        let b: i16x4 = i16x4::new(1, 3, 5, 7);
        let e: i16x4 = i16x4::new(0, 1, 4, 5);
        let r: i16x4 = transmute(vtrn1_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1q_s16() {
        let a: i16x8 = i16x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: i16x8 = i16x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: i16x8 = i16x8::new(0, 1, 4, 5, 8, 9, 12, 13);
        let r: i16x8 = transmute(vtrn1q_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1q_s32() {
        let a: i32x4 = i32x4::new(0, 2, 4, 6);
        let b: i32x4 = i32x4::new(1, 3, 5, 7);
        let e: i32x4 = i32x4::new(0, 1, 4, 5);
        let r: i32x4 = transmute(vtrn1q_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1_u8() {
        let a: u8x8 = u8x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: u8x8 = u8x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: u8x8 = u8x8::new(0, 1, 4, 5, 8, 9, 12, 13);
        let r: u8x8 = transmute(vtrn1_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1q_u8() {
        let a: u8x16 = u8x16::new(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let b: u8x16 = u8x16::new(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        let e: u8x16 = u8x16::new(0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29);
        let r: u8x16 = transmute(vtrn1q_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1_u16() {
        let a: u16x4 = u16x4::new(0, 2, 4, 6);
        let b: u16x4 = u16x4::new(1, 3, 5, 7);
        let e: u16x4 = u16x4::new(0, 1, 4, 5);
        let r: u16x4 = transmute(vtrn1_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1q_u16() {
        let a: u16x8 = u16x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: u16x8 = u16x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: u16x8 = u16x8::new(0, 1, 4, 5, 8, 9, 12, 13);
        let r: u16x8 = transmute(vtrn1q_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1q_u32() {
        let a: u32x4 = u32x4::new(0, 2, 4, 6);
        let b: u32x4 = u32x4::new(1, 3, 5, 7);
        let e: u32x4 = u32x4::new(0, 1, 4, 5);
        let r: u32x4 = transmute(vtrn1q_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1_p8() {
        let a: i8x8 = i8x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: i8x8 = i8x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: i8x8 = i8x8::new(0, 1, 4, 5, 8, 9, 12, 13);
        let r: i8x8 = transmute(vtrn1_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1q_p8() {
        let a: i8x16 = i8x16::new(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let b: i8x16 = i8x16::new(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        let e: i8x16 = i8x16::new(0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29);
        let r: i8x16 = transmute(vtrn1q_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1_p16() {
        let a: i16x4 = i16x4::new(0, 2, 4, 6);
        let b: i16x4 = i16x4::new(1, 3, 5, 7);
        let e: i16x4 = i16x4::new(0, 1, 4, 5);
        let r: i16x4 = transmute(vtrn1_p16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1q_p16() {
        let a: i16x8 = i16x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: i16x8 = i16x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: i16x8 = i16x8::new(0, 1, 4, 5, 8, 9, 12, 13);
        let r: i16x8 = transmute(vtrn1q_p16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1_s32() {
        let a: i32x2 = i32x2::new(0, 2);
        let b: i32x2 = i32x2::new(1, 3);
        let e: i32x2 = i32x2::new(0, 1);
        let r: i32x2 = transmute(vtrn1_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1q_s64() {
        let a: i64x2 = i64x2::new(0, 2);
        let b: i64x2 = i64x2::new(1, 3);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vtrn1q_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1_u32() {
        let a: u32x2 = u32x2::new(0, 2);
        let b: u32x2 = u32x2::new(1, 3);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vtrn1_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1q_u64() {
        let a: u64x2 = u64x2::new(0, 2);
        let b: u64x2 = u64x2::new(1, 3);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vtrn1q_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1q_p64() {
        let a: i64x2 = i64x2::new(0, 2);
        let b: i64x2 = i64x2::new(1, 3);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vtrn1q_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1q_f32() {
        let a: f32x4 = f32x4::new(0., 2., 4., 6.);
        let b: f32x4 = f32x4::new(1., 3., 5., 7.);
        let e: f32x4 = f32x4::new(0., 1., 4., 5.);
        let r: f32x4 = transmute(vtrn1q_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1_f32() {
        let a: f32x2 = f32x2::new(0., 2.);
        let b: f32x2 = f32x2::new(1., 3.);
        let e: f32x2 = f32x2::new(0., 1.);
        let r: f32x2 = transmute(vtrn1_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn1q_f64() {
        let a: f64x2 = f64x2::new(0., 2.);
        let b: f64x2 = f64x2::new(1., 3.);
        let e: f64x2 = f64x2::new(0., 1.);
        let r: f64x2 = transmute(vtrn1q_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2_s8() {
        let a: i8x8 = i8x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: i8x8 = i8x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: i8x8 = i8x8::new(2, 3, 6, 7, 10, 11, 14, 15);
        let r: i8x8 = transmute(vtrn2_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2q_s8() {
        let a: i8x16 = i8x16::new(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let b: i8x16 = i8x16::new(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        let e: i8x16 = i8x16::new(2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31);
        let r: i8x16 = transmute(vtrn2q_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2_s16() {
        let a: i16x4 = i16x4::new(0, 2, 4, 6);
        let b: i16x4 = i16x4::new(1, 3, 5, 7);
        let e: i16x4 = i16x4::new(2, 3, 6, 7);
        let r: i16x4 = transmute(vtrn2_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2q_s16() {
        let a: i16x8 = i16x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: i16x8 = i16x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: i16x8 = i16x8::new(2, 3, 6, 7, 10, 11, 14, 15);
        let r: i16x8 = transmute(vtrn2q_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2q_s32() {
        let a: i32x4 = i32x4::new(0, 2, 4, 6);
        let b: i32x4 = i32x4::new(1, 3, 5, 7);
        let e: i32x4 = i32x4::new(2, 3, 6, 7);
        let r: i32x4 = transmute(vtrn2q_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2_u8() {
        let a: u8x8 = u8x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: u8x8 = u8x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: u8x8 = u8x8::new(2, 3, 6, 7, 10, 11, 14, 15);
        let r: u8x8 = transmute(vtrn2_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2q_u8() {
        let a: u8x16 = u8x16::new(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let b: u8x16 = u8x16::new(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        let e: u8x16 = u8x16::new(2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31);
        let r: u8x16 = transmute(vtrn2q_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2_u16() {
        let a: u16x4 = u16x4::new(0, 2, 4, 6);
        let b: u16x4 = u16x4::new(1, 3, 5, 7);
        let e: u16x4 = u16x4::new(2, 3, 6, 7);
        let r: u16x4 = transmute(vtrn2_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2q_u16() {
        let a: u16x8 = u16x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: u16x8 = u16x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: u16x8 = u16x8::new(2, 3, 6, 7, 10, 11, 14, 15);
        let r: u16x8 = transmute(vtrn2q_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2q_u32() {
        let a: u32x4 = u32x4::new(0, 2, 4, 6);
        let b: u32x4 = u32x4::new(1, 3, 5, 7);
        let e: u32x4 = u32x4::new(2, 3, 6, 7);
        let r: u32x4 = transmute(vtrn2q_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2_p8() {
        let a: i8x8 = i8x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: i8x8 = i8x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: i8x8 = i8x8::new(2, 3, 6, 7, 10, 11, 14, 15);
        let r: i8x8 = transmute(vtrn2_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2q_p8() {
        let a: i8x16 = i8x16::new(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let b: i8x16 = i8x16::new(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        let e: i8x16 = i8x16::new(2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31);
        let r: i8x16 = transmute(vtrn2q_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2_p16() {
        let a: i16x4 = i16x4::new(0, 2, 4, 6);
        let b: i16x4 = i16x4::new(1, 3, 5, 7);
        let e: i16x4 = i16x4::new(2, 3, 6, 7);
        let r: i16x4 = transmute(vtrn2_p16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2q_p16() {
        let a: i16x8 = i16x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: i16x8 = i16x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: i16x8 = i16x8::new(2, 3, 6, 7, 10, 11, 14, 15);
        let r: i16x8 = transmute(vtrn2q_p16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2_s32() {
        let a: i32x2 = i32x2::new(0, 2);
        let b: i32x2 = i32x2::new(1, 3);
        let e: i32x2 = i32x2::new(2, 3);
        let r: i32x2 = transmute(vtrn2_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2q_s64() {
        let a: i64x2 = i64x2::new(0, 2);
        let b: i64x2 = i64x2::new(1, 3);
        let e: i64x2 = i64x2::new(2, 3);
        let r: i64x2 = transmute(vtrn2q_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2_u32() {
        let a: u32x2 = u32x2::new(0, 2);
        let b: u32x2 = u32x2::new(1, 3);
        let e: u32x2 = u32x2::new(2, 3);
        let r: u32x2 = transmute(vtrn2_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2q_u64() {
        let a: u64x2 = u64x2::new(0, 2);
        let b: u64x2 = u64x2::new(1, 3);
        let e: u64x2 = u64x2::new(2, 3);
        let r: u64x2 = transmute(vtrn2q_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2q_p64() {
        let a: i64x2 = i64x2::new(0, 2);
        let b: i64x2 = i64x2::new(1, 3);
        let e: i64x2 = i64x2::new(2, 3);
        let r: i64x2 = transmute(vtrn2q_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2q_f32() {
        let a: f32x4 = f32x4::new(0., 2., 4., 6.);
        let b: f32x4 = f32x4::new(1., 3., 5., 7.);
        let e: f32x4 = f32x4::new(2., 3., 6., 7.);
        let r: f32x4 = transmute(vtrn2q_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2_f32() {
        let a: f32x2 = f32x2::new(0., 2.);
        let b: f32x2 = f32x2::new(1., 3.);
        let e: f32x2 = f32x2::new(2., 3.);
        let r: f32x2 = transmute(vtrn2_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtrn2q_f64() {
        let a: f64x2 = f64x2::new(0., 2.);
        let b: f64x2 = f64x2::new(1., 3.);
        let e: f64x2 = f64x2::new(2., 3.);
        let r: f64x2 = transmute(vtrn2q_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1_s8() {
        let a: i8x8 = i8x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: i8x8 = i8x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i8x8 = transmute(vzip1_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1q_s8() {
        let a: i8x16 = i8x16::new(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let b: i8x16 = i8x16::new(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        let e: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: i8x16 = transmute(vzip1q_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1_s16() {
        let a: i16x4 = i16x4::new(0, 2, 4, 6);
        let b: i16x4 = i16x4::new(1, 3, 5, 7);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vzip1_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1q_s16() {
        let a: i16x8 = i16x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: i16x8 = i16x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vzip1q_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1_s32() {
        let a: i32x2 = i32x2::new(0, 2);
        let b: i32x2 = i32x2::new(1, 3);
        let e: i32x2 = i32x2::new(0, 1);
        let r: i32x2 = transmute(vzip1_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1q_s32() {
        let a: i32x4 = i32x4::new(0, 2, 4, 6);
        let b: i32x4 = i32x4::new(1, 3, 5, 7);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vzip1q_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1q_s64() {
        let a: i64x2 = i64x2::new(0, 2);
        let b: i64x2 = i64x2::new(1, 3);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vzip1q_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1_u8() {
        let a: u8x8 = u8x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: u8x8 = u8x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u8x8 = transmute(vzip1_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1q_u8() {
        let a: u8x16 = u8x16::new(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let b: u8x16 = u8x16::new(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        let e: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: u8x16 = transmute(vzip1q_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1_u16() {
        let a: u16x4 = u16x4::new(0, 2, 4, 6);
        let b: u16x4 = u16x4::new(1, 3, 5, 7);
        let e: u16x4 = u16x4::new(0, 1, 2, 3);
        let r: u16x4 = transmute(vzip1_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1q_u16() {
        let a: u16x8 = u16x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: u16x8 = u16x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u16x8 = transmute(vzip1q_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1_u32() {
        let a: u32x2 = u32x2::new(0, 2);
        let b: u32x2 = u32x2::new(1, 3);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vzip1_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1q_u32() {
        let a: u32x4 = u32x4::new(0, 2, 4, 6);
        let b: u32x4 = u32x4::new(1, 3, 5, 7);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vzip1q_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1q_u64() {
        let a: u64x2 = u64x2::new(0, 2);
        let b: u64x2 = u64x2::new(1, 3);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vzip1q_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1_p8() {
        let a: i8x8 = i8x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: i8x8 = i8x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i8x8 = transmute(vzip1_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1q_p8() {
        let a: i8x16 = i8x16::new(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let b: i8x16 = i8x16::new(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        let e: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: i8x16 = transmute(vzip1q_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1_p16() {
        let a: i16x4 = i16x4::new(0, 2, 4, 6);
        let b: i16x4 = i16x4::new(1, 3, 5, 7);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vzip1_p16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1q_p16() {
        let a: i16x8 = i16x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        let b: i16x8 = i16x8::new(1, 3, 5, 7, 9, 11, 13, 15);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vzip1q_p16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1q_p64() {
        let a: i64x2 = i64x2::new(0, 2);
        let b: i64x2 = i64x2::new(1, 3);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vzip1q_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1_f32() {
        let a: f32x2 = f32x2::new(0., 2.);
        let b: f32x2 = f32x2::new(1., 3.);
        let e: f32x2 = f32x2::new(0., 1.);
        let r: f32x2 = transmute(vzip1_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1q_f32() {
        let a: f32x4 = f32x4::new(0., 2., 4., 6.);
        let b: f32x4 = f32x4::new(1., 3., 5., 7.);
        let e: f32x4 = f32x4::new(0., 1., 2., 3.);
        let r: f32x4 = transmute(vzip1q_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip1q_f64() {
        let a: f64x2 = f64x2::new(0., 2.);
        let b: f64x2 = f64x2::new(1., 3.);
        let e: f64x2 = f64x2::new(0., 1.);
        let r: f64x2 = transmute(vzip1q_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2_s8() {
        let a: i8x8 = i8x8::new(0, 16, 16, 18, 16, 18, 20, 22);
        let b: i8x8 = i8x8::new(1, 17, 17, 19, 17, 19, 21, 23);
        let e: i8x8 = i8x8::new(16, 17, 18, 19, 20, 21, 22, 23);
        let r: i8x8 = transmute(vzip2_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2q_s8() {
        let a: i8x16 = i8x16::new(0, 16, 16, 18, 16, 18, 20, 22, 16, 18, 20, 22, 24, 26, 28, 30);
        let b: i8x16 = i8x16::new(1, 17, 17, 19, 17, 19, 21, 23, 17, 19, 21, 23, 25, 27, 29, 31);
        let e: i8x16 = i8x16::new(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r: i8x16 = transmute(vzip2q_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2_s16() {
        let a: i16x4 = i16x4::new(0, 16, 16, 18);
        let b: i16x4 = i16x4::new(1, 17, 17, 19);
        let e: i16x4 = i16x4::new(16, 17, 18, 19);
        let r: i16x4 = transmute(vzip2_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2q_s16() {
        let a: i16x8 = i16x8::new(0, 16, 16, 18, 16, 18, 20, 22);
        let b: i16x8 = i16x8::new(1, 17, 17, 19, 17, 19, 21, 23);
        let e: i16x8 = i16x8::new(16, 17, 18, 19, 20, 21, 22, 23);
        let r: i16x8 = transmute(vzip2q_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2_s32() {
        let a: i32x2 = i32x2::new(0, 16);
        let b: i32x2 = i32x2::new(1, 17);
        let e: i32x2 = i32x2::new(16, 17);
        let r: i32x2 = transmute(vzip2_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2q_s32() {
        let a: i32x4 = i32x4::new(0, 16, 16, 18);
        let b: i32x4 = i32x4::new(1, 17, 17, 19);
        let e: i32x4 = i32x4::new(16, 17, 18, 19);
        let r: i32x4 = transmute(vzip2q_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2q_s64() {
        let a: i64x2 = i64x2::new(0, 16);
        let b: i64x2 = i64x2::new(1, 17);
        let e: i64x2 = i64x2::new(16, 17);
        let r: i64x2 = transmute(vzip2q_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2_u8() {
        let a: u8x8 = u8x8::new(0, 16, 16, 18, 16, 18, 20, 22);
        let b: u8x8 = u8x8::new(1, 17, 17, 19, 17, 19, 21, 23);
        let e: u8x8 = u8x8::new(16, 17, 18, 19, 20, 21, 22, 23);
        let r: u8x8 = transmute(vzip2_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2q_u8() {
        let a: u8x16 = u8x16::new(0, 16, 16, 18, 16, 18, 20, 22, 16, 18, 20, 22, 24, 26, 28, 30);
        let b: u8x16 = u8x16::new(1, 17, 17, 19, 17, 19, 21, 23, 17, 19, 21, 23, 25, 27, 29, 31);
        let e: u8x16 = u8x16::new(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r: u8x16 = transmute(vzip2q_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2_u16() {
        let a: u16x4 = u16x4::new(0, 16, 16, 18);
        let b: u16x4 = u16x4::new(1, 17, 17, 19);
        let e: u16x4 = u16x4::new(16, 17, 18, 19);
        let r: u16x4 = transmute(vzip2_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2q_u16() {
        let a: u16x8 = u16x8::new(0, 16, 16, 18, 16, 18, 20, 22);
        let b: u16x8 = u16x8::new(1, 17, 17, 19, 17, 19, 21, 23);
        let e: u16x8 = u16x8::new(16, 17, 18, 19, 20, 21, 22, 23);
        let r: u16x8 = transmute(vzip2q_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2_u32() {
        let a: u32x2 = u32x2::new(0, 16);
        let b: u32x2 = u32x2::new(1, 17);
        let e: u32x2 = u32x2::new(16, 17);
        let r: u32x2 = transmute(vzip2_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2q_u32() {
        let a: u32x4 = u32x4::new(0, 16, 16, 18);
        let b: u32x4 = u32x4::new(1, 17, 17, 19);
        let e: u32x4 = u32x4::new(16, 17, 18, 19);
        let r: u32x4 = transmute(vzip2q_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2q_u64() {
        let a: u64x2 = u64x2::new(0, 16);
        let b: u64x2 = u64x2::new(1, 17);
        let e: u64x2 = u64x2::new(16, 17);
        let r: u64x2 = transmute(vzip2q_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2_p8() {
        let a: i8x8 = i8x8::new(0, 16, 16, 18, 16, 18, 20, 22);
        let b: i8x8 = i8x8::new(1, 17, 17, 19, 17, 19, 21, 23);
        let e: i8x8 = i8x8::new(16, 17, 18, 19, 20, 21, 22, 23);
        let r: i8x8 = transmute(vzip2_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2q_p8() {
        let a: i8x16 = i8x16::new(0, 16, 16, 18, 16, 18, 20, 22, 16, 18, 20, 22, 24, 26, 28, 30);
        let b: i8x16 = i8x16::new(1, 17, 17, 19, 17, 19, 21, 23, 17, 19, 21, 23, 25, 27, 29, 31);
        let e: i8x16 = i8x16::new(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r: i8x16 = transmute(vzip2q_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2_p16() {
        let a: i16x4 = i16x4::new(0, 16, 16, 18);
        let b: i16x4 = i16x4::new(1, 17, 17, 19);
        let e: i16x4 = i16x4::new(16, 17, 18, 19);
        let r: i16x4 = transmute(vzip2_p16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2q_p16() {
        let a: i16x8 = i16x8::new(0, 16, 16, 18, 16, 18, 20, 22);
        let b: i16x8 = i16x8::new(1, 17, 17, 19, 17, 19, 21, 23);
        let e: i16x8 = i16x8::new(16, 17, 18, 19, 20, 21, 22, 23);
        let r: i16x8 = transmute(vzip2q_p16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2q_p64() {
        let a: i64x2 = i64x2::new(0, 16);
        let b: i64x2 = i64x2::new(1, 17);
        let e: i64x2 = i64x2::new(16, 17);
        let r: i64x2 = transmute(vzip2q_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2_f32() {
        let a: f32x2 = f32x2::new(0., 8.);
        let b: f32x2 = f32x2::new(1., 9.);
        let e: f32x2 = f32x2::new(8., 9.);
        let r: f32x2 = transmute(vzip2_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2q_f32() {
        let a: f32x4 = f32x4::new(0., 8., 8., 10.);
        let b: f32x4 = f32x4::new(1., 9., 9., 11.);
        let e: f32x4 = f32x4::new(8., 9., 10., 11.);
        let r: f32x4 = transmute(vzip2q_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vzip2q_f64() {
        let a: f64x2 = f64x2::new(0., 8.);
        let b: f64x2 = f64x2::new(1., 9.);
        let e: f64x2 = f64x2::new(8., 9.);
        let r: f64x2 = transmute(vzip2q_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1_s8() {
        let a: i8x8 = i8x8::new(1, 0, 2, 0, 2, 0, 3, 0);
        let b: i8x8 = i8x8::new(2, 0, 3, 0, 7, 0, 8, 0);
        let e: i8x8 = i8x8::new(1, 2, 2, 3, 2, 3, 7, 8);
        let r: i8x8 = transmute(vuzp1_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1q_s8() {
        let a: i8x16 = i8x16::new(1, 0, 2, 0, 2, 0, 3, 0, 2, 0, 3, 0, 7, 0, 8, 0);
        let b: i8x16 = i8x16::new(2, 0, 3, 0, 7, 0, 8, 0, 13, 0, 14, 0, 15, 0, 16, 0);
        let e: i8x16 = i8x16::new(1, 2, 2, 3, 2, 3, 7, 8, 2, 3, 7, 8, 13, 14, 15, 16);
        let r: i8x16 = transmute(vuzp1q_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1_s16() {
        let a: i16x4 = i16x4::new(1, 0, 2, 0);
        let b: i16x4 = i16x4::new(2, 0, 3, 0);
        let e: i16x4 = i16x4::new(1, 2, 2, 3);
        let r: i16x4 = transmute(vuzp1_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1q_s16() {
        let a: i16x8 = i16x8::new(1, 0, 2, 0, 2, 0, 3, 0);
        let b: i16x8 = i16x8::new(2, 0, 3, 0, 7, 0, 8, 0);
        let e: i16x8 = i16x8::new(1, 2, 2, 3, 2, 3, 7, 8);
        let r: i16x8 = transmute(vuzp1q_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1q_s32() {
        let a: i32x4 = i32x4::new(1, 0, 2, 0);
        let b: i32x4 = i32x4::new(2, 0, 3, 0);
        let e: i32x4 = i32x4::new(1, 2, 2, 3);
        let r: i32x4 = transmute(vuzp1q_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1_u8() {
        let a: u8x8 = u8x8::new(1, 0, 2, 0, 2, 0, 3, 0);
        let b: u8x8 = u8x8::new(2, 0, 3, 0, 7, 0, 8, 0);
        let e: u8x8 = u8x8::new(1, 2, 2, 3, 2, 3, 7, 8);
        let r: u8x8 = transmute(vuzp1_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1q_u8() {
        let a: u8x16 = u8x16::new(1, 0, 2, 0, 2, 0, 3, 0, 2, 0, 3, 0, 7, 0, 8, 0);
        let b: u8x16 = u8x16::new(2, 0, 3, 0, 7, 0, 8, 0, 13, 0, 14, 0, 15, 0, 16, 0);
        let e: u8x16 = u8x16::new(1, 2, 2, 3, 2, 3, 7, 8, 2, 3, 7, 8, 13, 14, 15, 16);
        let r: u8x16 = transmute(vuzp1q_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1_u16() {
        let a: u16x4 = u16x4::new(1, 0, 2, 0);
        let b: u16x4 = u16x4::new(2, 0, 3, 0);
        let e: u16x4 = u16x4::new(1, 2, 2, 3);
        let r: u16x4 = transmute(vuzp1_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1q_u16() {
        let a: u16x8 = u16x8::new(1, 0, 2, 0, 2, 0, 3, 0);
        let b: u16x8 = u16x8::new(2, 0, 3, 0, 7, 0, 8, 0);
        let e: u16x8 = u16x8::new(1, 2, 2, 3, 2, 3, 7, 8);
        let r: u16x8 = transmute(vuzp1q_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1q_u32() {
        let a: u32x4 = u32x4::new(1, 0, 2, 0);
        let b: u32x4 = u32x4::new(2, 0, 3, 0);
        let e: u32x4 = u32x4::new(1, 2, 2, 3);
        let r: u32x4 = transmute(vuzp1q_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1_p8() {
        let a: i8x8 = i8x8::new(1, 0, 2, 0, 2, 0, 3, 0);
        let b: i8x8 = i8x8::new(2, 0, 3, 0, 7, 0, 8, 0);
        let e: i8x8 = i8x8::new(1, 2, 2, 3, 2, 3, 7, 8);
        let r: i8x8 = transmute(vuzp1_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1q_p8() {
        let a: i8x16 = i8x16::new(1, 0, 2, 0, 2, 0, 3, 0, 2, 0, 3, 0, 7, 0, 8, 0);
        let b: i8x16 = i8x16::new(2, 0, 3, 0, 7, 0, 8, 0, 13, 0, 14, 0, 15, 0, 16, 0);
        let e: i8x16 = i8x16::new(1, 2, 2, 3, 2, 3, 7, 8, 2, 3, 7, 8, 13, 14, 15, 16);
        let r: i8x16 = transmute(vuzp1q_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1_p16() {
        let a: i16x4 = i16x4::new(1, 0, 2, 0);
        let b: i16x4 = i16x4::new(2, 0, 3, 0);
        let e: i16x4 = i16x4::new(1, 2, 2, 3);
        let r: i16x4 = transmute(vuzp1_p16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1q_p16() {
        let a: i16x8 = i16x8::new(1, 0, 2, 0, 2, 0, 3, 0);
        let b: i16x8 = i16x8::new(2, 0, 3, 0, 7, 0, 8, 0);
        let e: i16x8 = i16x8::new(1, 2, 2, 3, 2, 3, 7, 8);
        let r: i16x8 = transmute(vuzp1q_p16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1_s32() {
        let a: i32x2 = i32x2::new(1, 0);
        let b: i32x2 = i32x2::new(2, 0);
        let e: i32x2 = i32x2::new(1, 2);
        let r: i32x2 = transmute(vuzp1_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1q_s64() {
        let a: i64x2 = i64x2::new(1, 0);
        let b: i64x2 = i64x2::new(2, 0);
        let e: i64x2 = i64x2::new(1, 2);
        let r: i64x2 = transmute(vuzp1q_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1_u32() {
        let a: u32x2 = u32x2::new(1, 0);
        let b: u32x2 = u32x2::new(2, 0);
        let e: u32x2 = u32x2::new(1, 2);
        let r: u32x2 = transmute(vuzp1_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1q_u64() {
        let a: u64x2 = u64x2::new(1, 0);
        let b: u64x2 = u64x2::new(2, 0);
        let e: u64x2 = u64x2::new(1, 2);
        let r: u64x2 = transmute(vuzp1q_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1q_p64() {
        let a: i64x2 = i64x2::new(1, 0);
        let b: i64x2 = i64x2::new(2, 0);
        let e: i64x2 = i64x2::new(1, 2);
        let r: i64x2 = transmute(vuzp1q_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1q_f32() {
        let a: f32x4 = f32x4::new(0., 8., 1., 9.);
        let b: f32x4 = f32x4::new(1., 10., 3., 11.);
        let e: f32x4 = f32x4::new(0., 1., 1., 3.);
        let r: f32x4 = transmute(vuzp1q_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1_f32() {
        let a: f32x2 = f32x2::new(0., 8.);
        let b: f32x2 = f32x2::new(1., 10.);
        let e: f32x2 = f32x2::new(0., 1.);
        let r: f32x2 = transmute(vuzp1_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp1q_f64() {
        let a: f64x2 = f64x2::new(0., 8.);
        let b: f64x2 = f64x2::new(1., 10.);
        let e: f64x2 = f64x2::new(0., 1.);
        let r: f64x2 = transmute(vuzp1q_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2_s8() {
        let a: i8x8 = i8x8::new(0, 17, 0, 18, 0, 18, 0, 19);
        let b: i8x8 = i8x8::new(0, 18, 0, 19, 0, 23, 0, 24);
        let e: i8x8 = i8x8::new(17, 18, 18, 19, 18, 19, 23, 24);
        let r: i8x8 = transmute(vuzp2_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2q_s8() {
        let a: i8x16 = i8x16::new(0, 17, 0, 18, 0, 18, 0, 19, 0, 18, 0, 19, 0, 23, 0, 24);
        let b: i8x16 = i8x16::new(0, 18, 0, 19, 0, 23, 0, 24, 0, 29, 0, 30, 0, 31, 0, 32);
        let e: i8x16 = i8x16::new(17, 18, 18, 19, 18, 19, 23, 24, 18, 19, 23, 24, 29, 30, 31, 32);
        let r: i8x16 = transmute(vuzp2q_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2_s16() {
        let a: i16x4 = i16x4::new(0, 17, 0, 18);
        let b: i16x4 = i16x4::new(0, 18, 0, 19);
        let e: i16x4 = i16x4::new(17, 18, 18, 19);
        let r: i16x4 = transmute(vuzp2_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2q_s16() {
        let a: i16x8 = i16x8::new(0, 17, 0, 18, 0, 18, 0, 19);
        let b: i16x8 = i16x8::new(0, 18, 0, 19, 0, 23, 0, 24);
        let e: i16x8 = i16x8::new(17, 18, 18, 19, 18, 19, 23, 24);
        let r: i16x8 = transmute(vuzp2q_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2q_s32() {
        let a: i32x4 = i32x4::new(0, 17, 0, 18);
        let b: i32x4 = i32x4::new(0, 18, 0, 19);
        let e: i32x4 = i32x4::new(17, 18, 18, 19);
        let r: i32x4 = transmute(vuzp2q_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2_u8() {
        let a: u8x8 = u8x8::new(0, 17, 0, 18, 0, 18, 0, 19);
        let b: u8x8 = u8x8::new(0, 18, 0, 19, 0, 23, 0, 24);
        let e: u8x8 = u8x8::new(17, 18, 18, 19, 18, 19, 23, 24);
        let r: u8x8 = transmute(vuzp2_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2q_u8() {
        let a: u8x16 = u8x16::new(0, 17, 0, 18, 0, 18, 0, 19, 0, 18, 0, 19, 0, 23, 0, 24);
        let b: u8x16 = u8x16::new(0, 18, 0, 19, 0, 23, 0, 24, 0, 29, 0, 30, 0, 31, 0, 32);
        let e: u8x16 = u8x16::new(17, 18, 18, 19, 18, 19, 23, 24, 18, 19, 23, 24, 29, 30, 31, 32);
        let r: u8x16 = transmute(vuzp2q_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2_u16() {
        let a: u16x4 = u16x4::new(0, 17, 0, 18);
        let b: u16x4 = u16x4::new(0, 18, 0, 19);
        let e: u16x4 = u16x4::new(17, 18, 18, 19);
        let r: u16x4 = transmute(vuzp2_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2q_u16() {
        let a: u16x8 = u16x8::new(0, 17, 0, 18, 0, 18, 0, 19);
        let b: u16x8 = u16x8::new(0, 18, 0, 19, 0, 23, 0, 24);
        let e: u16x8 = u16x8::new(17, 18, 18, 19, 18, 19, 23, 24);
        let r: u16x8 = transmute(vuzp2q_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2q_u32() {
        let a: u32x4 = u32x4::new(0, 17, 0, 18);
        let b: u32x4 = u32x4::new(0, 18, 0, 19);
        let e: u32x4 = u32x4::new(17, 18, 18, 19);
        let r: u32x4 = transmute(vuzp2q_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2_p8() {
        let a: i8x8 = i8x8::new(0, 17, 0, 18, 0, 18, 0, 19);
        let b: i8x8 = i8x8::new(0, 18, 0, 19, 0, 23, 0, 24);
        let e: i8x8 = i8x8::new(17, 18, 18, 19, 18, 19, 23, 24);
        let r: i8x8 = transmute(vuzp2_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2q_p8() {
        let a: i8x16 = i8x16::new(0, 17, 0, 18, 0, 18, 0, 19, 0, 18, 0, 19, 0, 23, 0, 24);
        let b: i8x16 = i8x16::new(0, 18, 0, 19, 0, 23, 0, 24, 0, 29, 0, 30, 0, 31, 0, 32);
        let e: i8x16 = i8x16::new(17, 18, 18, 19, 18, 19, 23, 24, 18, 19, 23, 24, 29, 30, 31, 32);
        let r: i8x16 = transmute(vuzp2q_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2_p16() {
        let a: i16x4 = i16x4::new(0, 17, 0, 18);
        let b: i16x4 = i16x4::new(0, 18, 0, 19);
        let e: i16x4 = i16x4::new(17, 18, 18, 19);
        let r: i16x4 = transmute(vuzp2_p16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2q_p16() {
        let a: i16x8 = i16x8::new(0, 17, 0, 18, 0, 18, 0, 19);
        let b: i16x8 = i16x8::new(0, 18, 0, 19, 0, 23, 0, 24);
        let e: i16x8 = i16x8::new(17, 18, 18, 19, 18, 19, 23, 24);
        let r: i16x8 = transmute(vuzp2q_p16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2_s32() {
        let a: i32x2 = i32x2::new(0, 17);
        let b: i32x2 = i32x2::new(0, 18);
        let e: i32x2 = i32x2::new(17, 18);
        let r: i32x2 = transmute(vuzp2_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2q_s64() {
        let a: i64x2 = i64x2::new(0, 17);
        let b: i64x2 = i64x2::new(0, 18);
        let e: i64x2 = i64x2::new(17, 18);
        let r: i64x2 = transmute(vuzp2q_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2_u32() {
        let a: u32x2 = u32x2::new(0, 17);
        let b: u32x2 = u32x2::new(0, 18);
        let e: u32x2 = u32x2::new(17, 18);
        let r: u32x2 = transmute(vuzp2_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2q_u64() {
        let a: u64x2 = u64x2::new(0, 17);
        let b: u64x2 = u64x2::new(0, 18);
        let e: u64x2 = u64x2::new(17, 18);
        let r: u64x2 = transmute(vuzp2q_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2q_p64() {
        let a: i64x2 = i64x2::new(0, 17);
        let b: i64x2 = i64x2::new(0, 18);
        let e: i64x2 = i64x2::new(17, 18);
        let r: i64x2 = transmute(vuzp2q_p64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2q_f32() {
        let a: f32x4 = f32x4::new(0., 8., 1., 9.);
        let b: f32x4 = f32x4::new(2., 9., 3., 11.);
        let e: f32x4 = f32x4::new(8., 9., 9., 11.);
        let r: f32x4 = transmute(vuzp2q_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2_f32() {
        let a: f32x2 = f32x2::new(0., 8.);
        let b: f32x2 = f32x2::new(2., 9.);
        let e: f32x2 = f32x2::new(8., 9.);
        let r: f32x2 = transmute(vuzp2_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vuzp2q_f64() {
        let a: f64x2 = f64x2::new(0., 8.);
        let b: f64x2 = f64x2::new(2., 9.);
        let e: f64x2 = f64x2::new(8., 9.);
        let r: f64x2 = transmute(vuzp2q_f64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabal_high_u8() {
        let a: u16x8 = u16x8::new(9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let c: u8x16 = u8x16::new(10, 10, 10, 10, 10, 10, 10, 10, 20, 0, 2, 4, 6, 8, 10, 12);
        let e: u16x8 = u16x8::new(20, 20, 20, 20, 20, 20, 20, 20);
        let r: u16x8 = transmute(vabal_high_u8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabal_high_u16() {
        let a: u32x4 = u32x4::new(9, 10, 11, 12);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 9, 10, 11, 12);
        let c: u16x8 = u16x8::new(10, 10, 10, 10, 20, 0, 2, 4);
        let e: u32x4 = u32x4::new(20, 20, 20, 20);
        let r: u32x4 = transmute(vabal_high_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabal_high_u32() {
        let a: u64x2 = u64x2::new(15, 16);
        let b: u32x4 = u32x4::new(1, 2, 15, 16);
        let c: u32x4 = u32x4::new(10, 10, 10, 12);
        let e: u64x2 = u64x2::new(20, 20);
        let r: u64x2 = transmute(vabal_high_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabal_high_s8() {
        let a: i16x8 = i16x8::new(9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let c: i8x16 = i8x16::new(10, 10, 10, 10, 10, 10, 10, 10, 20, 0, 2, 4, 6, 8, 10, 12);
        let e: i16x8 = i16x8::new(20, 20, 20, 20, 20, 20, 20, 20);
        let r: i16x8 = transmute(vabal_high_s8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabal_high_s16() {
        let a: i32x4 = i32x4::new(9, 10, 11, 12);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 9, 10, 11, 12);
        let c: i16x8 = i16x8::new(10, 10, 10, 10, 20, 0, 2, 4);
        let e: i32x4 = i32x4::new(20, 20, 20, 20);
        let r: i32x4 = transmute(vabal_high_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabal_high_s32() {
        let a: i64x2 = i64x2::new(15, 16);
        let b: i32x4 = i32x4::new(1, 2, 15, 16);
        let c: i32x4 = i32x4::new(10, 10, 10, 12);
        let e: i64x2 = i64x2::new(20, 20);
        let r: i64x2 = transmute(vabal_high_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqabs_s64() {
        let a: i64x1 = i64x1::new(-9223372036854775808);
        let e: i64x1 = i64x1::new(0x7F_FF_FF_FF_FF_FF_FF_FF);
        let r: i64x1 = transmute(vqabs_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqabsq_s64() {
        let a: i64x2 = i64x2::new(-9223372036854775808, -7);
        let e: i64x2 = i64x2::new(0x7F_FF_FF_FF_FF_FF_FF_FF, 7);
        let r: i64x2 = transmute(vqabsq_s64(transmute(a)));
        assert_eq!(r, e);
    }
}
