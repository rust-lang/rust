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
    let b: float32x2_t = simd_shuffle2(a, a, [2, 3]);
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
    simd_shuffle4(a, simd_cast(b), [0, 1, 2, 3])
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
    simd_shuffle4(a, vcvtx_f32_f64(b), [0, 1, 2, 3])
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
}
