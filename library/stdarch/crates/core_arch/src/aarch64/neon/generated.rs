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
        fn vabd_f64_(a: float64x1_t, a: float64x1_t) -> float64x1_t;
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
        fn vabdq_f64_(a: float64x2_t, a: float64x2_t) -> float64x2_t;
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
        fn vmax_f64_(a: float64x1_t, a: float64x1_t) -> float64x1_t;
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
        fn vmaxq_f64_(a: float64x2_t, a: float64x2_t) -> float64x2_t;
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
        fn vmin_f64_(a: float64x1_t, a: float64x1_t) -> float64x1_t;
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
        fn vminq_f64_(a: float64x2_t, a: float64x2_t) -> float64x2_t;
    }
    vminq_f64_(a, b)
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
}
