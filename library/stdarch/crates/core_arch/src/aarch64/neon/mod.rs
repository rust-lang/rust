//! ARMv8 ASIMD intrinsics

#![allow(non_camel_case_types)]

#[rustfmt::skip]
mod generated;
#[rustfmt::skip]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub use self::generated::*;

// FIXME: replace neon with asimd

use crate::{
    core_arch::{arm_shared::*, simd::*},
    hint::unreachable_unchecked,
    intrinsics::{simd::*, *},
    mem::transmute,
};
#[cfg(test)]
use stdarch_test::assert_instr;

types! {
    #![stable(feature = "neon_intrinsics", since = "1.59.0")]

    /// ARM-specific 64-bit wide vector of one packed `f64`.
    pub struct float64x1_t(1 x f64); // FIXME: check this!
    /// ARM-specific 128-bit wide vector of two packed `f64`.
    pub struct float64x2_t(2 x f64);
}

/// ARM-specific type containing two `float64x1_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub struct float64x1x2_t(pub float64x1_t, pub float64x1_t);
/// ARM-specific type containing three `float64x1_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub struct float64x1x3_t(pub float64x1_t, pub float64x1_t, pub float64x1_t);
/// ARM-specific type containing four `float64x1_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub struct float64x1x4_t(
    pub float64x1_t,
    pub float64x1_t,
    pub float64x1_t,
    pub float64x1_t,
);

/// ARM-specific type containing two `float64x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub struct float64x2x2_t(pub float64x2_t, pub float64x2_t);
/// ARM-specific type containing three `float64x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub struct float64x2x3_t(pub float64x2_t, pub float64x2_t, pub float64x2_t);
/// ARM-specific type containing four `float64x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub struct float64x2x4_t(
    pub float64x2_t,
    pub float64x2_t,
    pub float64x2_t,
    pub float64x2_t,
);

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N1 = 0, N2 = 0))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vcopy_lane_s64<const N1: i32, const N2: i32>(_a: int64x1_t, b: int64x1_t) -> int64x1_t {
    static_assert!(N1 == 0);
    static_assert!(N2 == 0);
    b
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N1 = 0, N2 = 0))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vcopy_lane_u64<const N1: i32, const N2: i32>(_a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    static_assert!(N1 == 0);
    static_assert!(N2 == 0);
    b
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N1 = 0, N2 = 0))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vcopy_lane_p64<const N1: i32, const N2: i32>(_a: poly64x1_t, b: poly64x1_t) -> poly64x1_t {
    static_assert!(N1 == 0);
    static_assert!(N2 == 0);
    b
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N1 = 0, N2 = 0))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vcopy_lane_f64<const N1: i32, const N2: i32>(
    _a: float64x1_t,
    b: float64x1_t,
) -> float64x1_t {
    static_assert!(N1 == 0);
    static_assert!(N2 == 0);
    b
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vcopy_laneq_s64<const LANE1: i32, const LANE2: i32>(
    _a: int64x1_t,
    b: int64x2_t,
) -> int64x1_t {
    static_assert!(LANE1 == 0);
    static_assert_uimm_bits!(LANE2, 1);
    unsafe { transmute::<i64, _>(simd_extract!(b, LANE2 as u32)) }
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vcopy_laneq_u64<const LANE1: i32, const LANE2: i32>(
    _a: uint64x1_t,
    b: uint64x2_t,
) -> uint64x1_t {
    static_assert!(LANE1 == 0);
    static_assert_uimm_bits!(LANE2, 1);
    unsafe { transmute::<u64, _>(simd_extract!(b, LANE2 as u32)) }
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vcopy_laneq_p64<const LANE1: i32, const LANE2: i32>(
    _a: poly64x1_t,
    b: poly64x2_t,
) -> poly64x1_t {
    static_assert!(LANE1 == 0);
    static_assert_uimm_bits!(LANE2, 1);
    unsafe { transmute::<u64, _>(simd_extract!(b, LANE2 as u32)) }
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vcopy_laneq_f64<const LANE1: i32, const LANE2: i32>(
    _a: float64x1_t,
    b: float64x2_t,
) -> float64x1_t {
    static_assert!(LANE1 == 0);
    static_assert_uimm_bits!(LANE2, 1);
    unsafe { transmute::<f64, _>(simd_extract!(b, LANE2 as u32)) }
}

/// Load multiple single-element structures to one, two, three, or four registers
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ldr))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld1_dup_f64(ptr: *const f64) -> float64x1_t {
    vld1_f64(ptr)
}

/// Load multiple single-element structures to one, two, three, or four registers
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld1r))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld1q_dup_f64(ptr: *const f64) -> float64x2_t {
    let x = vld1q_lane_f64::<0>(ptr, transmute(f64x2::splat(0.)));
    simd_shuffle!(x, x, [0, 0])
}

/// Load one single-element structure to one lane of one register.
#[inline]
#[target_feature(enable = "neon")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(ldr, LANE = 0))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld1_lane_f64<const LANE: i32>(ptr: *const f64, src: float64x1_t) -> float64x1_t {
    static_assert!(LANE == 0);
    simd_insert!(src, LANE as u32, *ptr)
}

/// Load one single-element structure to one lane of one register.
#[inline]
#[target_feature(enable = "neon")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(ld1, LANE = 1))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld1q_lane_f64<const LANE: i32>(ptr: *const f64, src: float64x2_t) -> float64x2_t {
    static_assert_uimm_bits!(LANE, 1);
    simd_insert!(src, LANE as u32, *ptr)
}

/// Bitwise Select instructions. This instruction sets each bit in the destination SIMD&FP register
/// to the corresponding bit from the first source SIMD&FP register when the original
/// destination bit was 1, otherwise from the second source SIMD&FP register.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(bsl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vbsl_f64(a: uint64x1_t, b: float64x1_t, c: float64x1_t) -> float64x1_t {
    let not = int64x1_t::splat(-1);
    unsafe {
        transmute(simd_or(
            simd_and(a, transmute(b)),
            simd_and(simd_xor(a, transmute(not)), transmute(c)),
        ))
    }
}
/// Bitwise Select.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(bsl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vbsl_p64(a: poly64x1_t, b: poly64x1_t, c: poly64x1_t) -> poly64x1_t {
    let not = int64x1_t::splat(-1);
    unsafe { simd_or(simd_and(a, b), simd_and(simd_xor(a, transmute(not)), c)) }
}
/// Bitwise Select. (128-bit)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(bsl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vbslq_f64(a: uint64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    let not = int64x2_t::splat(-1);
    unsafe {
        transmute(simd_or(
            simd_and(a, transmute(b)),
            simd_and(simd_xor(a, transmute(not)), transmute(c)),
        ))
    }
}
/// Bitwise Select. (128-bit)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(bsl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vbslq_p64(a: poly64x2_t, b: poly64x2_t, c: poly64x2_t) -> poly64x2_t {
    let not = int64x2_t::splat(-1);
    unsafe { simd_or(simd_and(a, b), simd_and(simd_xor(a, transmute(not)), c)) }
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fadd))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vadd_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    unsafe { simd_add(a, b) }
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fadd))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vaddq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    unsafe { simd_add(a, b) }
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(add))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vadd_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    unsafe { simd_add(a, b) }
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(add))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vadd_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    unsafe { simd_add(a, b) }
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(add))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vaddd_s64(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}

/// Vector add.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(add))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vaddd_u64(a: u64, b: u64) -> u64 {
    a.wrapping_add(b)
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vext_p64<const N: i32>(a: poly64x1_t, _b: poly64x1_t) -> poly64x1_t {
    static_assert!(N == 0);
    a
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vext_f64<const N: i32>(a: float64x1_t, _b: float64x1_t) -> float64x1_t {
    static_assert!(N == 0);
    a
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmov))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vdup_n_p64(value: p64) -> poly64x1_t {
    unsafe { transmute(u64x1::new(value)) }
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vdup_n_f64(value: f64) -> float64x1_t {
    float64x1_t::splat(value)
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(dup))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vdupq_n_p64(value: p64) -> poly64x2_t {
    unsafe { transmute(u64x2::new(value, value)) }
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(dup))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vdupq_n_f64(value: f64) -> float64x2_t {
    float64x2_t::splat(value)
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmov))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vmov_n_p64(value: p64) -> poly64x1_t {
    vdup_n_p64(value)
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vmov_n_f64(value: f64) -> float64x1_t {
    vdup_n_f64(value)
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(dup))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vmovq_n_p64(value: p64) -> poly64x2_t {
    vdupq_n_p64(value)
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(dup))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vmovq_n_f64(value: f64) -> float64x2_t {
    vdupq_n_f64(value)
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vget_high_f64(a: float64x2_t) -> float64x1_t {
    unsafe { float64x1_t([simd_extract!(a, 1)]) }
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ext))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vget_high_p64(a: poly64x2_t) -> poly64x1_t {
    unsafe { transmute(u64x1::new(simd_extract!(a, 1))) }
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vget_low_f64(a: float64x2_t) -> float64x1_t {
    unsafe { float64x1_t([simd_extract!(a, 0)]) }
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vget_low_p64(a: poly64x2_t) -> poly64x1_t {
    unsafe { transmute(u64x1::new(simd_extract!(a, 0))) }
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(
    all(test, any(target_arch = "aarch64", target_arch = "arm64ec")),
    assert_instr(nop, IMM5 = 0)
)]
pub fn vget_lane_f64<const IMM5: i32>(v: float64x1_t) -> f64 {
    static_assert!(IMM5 == 0);
    unsafe { simd_extract!(v, IMM5 as u32) }
}

/// Duplicate vector element to vector or scalar
#[inline]
#[target_feature(enable = "neon")]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(
    all(test, any(target_arch = "aarch64", target_arch = "arm64ec")),
    assert_instr(nop, IMM5 = 0)
)]
pub fn vgetq_lane_f64<const IMM5: i32>(v: float64x2_t) -> f64 {
    static_assert_uimm_bits!(IMM5, 1);
    unsafe { simd_extract!(v, IMM5 as u32) }
}

/// Vector combine
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vcombine_f64(low: float64x1_t, high: float64x1_t) -> float64x2_t {
    unsafe { simd_shuffle!(low, high, [0, 1]) }
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vshld_n_s64<const N: i32>(a: i64) -> i64 {
    static_assert_uimm_bits!(N, 6);
    a << N
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vshld_n_u64<const N: i32>(a: u64) -> u64 {
    static_assert_uimm_bits!(N, 6);
    a << N
}

/// Signed shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vshrd_n_s64<const N: i32>(a: i64) -> i64 {
    static_assert!(N >= 1 && N <= 64);
    let n: i32 = if N == 64 { 63 } else { N };
    a >> n
}

/// Unsigned shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vshrd_n_u64<const N: i32>(a: u64) -> u64 {
    static_assert!(N >= 1 && N <= 64);
    let n: i32 = if N == 64 {
        return 0;
    } else {
        N
    };
    a >> n
}

/// Signed shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vsrad_n_s64<const N: i32>(a: i64, b: i64) -> i64 {
    static_assert!(N >= 1 && N <= 64);
    a.wrapping_add(vshrd_n_s64::<N>(b))
}

/// Unsigned shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub fn vsrad_n_u64<const N: i32>(a: u64, b: u64) -> u64 {
    static_assert!(N >= 1 && N <= 64);
    a.wrapping_add(vshrd_n_u64::<N>(b))
}

#[cfg(test)]
mod tests {
    use crate::core_arch::aarch64::test_support::*;
    use crate::core_arch::arm_shared::test_support::*;
    use crate::core_arch::{aarch64::neon::*, aarch64::*, simd::*};
    use std::mem::transmute;
    use stdarch_test::simd_test;

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
        let r: i64 = vaddd_s64(a, b);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddd_u64() {
        let a = 1_u64;
        let b = 8_u64;
        let e = 9_u64;
        let r: u64 = vaddd_u64(a, b);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vext_p64() {
        let a: i64x1 = i64x1::new(0);
        let b: i64x1 = i64x1::new(1);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vext_p64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vext_f64() {
        let a: f64x1 = f64x1::new(0.);
        let b: f64x1 = f64x1::new(1.);
        let e: f64x1 = f64x1::new(0.);
        let r: f64x1 = transmute(vext_f64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshld_n_s64() {
        let a: i64 = 1;
        let e: i64 = 4;
        let r: i64 = vshld_n_s64::<2>(a);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshld_n_u64() {
        let a: u64 = 1;
        let e: u64 = 4;
        let r: u64 = vshld_n_u64::<2>(a);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrd_n_s64() {
        let a: i64 = 4;
        let e: i64 = 1;
        let r: i64 = vshrd_n_s64::<2>(a);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrd_n_u64() {
        let a: u64 = 4;
        let e: u64 = 1;
        let r: u64 = vshrd_n_u64::<2>(a);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsrad_n_s64() {
        let a: i64 = 1;
        let b: i64 = 4;
        let e: i64 = 2;
        let r: i64 = vsrad_n_s64::<2>(a, b);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsrad_n_u64() {
        let a: u64 = 1;
        let b: u64 = 4;
        let e: u64 = 2;
        let r: u64 = vsrad_n_u64::<2>(a, b);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_n_f64() {
        let a: f64 = 3.3;
        let e = f64x1::new(3.3);
        let r: f64x1 = transmute(vdup_n_f64(a));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_n_p64() {
        let a: u64 = 3;
        let e = u64x1::new(3);
        let r: u64x1 = transmute(vdup_n_p64(a));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_f64() {
        let a: f64 = 3.3;
        let e = f64x2::new(3.3, 3.3);
        let r: f64x2 = transmute(vdupq_n_f64(a));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_p64() {
        let a: u64 = 3;
        let e = u64x2::new(3, 3);
        let r: u64x2 = transmute(vdupq_n_p64(a));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmov_n_p64() {
        let a: u64 = 3;
        let e = u64x1::new(3);
        let r: u64x1 = transmute(vmov_n_p64(a));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmov_n_f64() {
        let a: f64 = 3.3;
        let e = f64x1::new(3.3);
        let r: f64x1 = transmute(vmov_n_f64(a));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovq_n_p64() {
        let a: u64 = 3;
        let e = u64x2::new(3, 3);
        let r: u64x2 = transmute(vmovq_n_p64(a));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovq_n_f64() {
        let a: f64 = 3.3;
        let e = f64x2::new(3.3, 3.3);
        let r: f64x2 = transmute(vmovq_n_f64(a));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_high_f64() {
        let a = f64x2::new(1.0, 2.0);
        let e = f64x1::new(2.0);
        let r: f64x1 = transmute(vget_high_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_high_p64() {
        let a = u64x2::new(1, 2);
        let e = u64x1::new(2);
        let r: u64x1 = transmute(vget_high_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_low_f64() {
        let a = f64x2::new(1.0, 2.0);
        let e = f64x1::new(1.0);
        let r: f64x1 = transmute(vget_low_f64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_low_p64() {
        let a = u64x2::new(1, 2);
        let e = u64x1::new(1);
        let r: u64x1 = transmute(vget_low_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_f64() {
        let v = f64x1::new(1.0);
        let r = vget_lane_f64::<0>(transmute(v));
        assert_eq!(r, 1.0);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_f64() {
        let v = f64x2::new(0.0, 1.0);
        let r = vgetq_lane_f64::<1>(transmute(v));
        assert_eq!(r, 1.0);
        let r = vgetq_lane_f64::<0>(transmute(v));
        assert_eq!(r, 0.0);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_lane_s64() {
        let a: i64x1 = i64x1::new(1);
        let b: i64x1 = i64x1::new(0x7F_FF_FF_FF_FF_FF_FF_FF);
        let e: i64x1 = i64x1::new(0x7F_FF_FF_FF_FF_FF_FF_FF);
        let r: i64x1 = transmute(vcopy_lane_s64::<0, 0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_lane_u64() {
        let a: u64x1 = u64x1::new(1);
        let b: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcopy_lane_u64::<0, 0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_lane_p64() {
        let a: i64x1 = i64x1::new(1);
        let b: i64x1 = i64x1::new(0x7F_FF_FF_FF_FF_FF_FF_FF);
        let e: i64x1 = i64x1::new(0x7F_FF_FF_FF_FF_FF_FF_FF);
        let r: i64x1 = transmute(vcopy_lane_p64::<0, 0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_lane_f64() {
        let a: f64 = 1.;
        let b: f64 = 0.;
        let e: f64 = 0.;
        let r: f64 = transmute(vcopy_lane_f64::<0, 0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_laneq_s64() {
        let a: i64x1 = i64x1::new(1);
        let b: i64x2 = i64x2::new(0, 0x7F_FF_FF_FF_FF_FF_FF_FF);
        let e: i64x1 = i64x1::new(0x7F_FF_FF_FF_FF_FF_FF_FF);
        let r: i64x1 = transmute(vcopy_laneq_s64::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_laneq_u64() {
        let a: u64x1 = u64x1::new(1);
        let b: u64x2 = u64x2::new(0, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let e: u64x1 = u64x1::new(0xFF_FF_FF_FF_FF_FF_FF_FF);
        let r: u64x1 = transmute(vcopy_laneq_u64::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_laneq_p64() {
        let a: i64x1 = i64x1::new(1);
        let b: i64x2 = i64x2::new(0, 0x7F_FF_FF_FF_FF_FF_FF_FF);
        let e: i64x1 = i64x1::new(0x7F_FF_FF_FF_FF_FF_FF_FF);
        let r: i64x1 = transmute(vcopy_laneq_p64::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcopy_laneq_f64() {
        let a: f64 = 1.;
        let b: f64x2 = f64x2::new(0., 0.5);
        let e: f64 = 0.5;
        let r: f64 = transmute(vcopy_laneq_f64::<0, 1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbsl_f64() {
        let a = u64x1::new(0x8000000000000000);
        let b = f64x1::new(-1.23f64);
        let c = f64x1::new(2.34f64);
        let e = f64x1::new(-2.34f64);
        let r: f64x1 = transmute(vbsl_f64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbsl_p64() {
        let a = u64x1::new(1);
        let b = u64x1::new(u64::MAX);
        let c = u64x1::new(u64::MIN);
        let e = u64x1::new(1);
        let r: u64x1 = transmute(vbsl_p64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbslq_f64() {
        let a = u64x2::new(1, 0x8000000000000000);
        let b = f64x2::new(f64::MAX, -1.23f64);
        let c = f64x2::new(f64::MIN, 2.34f64);
        let e = f64x2::new(f64::MIN, -2.34f64);
        let r: f64x2 = transmute(vbslq_f64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbslq_p64() {
        let a = u64x2::new(u64::MAX, 1);
        let b = u64x2::new(u64::MAX, u64::MAX);
        let c = u64x2::new(u64::MIN, u64::MIN);
        let e = u64x2::new(u64::MAX, 1);
        let r: u64x2 = transmute(vbslq_p64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_f64() {
        let a: [f64; 2] = [0., 1.];
        let e = f64x1::new(1.);
        let r: f64x1 = transmute(vld1_f64(a[1..].as_ptr()));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_f64() {
        let a: [f64; 3] = [0., 1., 2.];
        let e = f64x2::new(1., 2.);
        let r: f64x2 = transmute(vld1q_f64(a[1..].as_ptr()));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_dup_f64() {
        let a: [f64; 2] = [1., 42.];
        let e = f64x1::new(42.);
        let r: f64x1 = transmute(vld1_dup_f64(a[1..].as_ptr()));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_dup_f64() {
        let elem: f64 = 42.;
        let e = f64x2::new(42., 42.);
        let r: f64x2 = transmute(vld1q_dup_f64(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_lane_f64() {
        let a = f64x1::new(0.);
        let elem: f64 = 42.;
        let e = f64x1::new(42.);
        let r: f64x1 = transmute(vld1_lane_f64::<0>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_lane_f64() {
        let a = f64x2::new(0., 1.);
        let elem: f64 = 42.;
        let e = f64x2::new(0., 42.);
        let r: f64x2 = transmute(vld1q_lane_f64::<1>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vst1_f64() {
        let mut vals = [0_f64; 2];
        let a = f64x1::new(1.);

        vst1_f64(vals[1..].as_mut_ptr(), transmute(a));

        assert_eq!(vals[0], 0.);
        assert_eq!(vals[1], 1.);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vst1q_f64() {
        let mut vals = [0_f64; 3];
        let a = f64x2::new(1., 2.);

        vst1q_f64(vals[1..].as_mut_ptr(), transmute(a));

        assert_eq!(vals[0], 0.);
        assert_eq!(vals[1], 1.);
        assert_eq!(vals[2], 2.);
    }
}

#[cfg(test)]
#[path = "../../arm_shared/neon/table_lookup_tests.rs"]
mod table_lookup_tests;

#[cfg(test)]
#[path = "../../arm_shared/neon/shift_and_insert_tests.rs"]
mod shift_and_insert_tests;

#[cfg(test)]
#[path = "../../arm_shared/neon/load_tests.rs"]
mod load_tests;

#[cfg(test)]
#[path = "../../arm_shared/neon/store_tests.rs"]
mod store_tests;
