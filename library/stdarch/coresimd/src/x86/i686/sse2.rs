//! `i686`'s Streaming SIMD Extensions 2 (SSE2)

use core::mem;
use v128::*;
use v64::{__m64, i32x2};

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Return `a` with its lower element replaced by `b` after converting it to
/// an `f64`.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(all(test, not(target_arch = "x86")), assert_instr(cvtsi2sd))]
pub unsafe fn _mm_cvtsi64_sd(a: f64x2, b: i64) -> f64x2 {
    a.replace(0, b as f64)
}

/// Return `a` with its lower element replaced by `b` after converting it to
/// an `f64`.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(all(test, not(target_arch = "x86")), assert_instr(cvtsi2sd))]
pub unsafe fn _mm_cvtsi64x_sd(a: f64x2, b: i64) -> f64x2 {
    _mm_cvtsi64_sd(a, b)
}

/// Return a vector whose lowest element is `a` and all higher elements are
/// `0`.
#[inline(always)]
#[target_feature = "+sse2"]
// no particular instruction to test
pub unsafe fn _mm_cvtsi64_si128(a: i64) -> i64x2 {
    i64x2::new(a, 0)
}

/// Return a vector whose lowest element is `a` and all higher elements are
/// `0`.
#[inline(always)]
#[target_feature = "+sse2"]
// no particular instruction to test
pub unsafe fn _mm_cvtsi64x_si128(a: i64) -> i64x2 {
    _mm_cvtsi64_si128(a)
}

/// Return the lowest element of `a`.
#[inline(always)]
#[target_feature = "+sse2"]
// no particular instruction to test
pub unsafe fn _mm_cvtsi128_si64(a: i64x2) -> i64 {
    a.extract(0)
}

/// Return the lowest element of `a`.
#[inline(always)]
#[target_feature = "+sse2"]
// no particular instruction to test
pub unsafe fn _mm_cvtsi128_si64x(a: i64x2) -> i64 {
    _mm_cvtsi128_si64(a)
}

/// Returns the lower 64 bits of a 128-bit integer vector as a 64-bit
/// integer.
#[inline(always)]
#[target_feature = "+sse2"]
// no particular instruction to test
pub unsafe fn _mm_movepi64_pi64(a: i64x2) -> i64 {
    a.extract(0)
}

/// Moves the 64-bit operand to a 128-bit integer vector, zeroing the
/// upper bits.
#[inline(always)]
#[target_feature = "+sse2"]
// #[cfg_attr(test, assert_instr(movq2dq))] FIXME
pub unsafe fn _mm_movpi64_epi64(a: i64) -> i64x2 {
    i64x2::new(a, 0)
}

/// Converts the two double-precision floating-point elements of a
/// 128-bit vector of [2 x double] into two signed 32-bit integer values,
/// returned in a 64-bit vector of [2 x i32].
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(test, assert_instr(cvtpd2pi))]
pub unsafe fn _mm_cvtpd_pi32(a: f64x2) -> i32x2 {
    mem::transmute(cvtpd2pi(a))
}

/// Converts the two double-precision floating-point elements of a
/// 128-bit vector of [2 x double] into two signed 32-bit integer values,
/// returned in a 64-bit vector of [2 x i32].
/// If the result of either conversion is inexact, the result is truncated
/// (rounded towards zero) regardless of the current MXCSR setting.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(test, assert_instr(cvttpd2pi))]
pub unsafe fn _mm_cvttpd_pi32(a: f64x2) -> i32x2 {
    mem::transmute(cvttpd2pi(a))
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.sse.cvtpd2pi"]
    fn cvtpd2pi(a: f64x2) -> __m64;
    #[link_name = "llvm.x86.sse.cvttpd2pi"]
    fn cvttpd2pi(a: f64x2) -> __m64;
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use v128::*;
    use v64::i32x2;
    use x86::i686::sse2;

    #[simd_test = "sse2"]
    unsafe fn _mm_cvtsi64_sd() {
        let a = f64x2::splat(3.5);
        let r = sse2::_mm_cvtsi64_sd(a, 5);
        assert_eq!(r, f64x2::new(5.0, 3.5));
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_cvtsi64_si128() {
        let r = sse2::_mm_cvtsi64_si128(5);
        assert_eq!(r, i64x2::new(5, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_cvtsi128_si64() {
        let r = sse2::_mm_cvtsi128_si64(i64x2::new(5, 0));
        assert_eq!(r, 5);
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_movepi64_pi64() {
        let r = sse2::_mm_movepi64_pi64(i64x2::new(5, 0));
        assert_eq!(r, 5);
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_movpi64_epi64() {
        let r = sse2::_mm_movpi64_epi64(5);
        assert_eq!(r, i64x2::new(5, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_cvtpd_pi32() {
        let a = f64x2::new(5., 0.);
        let r = sse2::_mm_cvtpd_pi32(a);
        assert_eq!(r, i32x2::new(5, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_cvttpd_pi32() {
        use std::{f64, i32};

        let a = f64x2::new(5., 0.);
        let r = sse2::_mm_cvttpd_pi32(a);
        assert_eq!(r, i32x2::new(5, 0));

        let a = f64x2::new(f64::NEG_INFINITY, f64::NAN);
        let r = sse2::_mm_cvttpd_pi32(a);
        assert_eq!(r, i32x2::new(i32::MIN, i32::MIN));
    }
}
