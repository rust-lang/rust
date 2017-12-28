//! `i686`'s Streaming SIMD Extensions 2 (SSE2)

use core::mem;
use v128::*;
use v64::{__m64, i32x2, u32x2};

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Adds two signed or unsigned 64-bit integer values, returning the
/// lower 64 bits of the sum.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(test, assert_instr(paddq))]
pub unsafe fn _mm_add_si64(a: __m64, b: __m64) -> __m64 {
    paddq(a, b)
}

/// Multiplies 32-bit unsigned integer values contained in the lower bits
/// of the two 64-bit integer vectors and returns the 64-bit unsigned
/// product.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(test, assert_instr(pmuludq))]
pub unsafe fn _mm_mul_su32(a: u32x2, b: u32x2) -> u64 {
    mem::transmute(pmuludq(mem::transmute(a), mem::transmute(b)))
}

/// Subtracts signed or unsigned 64-bit integer values and writes the
/// difference to the corresponding bits in the destination.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(test, assert_instr(psubq))]
pub unsafe fn _mm_sub_si64(a: __m64, b: __m64) -> __m64 {
    psubq(a, b)
}

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

/// Converts the two signed 32-bit integer elements of a 64-bit vector of
/// [2 x i32] into two double-precision floating-point values, returned in a
/// 128-bit vector of [2 x double].
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(test, assert_instr(cvtpi2pd))]
pub unsafe fn _mm_cvtpi32_pd(a: i32x2) -> f64x2 {
    cvtpi2pd(mem::transmute(a))
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

/// Initializes both 64-bit values in a 128-bit vector of [2 x i64] with
/// the specified 64-bit integer values.
#[inline(always)]
#[target_feature = "+sse2"]
// no particular instruction to test
pub unsafe fn _mm_set_epi64(e1: i64, e0: i64) -> i64x2 {
    i64x2::new(e0, e1)
}

/// Initializes both values in a 128-bit vector of [2 x i64] with the
/// specified 64-bit value.
#[inline(always)]
#[target_feature = "+sse2"]
// no particular instruction to test
pub unsafe fn _mm_set1_epi64(a: i64) -> i64x2 {
    i64x2::new(a, a)
}

/// Constructs a 128-bit integer vector, initialized in reverse order
/// with the specified 64-bit integral values.
#[inline(always)]
#[target_feature = "+sse2"]
// no particular instruction to test
pub unsafe fn _mm_setr_epi64(e1: i64, e0: i64) -> i64x2 {
    i64x2::new(e1, e0)
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
    #[link_name = "llvm.x86.mmx.padd.q"]
    fn paddq(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pmulu.dq"]
    fn pmuludq(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.psub.q"]
    fn psubq(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.sse.cvtpi2pd"]
    fn cvtpi2pd(a: __m64) -> f64x2;
    #[link_name = "llvm.x86.sse.cvtpd2pi"]
    fn cvtpd2pi(a: f64x2) -> __m64;
    #[link_name = "llvm.x86.sse.cvttpd2pi"]
    fn cvttpd2pi(a: f64x2) -> __m64;
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    #[cfg(not(windows))]
    use core::mem;
    use v128::*;
    #[cfg(not(windows))]
    use v64::{__m64, i32x2, u32x2};
    #[cfg(windows)]
    use v64::i32x2;
    use x86::i686::sse2;

    #[simd_test = "sse2"]
    #[cfg(not(windows))] // FIXME
    unsafe fn _mm_add_si64() {
        let a = 1i64;
        let b = 2i64;
        let expected = 3i64;
        let r = sse2::_mm_add_si64(mem::transmute(a), mem::transmute(b));
        assert_eq!(mem::transmute::<__m64, i64>(r), expected);
    }

    #[simd_test = "sse2"]
    #[cfg(not(windows))] // FIXME
    unsafe fn _mm_mul_su32() {
        let a = u32x2::new(1, 2);
        let b = u32x2::new(3, 4);
        let expected = 3u64;
        let r = sse2::_mm_mul_su32(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "sse2"]
    #[cfg(not(windows))] // FIXME
    unsafe fn _mm_sub_si64() {
        let a = 1i64;
        let b = 2i64;
        let expected = -1i64;
        let r = sse2::_mm_sub_si64(mem::transmute(a), mem::transmute(b));
        assert_eq!(mem::transmute::<__m64, i64>(r), expected);
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_cvtsi64_sd() {
        let a = f64x2::splat(3.5);
        let r = sse2::_mm_cvtsi64_sd(a, 5);
        assert_eq!(r, f64x2::new(5.0, 3.5));
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_cvtpi32_pd() {
        let a = i32x2::new(1, 2);
        let expected = f64x2::new(1., 2.);
        let r = sse2::_mm_cvtpi32_pd(a);
        assert_eq!(r, expected);
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
    unsafe fn _mm_set_epi64() {
        let r = sse2::_mm_set_epi64(1, 2);
        assert_eq!(r, i64x2::new(2, 1));
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_set1_epi64() {
        let r = sse2::_mm_set1_epi64(1);
        assert_eq!(r, i64x2::new(1, 1));
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_setr_epi64() {
        let r = sse2::_mm_setr_epi64(1, 2);
        assert_eq!(r, i64x2::new(1, 2));
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
