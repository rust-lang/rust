//! `i686`'s Streaming SIMD Extensions 2 (SSE2)

use core::mem;

use simd_llvm::simd_extract;
use x86::*;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Adds two signed or unsigned 64-bit integer values, returning the
/// lower 64 bits of the sum.
#[inline(always)]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(paddq))]
pub unsafe fn _mm_add_si64(a: __m64, b: __m64) -> __m64 {
    paddq(a, b)
}

/// Multiplies 32-bit unsigned integer values contained in the lower bits
/// of the two 64-bit integer vectors and returns the 64-bit unsigned
/// product.
#[inline(always)]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pmuludq))]
pub unsafe fn _mm_mul_su32(a: __m64, b: __m64) -> __m64 {
    pmuludq(mem::transmute(a), mem::transmute(b))
}

/// Subtracts signed or unsigned 64-bit integer values and writes the
/// difference to the corresponding bits in the destination.
#[inline(always)]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psubq))]
pub unsafe fn _mm_sub_si64(a: __m64, b: __m64) -> __m64 {
    psubq(a, b)
}

/// Converts the two signed 32-bit integer elements of a 64-bit vector of
/// [2 x i32] into two double-precision floating-point values, returned in a
/// 128-bit vector of [2 x double].
#[inline(always)]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvtpi2pd))]
pub unsafe fn _mm_cvtpi32_pd(a: __m64) -> __m128d {
    cvtpi2pd(a)
}

/// Initializes both 64-bit values in a 128-bit vector of [2 x i64] with
/// the specified 64-bit integer values.
#[inline(always)]
#[target_feature(enable = "sse2")]
// no particular instruction to test
pub unsafe fn _mm_set_epi64(e1: __m64, e0: __m64) -> __m128i {
    _mm_set_epi64x(mem::transmute(e1), mem::transmute(e0))
}

/// Initializes both values in a 128-bit vector of [2 x i64] with the
/// specified 64-bit value.
#[inline(always)]
#[target_feature(enable = "sse2")]
// no particular instruction to test
pub unsafe fn _mm_set1_epi64(a: __m64) -> __m128i {
    _mm_set_epi64x(mem::transmute(a), mem::transmute(a))
}

/// Constructs a 128-bit integer vector, initialized in reverse order
/// with the specified 64-bit integral values.
#[inline(always)]
#[target_feature(enable = "sse2")]
// no particular instruction to test
pub unsafe fn _mm_setr_epi64(e1: __m64, e0: __m64) -> __m128i {
    _mm_set_epi64x(mem::transmute(e0), mem::transmute(e1))
}

/// Returns the lower 64 bits of a 128-bit integer vector as a 64-bit
/// integer.
#[inline(always)]
#[target_feature(enable = "sse2")]
// #[cfg_attr(test, assert_instr(movdq2q))] // FIXME: llvm codegens wrong
// instr?
pub unsafe fn _mm_movepi64_pi64(a: __m128i) -> __m64 {
    mem::transmute(simd_extract::<_, i64>(a.as_i64x2(), 0))
}

/// Moves the 64-bit operand to a 128-bit integer vector, zeroing the
/// upper bits.
#[inline(always)]
#[target_feature(enable = "sse2")]
// #[cfg_attr(test, assert_instr(movq2dq))] // FIXME: llvm codegens wrong
// instr?
pub unsafe fn _mm_movpi64_epi64(a: __m64) -> __m128i {
    _mm_set_epi64x(0, mem::transmute(a))
}

/// Converts the two double-precision floating-point elements of a
/// 128-bit vector of [2 x double] into two signed 32-bit integer values,
/// returned in a 64-bit vector of [2 x i32].
#[inline(always)]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvtpd2pi))]
pub unsafe fn _mm_cvtpd_pi32(a: __m128d) -> __m64 {
    cvtpd2pi(a)
}

/// Converts the two double-precision floating-point elements of a
/// 128-bit vector of [2 x double] into two signed 32-bit integer values,
/// returned in a 64-bit vector of [2 x i32].
/// If the result of either conversion is inexact, the result is truncated
/// (rounded towards zero) regardless of the current MXCSR setting.
#[inline(always)]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvttpd2pi))]
pub unsafe fn _mm_cvttpd_pi32(a: __m128d) -> __m64 {
    cvttpd2pi(a)
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
    fn cvtpi2pd(a: __m64) -> __m128d;
    #[link_name = "llvm.x86.sse.cvtpd2pi"]
    fn cvtpd2pi(a: __m128d) -> __m64;
    #[link_name = "llvm.x86.sse.cvttpd2pi"]
    fn cvttpd2pi(a: __m128d) -> __m64;
}

#[cfg(test)]
mod tests {
    use std::mem;

    use stdsimd_test::simd_test;

    use x86::*;

    #[simd_test = "sse2"]
    unsafe fn test_mm_add_si64() {
        let a = 1i64;
        let b = 2i64;
        let expected = 3i64;
        let r = _mm_add_si64(mem::transmute(a), mem::transmute(b));
        assert_eq!(mem::transmute::<__m64, i64>(r), expected);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_mul_su32() {
        let a = _mm_setr_pi32(1, 2);
        let b = _mm_setr_pi32(3, 4);
        let expected = 3u64;
        let r = _mm_mul_su32(a, b);
        assert_eq!(r, mem::transmute(expected));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sub_si64() {
        let a = 1i64;
        let b = 2i64;
        let expected = -1i64;
        let r = _mm_sub_si64(mem::transmute(a), mem::transmute(b));
        assert_eq!(mem::transmute::<__m64, i64>(r), expected);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtpi32_pd() {
        let a = _mm_setr_pi32(1, 2);
        let expected = _mm_setr_pd(1., 2.);
        let r = _mm_cvtpi32_pd(a);
        assert_eq_m128d(r, expected);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_set_epi64() {
        let r =
            _mm_set_epi64(mem::transmute(1i64), mem::transmute(2i64));
        assert_eq!(r, _mm_setr_epi64x(2, 1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_set1_epi64() {
        let r = _mm_set1_epi64(mem::transmute(1i64));
        assert_eq!(r, _mm_setr_epi64x(1, 1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_setr_epi64() {
        let r =
            _mm_setr_epi64(mem::transmute(1i64), mem::transmute(2i64));
        assert_eq!(r, _mm_setr_epi64x(1, 2));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_movepi64_pi64() {
        let r = _mm_movepi64_pi64(_mm_setr_epi64x(5, 0));
        assert_eq!(r, _mm_setr_pi8(5, 0, 0, 0, 0, 0, 0, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_movpi64_epi64() {
        let r = _mm_movpi64_epi64(_mm_setr_pi8(
            5,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ));
        assert_eq!(r, _mm_setr_epi64x(5, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtpd_pi32() {
        let a = _mm_setr_pd(5., 0.);
        let r = _mm_cvtpd_pi32(a);
        assert_eq!(r, _mm_setr_pi32(5, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvttpd_pi32() {
        use std::{f64, i32};

        let a = _mm_setr_pd(5., 0.);
        let r = _mm_cvttpd_pi32(a);
        assert_eq!(r, _mm_setr_pi32(5, 0));

        let a = _mm_setr_pd(f64::NEG_INFINITY, f64::NAN);
        let r = _mm_cvttpd_pi32(a);
        assert_eq!(r, _mm_setr_pi32(i32::MIN, i32::MIN));
    }
}
