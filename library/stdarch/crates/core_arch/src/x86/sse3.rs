//! Streaming SIMD Extensions 3 (SSE3)

use crate::{
    core_arch::{
        simd::*,
        simd_llvm::{simd_shuffle2, simd_shuffle4},
        x86::*,
    },
    mem::transmute,
};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Alternatively add and subtract packed single-precision (32-bit)
/// floating-point elements in `a` to/from packed elements in `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_addsub_ps)
#[inline]
#[target_feature(enable = "sse3")]
#[cfg_attr(test, assert_instr(addsubps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_addsub_ps(a: __m128, b: __m128) -> __m128 {
    addsubps(a, b)
}

/// Alternatively add and subtract packed double-precision (64-bit)
/// floating-point elements in `a` to/from packed elements in `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_addsub_pd)
#[inline]
#[target_feature(enable = "sse3")]
#[cfg_attr(test, assert_instr(addsubpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_addsub_pd(a: __m128d, b: __m128d) -> __m128d {
    addsubpd(a, b)
}

/// Horizontally adds adjacent pairs of double-precision (64-bit)
/// floating-point elements in `a` and `b`, and pack the results.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_hadd_pd)
#[inline]
#[target_feature(enable = "sse3")]
#[cfg_attr(test, assert_instr(haddpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_hadd_pd(a: __m128d, b: __m128d) -> __m128d {
    haddpd(a, b)
}

/// Horizontally adds adjacent pairs of single-precision (32-bit)
/// floating-point elements in `a` and `b`, and pack the results.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_hadd_ps)
#[inline]
#[target_feature(enable = "sse3")]
#[cfg_attr(test, assert_instr(haddps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_hadd_ps(a: __m128, b: __m128) -> __m128 {
    haddps(a, b)
}

/// Horizontally subtract adjacent pairs of double-precision (64-bit)
/// floating-point elements in `a` and `b`, and pack the results.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_hsub_pd)
#[inline]
#[target_feature(enable = "sse3")]
#[cfg_attr(test, assert_instr(hsubpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_hsub_pd(a: __m128d, b: __m128d) -> __m128d {
    hsubpd(a, b)
}

/// Horizontally adds adjacent pairs of single-precision (32-bit)
/// floating-point elements in `a` and `b`, and pack the results.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_hsub_ps)
#[inline]
#[target_feature(enable = "sse3")]
#[cfg_attr(test, assert_instr(hsubps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_hsub_ps(a: __m128, b: __m128) -> __m128 {
    hsubps(a, b)
}

/// Loads 128-bits of integer data from unaligned memory.
/// This intrinsic may perform better than `_mm_loadu_si128`
/// when the data crosses a cache line boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_lddqu_si128)
#[inline]
#[target_feature(enable = "sse3")]
#[cfg_attr(test, assert_instr(lddqu))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_lddqu_si128(mem_addr: *const __m128i) -> __m128i {
    transmute(lddqu(mem_addr as *const _))
}

/// Duplicate the low double-precision (64-bit) floating-point element
/// from `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_movedup_pd)
#[inline]
#[target_feature(enable = "sse3")]
#[cfg_attr(test, assert_instr(movddup))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_movedup_pd(a: __m128d) -> __m128d {
    simd_shuffle2!(a, a, [0, 0])
}

/// Loads a double-precision (64-bit) floating-point element from memory
/// into both elements of return vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_loaddup_pd)
#[inline]
#[target_feature(enable = "sse3")]
#[cfg_attr(test, assert_instr(movddup))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_loaddup_pd(mem_addr: *const f64) -> __m128d {
    _mm_load1_pd(mem_addr)
}

/// Duplicate odd-indexed single-precision (32-bit) floating-point elements
/// from `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_movehdup_ps)
#[inline]
#[target_feature(enable = "sse3")]
#[cfg_attr(test, assert_instr(movshdup))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_movehdup_ps(a: __m128) -> __m128 {
    simd_shuffle4!(a, a, [1, 1, 3, 3])
}

/// Duplicate even-indexed single-precision (32-bit) floating-point elements
/// from `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_moveldup_ps)
#[inline]
#[target_feature(enable = "sse3")]
#[cfg_attr(test, assert_instr(movsldup))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_moveldup_ps(a: __m128) -> __m128 {
    simd_shuffle4!(a, a, [0, 0, 2, 2])
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.sse3.addsub.ps"]
    fn addsubps(a: __m128, b: __m128) -> __m128;
    #[link_name = "llvm.x86.sse3.addsub.pd"]
    fn addsubpd(a: __m128d, b: __m128d) -> __m128d;
    #[link_name = "llvm.x86.sse3.hadd.pd"]
    fn haddpd(a: __m128d, b: __m128d) -> __m128d;
    #[link_name = "llvm.x86.sse3.hadd.ps"]
    fn haddps(a: __m128, b: __m128) -> __m128;
    #[link_name = "llvm.x86.sse3.hsub.pd"]
    fn hsubpd(a: __m128d, b: __m128d) -> __m128d;
    #[link_name = "llvm.x86.sse3.hsub.ps"]
    fn hsubps(a: __m128, b: __m128) -> __m128;
    #[link_name = "llvm.x86.sse3.ldu.dq"]
    fn lddqu(mem_addr: *const i8) -> i8x16;
}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "sse3")]
    unsafe fn test_mm_addsub_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_addsub_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(99.0, 25.0, 0.0, -15.0));
    }

    #[simd_test(enable = "sse3")]
    unsafe fn test_mm_addsub_pd() {
        let a = _mm_setr_pd(-1.0, 5.0);
        let b = _mm_setr_pd(-100.0, 20.0);
        let r = _mm_addsub_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(99.0, 25.0));
    }

    #[simd_test(enable = "sse3")]
    unsafe fn test_mm_hadd_pd() {
        let a = _mm_setr_pd(-1.0, 5.0);
        let b = _mm_setr_pd(-100.0, 20.0);
        let r = _mm_hadd_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(4.0, -80.0));
    }

    #[simd_test(enable = "sse3")]
    unsafe fn test_mm_hadd_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_hadd_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(4.0, -10.0, -80.0, -5.0));
    }

    #[simd_test(enable = "sse3")]
    unsafe fn test_mm_hsub_pd() {
        let a = _mm_setr_pd(-1.0, 5.0);
        let b = _mm_setr_pd(-100.0, 20.0);
        let r = _mm_hsub_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(-6.0, -120.0));
    }

    #[simd_test(enable = "sse3")]
    unsafe fn test_mm_hsub_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_hsub_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(-6.0, 10.0, -120.0, 5.0));
    }

    #[simd_test(enable = "sse3")]
    unsafe fn test_mm_lddqu_si128() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
        );
        let r = _mm_lddqu_si128(&a);
        assert_eq_m128i(a, r);
    }

    #[simd_test(enable = "sse3")]
    unsafe fn test_mm_movedup_pd() {
        let a = _mm_setr_pd(-1.0, 5.0);
        let r = _mm_movedup_pd(a);
        assert_eq_m128d(r, _mm_setr_pd(-1.0, -1.0));
    }

    #[simd_test(enable = "sse3")]
    unsafe fn test_mm_movehdup_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let r = _mm_movehdup_ps(a);
        assert_eq_m128(r, _mm_setr_ps(5.0, 5.0, -10.0, -10.0));
    }

    #[simd_test(enable = "sse3")]
    unsafe fn test_mm_moveldup_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let r = _mm_moveldup_ps(a);
        assert_eq_m128(r, _mm_setr_ps(-1.0, -1.0, 0.0, 0.0));
    }

    #[simd_test(enable = "sse3")]
    unsafe fn test_mm_loaddup_pd() {
        let d = -5.0;
        let r = _mm_loaddup_pd(&d);
        assert_eq_m128d(r, _mm_setr_pd(d, d));
    }
}
