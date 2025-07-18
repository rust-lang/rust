//! Fused Multiply-Add instruction set (FMA)
//!
//! The FMA instruction set is an extension to the 128 and 256-bit SSE
//! instructions in the x86 microprocessor instruction set to perform fused
//! multiply–add (FMA) operations.
//!
//! The references are:
//!
//! - [Intel 64 and IA-32 Architectures Software Developer's Manual Volume 2:
//!   Instruction Set Reference, A-Z][intel64_ref].
//! - [AMD64 Architecture Programmer's Manual, Volume 3: General-Purpose and
//!   System Instructions][amd64_ref].
//!
//! Wikipedia's [FMA][wiki_fma] page provides a quick overview of the
//! instructions available.
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
//! [amd64_ref]: http://support.amd.com/TechDocs/24594.pdf
//! [wiki_fma]: https://en.wikipedia.org/wiki/Fused_multiply-accumulate

use crate::core_arch::x86::*;
use crate::intrinsics::simd::{simd_fma, simd_neg};
use crate::intrinsics::{fmaf32, fmaf64};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and add the intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmadd_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fmadd_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    unsafe { simd_fma(a, b, c) }
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and add the intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmadd_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm256_fmadd_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    unsafe { simd_fma(a, b, c) }
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and add the intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmadd_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fmadd_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    unsafe { simd_fma(a, b, c) }
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and add the intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmadd_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm256_fmadd_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    unsafe { simd_fma(a, b, c) }
}

/// Multiplies the lower double-precision (64-bit) floating-point elements in
/// `a` and `b`, and add the intermediate result to the lower element in `c`.
/// Stores the result in the lower element of the returned value, and copy the
/// upper element from `a` to the upper elements of the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmadd_sd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fmadd_sd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    unsafe {
        simd_insert!(
            a,
            0,
            fmaf64(_mm_cvtsd_f64(a), _mm_cvtsd_f64(b), _mm_cvtsd_f64(c))
        )
    }
}

/// Multiplies the lower single-precision (32-bit) floating-point elements in
/// `a` and `b`, and add the intermediate result to the lower element in `c`.
/// Stores the result in the lower element of the returned value, and copy the
/// 3 upper elements from `a` to the upper elements of the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmadd_ss)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fmadd_ss(a: __m128, b: __m128, c: __m128) -> __m128 {
    unsafe {
        simd_insert!(
            a,
            0,
            fmaf32(_mm_cvtss_f32(a), _mm_cvtss_f32(b), _mm_cvtss_f32(c))
        )
    }
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and alternatively add and subtract packed elements in `c` to/from
/// the intermediate result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmaddsub_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmaddsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fmaddsub_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    unsafe {
        let add = simd_fma(a, b, c);
        let sub = simd_fma(a, b, simd_neg(c));
        simd_shuffle!(add, sub, [2, 1])
    }
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and alternatively add and subtract packed elements in `c` to/from
/// the intermediate result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmaddsub_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmaddsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm256_fmaddsub_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    unsafe {
        let add = simd_fma(a, b, c);
        let sub = simd_fma(a, b, simd_neg(c));
        simd_shuffle!(add, sub, [4, 1, 6, 3])
    }
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and alternatively add and subtract packed elements in `c` to/from
/// the intermediate result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmaddsub_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmaddsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fmaddsub_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    unsafe {
        let add = simd_fma(a, b, c);
        let sub = simd_fma(a, b, simd_neg(c));
        simd_shuffle!(add, sub, [4, 1, 6, 3])
    }
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and alternatively add and subtract packed elements in `c` to/from
/// the intermediate result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmaddsub_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmaddsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm256_fmaddsub_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    unsafe {
        let add = simd_fma(a, b, c);
        let sub = simd_fma(a, b, simd_neg(c));
        simd_shuffle!(add, sub, [8, 1, 10, 3, 12, 5, 14, 7])
    }
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the intermediate result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmsub_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fmsub_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    unsafe { simd_fma(a, b, simd_neg(c)) }
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the intermediate result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmsub_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm256_fmsub_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    unsafe { simd_fma(a, b, simd_neg(c)) }
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the intermediate result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmsub_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsub213ps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fmsub_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    unsafe { simd_fma(a, b, simd_neg(c)) }
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the intermediate result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmsub_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsub213ps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm256_fmsub_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    unsafe { simd_fma(a, b, simd_neg(c)) }
}

/// Multiplies the lower double-precision (64-bit) floating-point elements in
/// `a` and `b`, and subtract the lower element in `c` from the intermediate
/// result. Store the result in the lower element of the returned value, and
/// copy the upper element from `a` to the upper elements of the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmsub_sd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fmsub_sd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    unsafe {
        simd_insert!(
            a,
            0,
            fmaf64(_mm_cvtsd_f64(a), _mm_cvtsd_f64(b), -_mm_cvtsd_f64(c))
        )
    }
}

/// Multiplies the lower single-precision (32-bit) floating-point elements in
/// `a` and `b`,  and subtract the lower element in `c` from the intermediate
/// result. Store the result in the lower element of the returned value, and
/// copy the 3 upper elements from `a` to the upper elements of the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmsub_ss)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fmsub_ss(a: __m128, b: __m128, c: __m128) -> __m128 {
    unsafe {
        simd_insert!(
            a,
            0,
            fmaf32(_mm_cvtss_f32(a), _mm_cvtss_f32(b), -_mm_cvtss_f32(c))
        )
    }
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and alternatively subtract and add packed elements in `c` from/to
/// the intermediate result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmsubadd_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsubadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fmsubadd_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    unsafe {
        let add = simd_fma(a, b, c);
        let sub = simd_fma(a, b, simd_neg(c));
        simd_shuffle!(add, sub, [0, 3])
    }
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and alternatively subtract and add packed elements in `c` from/to
/// the intermediate result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmsubadd_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsubadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm256_fmsubadd_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    unsafe {
        let add = simd_fma(a, b, c);
        let sub = simd_fma(a, b, simd_neg(c));
        simd_shuffle!(add, sub, [0, 5, 2, 7])
    }
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and alternatively subtract and add packed elements in `c` from/to
/// the intermediate result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fmsubadd_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsubadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fmsubadd_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    unsafe {
        let add = simd_fma(a, b, c);
        let sub = simd_fma(a, b, simd_neg(c));
        simd_shuffle!(add, sub, [0, 5, 2, 7])
    }
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and alternatively subtract and add packed elements in `c` from/to
/// the intermediate result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmsubadd_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsubadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm256_fmsubadd_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    unsafe {
        let add = simd_fma(a, b, c);
        let sub = simd_fma(a, b, simd_neg(c));
        simd_shuffle!(add, sub, [0, 9, 2, 11, 4, 13, 6, 15])
    }
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and add the negated intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fnmadd_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fnmadd_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    unsafe { simd_fma(simd_neg(a), b, c) }
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and add the negated intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fnmadd_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm256_fnmadd_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    unsafe { simd_fma(simd_neg(a), b, c) }
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and add the negated intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fnmadd_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fnmadd_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    unsafe { simd_fma(simd_neg(a), b, c) }
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and add the negated intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fnmadd_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm256_fnmadd_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    unsafe { simd_fma(simd_neg(a), b, c) }
}

/// Multiplies the lower double-precision (64-bit) floating-point elements in
/// `a` and `b`, and add the negated intermediate result to the lower element
/// in `c`. Store the result in the lower element of the returned value, and
/// copy the upper element from `a` to the upper elements of the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fnmadd_sd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fnmadd_sd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    unsafe {
        simd_insert!(
            a,
            0,
            fmaf64(_mm_cvtsd_f64(a), -_mm_cvtsd_f64(b), _mm_cvtsd_f64(c))
        )
    }
}

/// Multiplies the lower single-precision (32-bit) floating-point elements in
/// `a` and `b`, and add the negated intermediate result to the lower element
/// in `c`. Store the result in the lower element of the returned value, and
/// copy the 3 upper elements from `a` to the upper elements of the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fnmadd_ss)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fnmadd_ss(a: __m128, b: __m128, c: __m128) -> __m128 {
    unsafe {
        simd_insert!(
            a,
            0,
            fmaf32(_mm_cvtss_f32(a), -_mm_cvtss_f32(b), _mm_cvtss_f32(c))
        )
    }
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the negated intermediate
/// result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fnmsub_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fnmsub_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    unsafe { simd_fma(simd_neg(a), b, simd_neg(c)) }
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the negated intermediate
/// result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fnmsub_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm256_fnmsub_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    unsafe { simd_fma(simd_neg(a), b, simd_neg(c)) }
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the negated intermediate
/// result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fnmsub_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fnmsub_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    unsafe { simd_fma(simd_neg(a), b, simd_neg(c)) }
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the negated intermediate
/// result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fnmsub_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm256_fnmsub_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    unsafe { simd_fma(simd_neg(a), b, simd_neg(c)) }
}

/// Multiplies the lower double-precision (64-bit) floating-point elements in
/// `a` and `b`, and subtract packed elements in `c` from the negated
/// intermediate result. Store the result in the lower element of the returned
/// value, and copy the upper element from `a` to the upper elements of the
/// result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fnmsub_sd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fnmsub_sd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    unsafe {
        simd_insert!(
            a,
            0,
            fmaf64(_mm_cvtsd_f64(a), -_mm_cvtsd_f64(b), -_mm_cvtsd_f64(c))
        )
    }
}

/// Multiplies the lower single-precision (32-bit) floating-point elements in
/// `a` and `b`, and subtract packed elements in `c` from the negated
/// intermediate result. Store the result in the lower element of the
/// returned value, and copy the 3 upper elements from `a` to the upper
/// elements of the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fnmsub_ss)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_fnmsub_ss(a: __m128, b: __m128, c: __m128) -> __m128 {
    unsafe {
        simd_insert!(
            a,
            0,
            fmaf32(_mm_cvtss_f32(a), -_mm_cvtss_f32(b), -_mm_cvtss_f32(c))
        )
    }
}

#[cfg(test)]
mod tests {

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmadd_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(9., 15.);
        assert_eq_m128d(_mm_fmadd_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmadd_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 3., 7., 2.);
        let c = _mm256_setr_pd(4., 9., 1., 7.);
        let r = _mm256_setr_pd(9., 15., 22., 15.);
        assert_eq_m256d(_mm256_fmadd_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmadd_ps() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(9., 15., 22., 15.);
        assert_eq_m128(_mm_fmadd_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmadd_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 0., 10., -1., -2.);
        let b = _mm256_setr_ps(5., 3., 7., 2., 4., -6., 0., 14.);
        let c = _mm256_setr_ps(4., 9., 1., 7., -5., 11., -2., -3.);
        let r = _mm256_setr_ps(9., 15., 22., 15., -5., -49., -2., -31.);
        assert_eq_m256(_mm256_fmadd_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmadd_sd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(9., 2.);
        assert_eq_m128d(_mm_fmadd_sd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmadd_ss() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(9., 2., 3., 4.);
        assert_eq_m128(_mm_fmadd_ss(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmaddsub_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(1., 15.);
        assert_eq_m128d(_mm_fmaddsub_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmaddsub_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 3., 7., 2.);
        let c = _mm256_setr_pd(4., 9., 1., 7.);
        let r = _mm256_setr_pd(1., 15., 20., 15.);
        assert_eq_m256d(_mm256_fmaddsub_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmaddsub_ps() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(1., 15., 20., 15.);
        assert_eq_m128(_mm_fmaddsub_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmaddsub_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 0., 10., -1., -2.);
        let b = _mm256_setr_ps(5., 3., 7., 2., 4., -6., 0., 14.);
        let c = _mm256_setr_ps(4., 9., 1., 7., -5., 11., -2., -3.);
        let r = _mm256_setr_ps(1., 15., 20., 15., 5., -49., 2., -31.);
        assert_eq_m256(_mm256_fmaddsub_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmsub_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(1., -3.);
        assert_eq_m128d(_mm_fmsub_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmsub_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 3., 7., 2.);
        let c = _mm256_setr_pd(4., 9., 1., 7.);
        let r = _mm256_setr_pd(1., -3., 20., 1.);
        assert_eq_m256d(_mm256_fmsub_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmsub_ps() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(1., -3., 20., 1.);
        assert_eq_m128(_mm_fmsub_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmsub_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 0., 10., -1., -2.);
        let b = _mm256_setr_ps(5., 3., 7., 2., 4., -6., 0., 14.);
        let c = _mm256_setr_ps(4., 9., 1., 7., -5., 11., -2., -3.);
        let r = _mm256_setr_ps(1., -3., 20., 1., 5., -71., 2., -25.);
        assert_eq_m256(_mm256_fmsub_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmsub_sd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(1., 2.);
        assert_eq_m128d(_mm_fmsub_sd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmsub_ss() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(1., 2., 3., 4.);
        assert_eq_m128(_mm_fmsub_ss(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmsubadd_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(9., -3.);
        assert_eq_m128d(_mm_fmsubadd_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmsubadd_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 3., 7., 2.);
        let c = _mm256_setr_pd(4., 9., 1., 7.);
        let r = _mm256_setr_pd(9., -3., 22., 1.);
        assert_eq_m256d(_mm256_fmsubadd_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmsubadd_ps() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(9., -3., 22., 1.);
        assert_eq_m128(_mm_fmsubadd_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmsubadd_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 0., 10., -1., -2.);
        let b = _mm256_setr_ps(5., 3., 7., 2., 4., -6., 0., 14.);
        let c = _mm256_setr_ps(4., 9., 1., 7., -5., 11., -2., -3.);
        let r = _mm256_setr_ps(9., -3., 22., 1., -5., -71., -2., -25.);
        assert_eq_m256(_mm256_fmsubadd_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmadd_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(-1., 3.);
        assert_eq_m128d(_mm_fnmadd_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fnmadd_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 3., 7., 2.);
        let c = _mm256_setr_pd(4., 9., 1., 7.);
        let r = _mm256_setr_pd(-1., 3., -20., -1.);
        assert_eq_m256d(_mm256_fnmadd_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmadd_ps() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(-1., 3., -20., -1.);
        assert_eq_m128(_mm_fnmadd_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fnmadd_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 0., 10., -1., -2.);
        let b = _mm256_setr_ps(5., 3., 7., 2., 4., -6., 0., 14.);
        let c = _mm256_setr_ps(4., 9., 1., 7., -5., 11., -2., -3.);
        let r = _mm256_setr_ps(-1., 3., -20., -1., -5., 71., -2., 25.);
        assert_eq_m256(_mm256_fnmadd_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmadd_sd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(-1., 2.);
        assert_eq_m128d(_mm_fnmadd_sd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmadd_ss() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(-1., 2., 3., 4.);
        assert_eq_m128(_mm_fnmadd_ss(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmsub_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(-9., -15.);
        assert_eq_m128d(_mm_fnmsub_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fnmsub_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 3., 7., 2.);
        let c = _mm256_setr_pd(4., 9., 1., 7.);
        let r = _mm256_setr_pd(-9., -15., -22., -15.);
        assert_eq_m256d(_mm256_fnmsub_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmsub_ps() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(-9., -15., -22., -15.);
        assert_eq_m128(_mm_fnmsub_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fnmsub_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 0., 10., -1., -2.);
        let b = _mm256_setr_ps(5., 3., 7., 2., 4., -6., 0., 14.);
        let c = _mm256_setr_ps(4., 9., 1., 7., -5., 11., -2., -3.);
        let r = _mm256_setr_ps(-9., -15., -22., -15., 5., 49., 2., 31.);
        assert_eq_m256(_mm256_fnmsub_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmsub_sd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(-9., 2.);
        assert_eq_m128d(_mm_fnmsub_sd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmsub_ss() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(-9., 2., 3., 4.);
        assert_eq_m128(_mm_fnmsub_ss(a, b, c), r);
    }
}
