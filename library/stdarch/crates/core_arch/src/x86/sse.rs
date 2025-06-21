//! Streaming SIMD Extensions (SSE)

use crate::{
    core_arch::{simd::*, x86::*},
    intrinsics::simd::*,
    intrinsics::sqrtf32,
    mem, ptr,
};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Adds the first component of `a` and `b`, the other components are copied
/// from `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_add_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(addss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_add_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_insert!(a, 0, _mm_cvtss_f32(a) + _mm_cvtss_f32(b)) }
}

/// Adds packed single-precision (32-bit) floating-point elements in `a` and
/// `b`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_add_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(addps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_add_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_add(a, b) }
}

/// Subtracts the first component of `b` from `a`, the other components are
/// copied from `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sub_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(subss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_sub_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_insert!(a, 0, _mm_cvtss_f32(a) - _mm_cvtss_f32(b)) }
}

/// Subtracts packed single-precision (32-bit) floating-point elements in `a` and
/// `b`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sub_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(subps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_sub_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_sub(a, b) }
}

/// Multiplies the first component of `a` and `b`, the other components are
/// copied from `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mul_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(mulss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_mul_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_insert!(a, 0, _mm_cvtss_f32(a) * _mm_cvtss_f32(b)) }
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a` and
/// `b`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mul_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(mulps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_mul_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_mul(a, b) }
}

/// Divides the first component of `b` by `a`, the other components are
/// copied from `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_div_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(divss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_div_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_insert!(a, 0, _mm_cvtss_f32(a) / _mm_cvtss_f32(b)) }
}

/// Divides packed single-precision (32-bit) floating-point elements in `a` and
/// `b`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_div_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(divps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_div_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_div(a, b) }
}

/// Returns the square root of the first single-precision (32-bit)
/// floating-point element in `a`, the other elements are unchanged.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sqrt_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(sqrtss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_sqrt_ss(a: __m128) -> __m128 {
    unsafe { simd_insert!(a, 0, sqrtf32(_mm_cvtss_f32(a))) }
}

/// Returns the square root of packed single-precision (32-bit) floating-point
/// elements in `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sqrt_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(sqrtps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_sqrt_ps(a: __m128) -> __m128 {
    unsafe { simd_fsqrt(a) }
}

/// Returns the approximate reciprocal of the first single-precision
/// (32-bit) floating-point element in `a`, the other elements are unchanged.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_rcp_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(rcpss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_rcp_ss(a: __m128) -> __m128 {
    unsafe { rcpss(a) }
}

/// Returns the approximate reciprocal of packed single-precision (32-bit)
/// floating-point elements in `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_rcp_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(rcpps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_rcp_ps(a: __m128) -> __m128 {
    unsafe { rcpps(a) }
}

/// Returns the approximate reciprocal square root of the first single-precision
/// (32-bit) floating-point element in `a`, the other elements are unchanged.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_rsqrt_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(rsqrtss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_rsqrt_ss(a: __m128) -> __m128 {
    unsafe { rsqrtss(a) }
}

/// Returns the approximate reciprocal square root of packed single-precision
/// (32-bit) floating-point elements in `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_rsqrt_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(rsqrtps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_rsqrt_ps(a: __m128) -> __m128 {
    unsafe { rsqrtps(a) }
}

/// Compares the first single-precision (32-bit) floating-point element of `a`
/// and `b`, and return the minimum value in the first element of the return
/// value, the other elements are copied from `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_min_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(minss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_min_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { minss(a, b) }
}

/// Compares packed single-precision (32-bit) floating-point elements in `a` and
/// `b`, and return the corresponding minimum values.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_min_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(minps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_min_ps(a: __m128, b: __m128) -> __m128 {
    // See the `test_mm_min_ps` test why this can't be implemented using `simd_fmin`.
    unsafe { minps(a, b) }
}

/// Compares the first single-precision (32-bit) floating-point element of `a`
/// and `b`, and return the maximum value in the first element of the return
/// value, the other elements are copied from `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_max_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(maxss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_max_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { maxss(a, b) }
}

/// Compares packed single-precision (32-bit) floating-point elements in `a` and
/// `b`, and return the corresponding maximum values.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_max_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(maxps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_max_ps(a: __m128, b: __m128) -> __m128 {
    // See the `test_mm_min_ps` test why this can't be implemented using `simd_fmax`.
    unsafe { maxps(a, b) }
}

/// Bitwise AND of packed single-precision (32-bit) floating-point elements.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_and_ps)
#[inline]
#[target_feature(enable = "sse")]
// i586 only seems to generate plain `and` instructions, so ignore it.
#[cfg_attr(
    all(test, any(target_arch = "x86_64", target_feature = "sse2")),
    assert_instr(andps)
)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_and_ps(a: __m128, b: __m128) -> __m128 {
    unsafe {
        let a: __m128i = mem::transmute(a);
        let b: __m128i = mem::transmute(b);
        mem::transmute(simd_and(a, b))
    }
}

/// Bitwise AND-NOT of packed single-precision (32-bit) floating-point
/// elements.
///
/// Computes `!a & b` for each bit in `a` and `b`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_andnot_ps)
#[inline]
#[target_feature(enable = "sse")]
// i586 only seems to generate plain `not` and `and` instructions, so ignore
// it.
#[cfg_attr(
    all(test, any(target_arch = "x86_64", target_feature = "sse2")),
    assert_instr(andnps)
)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_andnot_ps(a: __m128, b: __m128) -> __m128 {
    unsafe {
        let a: __m128i = mem::transmute(a);
        let b: __m128i = mem::transmute(b);
        let mask: __m128i = mem::transmute(i32x4::splat(-1));
        mem::transmute(simd_and(simd_xor(mask, a), b))
    }
}

/// Bitwise OR of packed single-precision (32-bit) floating-point elements.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_or_ps)
#[inline]
#[target_feature(enable = "sse")]
// i586 only seems to generate plain `or` instructions, so we ignore it.
#[cfg_attr(
    all(test, any(target_arch = "x86_64", target_feature = "sse2")),
    assert_instr(orps)
)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_or_ps(a: __m128, b: __m128) -> __m128 {
    unsafe {
        let a: __m128i = mem::transmute(a);
        let b: __m128i = mem::transmute(b);
        mem::transmute(simd_or(a, b))
    }
}

/// Bitwise exclusive OR of packed single-precision (32-bit) floating-point
/// elements.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_xor_ps)
#[inline]
#[target_feature(enable = "sse")]
// i586 only seems to generate plain `xor` instructions, so we ignore it.
#[cfg_attr(
    all(test, any(target_arch = "x86_64", target_feature = "sse2")),
    assert_instr(xorps)
)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_xor_ps(a: __m128, b: __m128) -> __m128 {
    unsafe {
        let a: __m128i = mem::transmute(a);
        let b: __m128i = mem::transmute(b);
        mem::transmute(simd_xor(a, b))
    }
}

/// Compares the lowest `f32` of both inputs for equality. The lowest 32 bits of
/// the result will be `0xffffffff` if the two inputs are equal, or `0`
/// otherwise. The upper 96 bits of the result are the upper 96 bits of `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpeq_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpeqss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpeq_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpss(a, b, 0) }
}

/// Compares the lowest `f32` of both inputs for less than. The lowest 32 bits
/// of the result will be `0xffffffff` if `a.extract(0)` is less than
/// `b.extract(0)`, or `0` otherwise. The upper 96 bits of the result are the
/// upper 96 bits of `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmplt_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpltss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmplt_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpss(a, b, 1) }
}

/// Compares the lowest `f32` of both inputs for less than or equal. The lowest
/// 32 bits of the result will be `0xffffffff` if `a.extract(0)` is less than
/// or equal `b.extract(0)`, or `0` otherwise. The upper 96 bits of the result
/// are the upper 96 bits of `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmple_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpless))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmple_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpss(a, b, 2) }
}

/// Compares the lowest `f32` of both inputs for greater than. The lowest 32
/// bits of the result will be `0xffffffff` if `a.extract(0)` is greater
/// than `b.extract(0)`, or `0` otherwise. The upper 96 bits of the result
/// are the upper 96 bits of `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpgt_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpltss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpgt_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_shuffle!(a, cmpss(b, a, 1), [4, 1, 2, 3]) }
}

/// Compares the lowest `f32` of both inputs for greater than or equal. The
/// lowest 32 bits of the result will be `0xffffffff` if `a.extract(0)` is
/// greater than or equal `b.extract(0)`, or `0` otherwise. The upper 96 bits
/// of the result are the upper 96 bits of `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpge_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpless))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpge_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_shuffle!(a, cmpss(b, a, 2), [4, 1, 2, 3]) }
}

/// Compares the lowest `f32` of both inputs for inequality. The lowest 32 bits
/// of the result will be `0xffffffff` if `a.extract(0)` is not equal to
/// `b.extract(0)`, or `0` otherwise. The upper 96 bits of the result are the
/// upper 96 bits of `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpneq_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpneqss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpneq_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpss(a, b, 4) }
}

/// Compares the lowest `f32` of both inputs for not-less-than. The lowest 32
/// bits of the result will be `0xffffffff` if `a.extract(0)` is not less than
/// `b.extract(0)`, or `0` otherwise. The upper 96 bits of the result are the
/// upper 96 bits of `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpnlt_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpnltss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpnlt_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpss(a, b, 5) }
}

/// Compares the lowest `f32` of both inputs for not-less-than-or-equal. The
/// lowest 32 bits of the result will be `0xffffffff` if `a.extract(0)` is not
/// less than or equal to `b.extract(0)`, or `0` otherwise. The upper 96 bits
/// of the result are the upper 96 bits of `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpnle_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpnless))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpnle_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpss(a, b, 6) }
}

/// Compares the lowest `f32` of both inputs for not-greater-than. The lowest 32
/// bits of the result will be `0xffffffff` if `a.extract(0)` is not greater
/// than `b.extract(0)`, or `0` otherwise. The upper 96 bits of the result are
/// the upper 96 bits of `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpngt_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpnltss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpngt_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_shuffle!(a, cmpss(b, a, 5), [4, 1, 2, 3]) }
}

/// Compares the lowest `f32` of both inputs for not-greater-than-or-equal. The
/// lowest 32 bits of the result will be `0xffffffff` if `a.extract(0)` is not
/// greater than or equal to `b.extract(0)`, or `0` otherwise. The upper 96
/// bits of the result are the upper 96 bits of `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpnge_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpnless))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpnge_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_shuffle!(a, cmpss(b, a, 6), [4, 1, 2, 3]) }
}

/// Checks if the lowest `f32` of both inputs are ordered. The lowest 32 bits of
/// the result will be `0xffffffff` if neither of `a.extract(0)` or
/// `b.extract(0)` is a NaN, or `0` otherwise. The upper 96 bits of the result
/// are the upper 96 bits of `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpord_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpordss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpord_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpss(a, b, 7) }
}

/// Checks if the lowest `f32` of both inputs are unordered. The lowest 32 bits
/// of the result will be `0xffffffff` if any of `a.extract(0)` or
/// `b.extract(0)` is a NaN, or `0` otherwise. The upper 96 bits of the result
/// are the upper 96 bits of `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpunord_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpunordss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpunord_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpss(a, b, 3) }
}

/// Compares each of the four floats in `a` to the corresponding element in `b`.
/// The result in the output vector will be `0xffffffff` if the input elements
/// were equal, or `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpeq_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpeqps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpeq_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpps(a, b, 0) }
}

/// Compares each of the four floats in `a` to the corresponding element in `b`.
/// The result in the output vector will be `0xffffffff` if the input element
/// in `a` is less than the corresponding element in `b`, or `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmplt_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpltps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmplt_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpps(a, b, 1) }
}

/// Compares each of the four floats in `a` to the corresponding element in `b`.
/// The result in the output vector will be `0xffffffff` if the input element
/// in `a` is less than or equal to the corresponding element in `b`, or `0`
/// otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmple_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpleps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmple_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpps(a, b, 2) }
}

/// Compares each of the four floats in `a` to the corresponding element in `b`.
/// The result in the output vector will be `0xffffffff` if the input element
/// in `a` is greater than the corresponding element in `b`, or `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpgt_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpltps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpgt_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpps(b, a, 1) }
}

/// Compares each of the four floats in `a` to the corresponding element in `b`.
/// The result in the output vector will be `0xffffffff` if the input element
/// in `a` is greater than or equal to the corresponding element in `b`, or `0`
/// otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpge_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpleps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpge_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpps(b, a, 2) }
}

/// Compares each of the four floats in `a` to the corresponding element in `b`.
/// The result in the output vector will be `0xffffffff` if the input elements
/// are **not** equal, or `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpneq_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpneqps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpneq_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpps(a, b, 4) }
}

/// Compares each of the four floats in `a` to the corresponding element in `b`.
/// The result in the output vector will be `0xffffffff` if the input element
/// in `a` is **not** less than the corresponding element in `b`, or `0`
/// otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpnlt_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpnltps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpnlt_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpps(a, b, 5) }
}

/// Compares each of the four floats in `a` to the corresponding element in `b`.
/// The result in the output vector will be `0xffffffff` if the input element
/// in `a` is **not** less than or equal to the corresponding element in `b`, or
/// `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpnle_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpnleps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpnle_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpps(a, b, 6) }
}

/// Compares each of the four floats in `a` to the corresponding element in `b`.
/// The result in the output vector will be `0xffffffff` if the input element
/// in `a` is **not** greater than the corresponding element in `b`, or `0`
/// otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpngt_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpnltps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpngt_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpps(b, a, 5) }
}

/// Compares each of the four floats in `a` to the corresponding element in `b`.
/// The result in the output vector will be `0xffffffff` if the input element
/// in `a` is **not** greater than or equal to the corresponding element in `b`,
/// or `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpnge_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpnleps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpnge_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpps(b, a, 6) }
}

/// Compares each of the four floats in `a` to the corresponding element in `b`.
/// Returns four floats that have one of two possible bit patterns. The element
/// in the output vector will be `0xffffffff` if the input elements in `a` and
/// `b` are ordered (i.e., neither of them is a NaN), or 0 otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpord_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpordps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpord_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpps(b, a, 7) }
}

/// Compares each of the four floats in `a` to the corresponding element in `b`.
/// Returns four floats that have one of two possible bit patterns. The element
/// in the output vector will be `0xffffffff` if the input elements in `a` and
/// `b` are unordered (i.e., at least on of them is a NaN), or 0 otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpunord_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cmpunordps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpunord_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { cmpps(b, a, 3) }
}

/// Compares two 32-bit floats from the low-order bits of `a` and `b`. Returns
/// `1` if they are equal, or `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_comieq_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(comiss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_comieq_ss(a: __m128, b: __m128) -> i32 {
    unsafe { comieq_ss(a, b) }
}

/// Compares two 32-bit floats from the low-order bits of `a` and `b`. Returns
/// `1` if the value from `a` is less than the one from `b`, or `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_comilt_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(comiss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_comilt_ss(a: __m128, b: __m128) -> i32 {
    unsafe { comilt_ss(a, b) }
}

/// Compares two 32-bit floats from the low-order bits of `a` and `b`. Returns
/// `1` if the value from `a` is less than or equal to the one from `b`, or `0`
/// otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_comile_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(comiss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_comile_ss(a: __m128, b: __m128) -> i32 {
    unsafe { comile_ss(a, b) }
}

/// Compares two 32-bit floats from the low-order bits of `a` and `b`. Returns
/// `1` if the value from `a` is greater than the one from `b`, or `0`
/// otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_comigt_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(comiss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_comigt_ss(a: __m128, b: __m128) -> i32 {
    unsafe { comigt_ss(a, b) }
}

/// Compares two 32-bit floats from the low-order bits of `a` and `b`. Returns
/// `1` if the value from `a` is greater than or equal to the one from `b`, or
/// `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_comige_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(comiss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_comige_ss(a: __m128, b: __m128) -> i32 {
    unsafe { comige_ss(a, b) }
}

/// Compares two 32-bit floats from the low-order bits of `a` and `b`. Returns
/// `1` if they are **not** equal, or `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_comineq_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(comiss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_comineq_ss(a: __m128, b: __m128) -> i32 {
    unsafe { comineq_ss(a, b) }
}

/// Compares two 32-bit floats from the low-order bits of `a` and `b`. Returns
/// `1` if they are equal, or `0` otherwise. This instruction will not signal
/// an exception if either argument is a quiet NaN.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_ucomieq_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(ucomiss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_ucomieq_ss(a: __m128, b: __m128) -> i32 {
    unsafe { ucomieq_ss(a, b) }
}

/// Compares two 32-bit floats from the low-order bits of `a` and `b`. Returns
/// `1` if the value from `a` is less than the one from `b`, or `0` otherwise.
/// This instruction will not signal an exception if either argument is a quiet
/// NaN.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_ucomilt_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(ucomiss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_ucomilt_ss(a: __m128, b: __m128) -> i32 {
    unsafe { ucomilt_ss(a, b) }
}

/// Compares two 32-bit floats from the low-order bits of `a` and `b`. Returns
/// `1` if the value from `a` is less than or equal to the one from `b`, or `0`
/// otherwise. This instruction will not signal an exception if either argument
/// is a quiet NaN.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_ucomile_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(ucomiss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_ucomile_ss(a: __m128, b: __m128) -> i32 {
    unsafe { ucomile_ss(a, b) }
}

/// Compares two 32-bit floats from the low-order bits of `a` and `b`. Returns
/// `1` if the value from `a` is greater than the one from `b`, or `0`
/// otherwise. This instruction will not signal an exception if either argument
/// is a quiet NaN.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_ucomigt_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(ucomiss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_ucomigt_ss(a: __m128, b: __m128) -> i32 {
    unsafe { ucomigt_ss(a, b) }
}

/// Compares two 32-bit floats from the low-order bits of `a` and `b`. Returns
/// `1` if the value from `a` is greater than or equal to the one from `b`, or
/// `0` otherwise. This instruction will not signal an exception if either
/// argument is a quiet NaN.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_ucomige_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(ucomiss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_ucomige_ss(a: __m128, b: __m128) -> i32 {
    unsafe { ucomige_ss(a, b) }
}

/// Compares two 32-bit floats from the low-order bits of `a` and `b`. Returns
/// `1` if they are **not** equal, or `0` otherwise. This instruction will not
/// signal an exception if either argument is a quiet NaN.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_ucomineq_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(ucomiss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_ucomineq_ss(a: __m128, b: __m128) -> i32 {
    unsafe { ucomineq_ss(a, b) }
}

/// Converts the lowest 32 bit float in the input vector to a 32 bit integer.
///
/// The result is rounded according to the current rounding mode. If the result
/// cannot be represented as a 32 bit integer the result will be `0x8000_0000`
/// (`i32::MIN`).
///
/// This corresponds to the `CVTSS2SI` instruction (with 32 bit output).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtss_si32)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cvtss2si))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cvtss_si32(a: __m128) -> i32 {
    unsafe { cvtss2si(a) }
}

/// Alias for [`_mm_cvtss_si32`](fn._mm_cvtss_si32.html).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvt_ss2si)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cvtss2si))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cvt_ss2si(a: __m128) -> i32 {
    _mm_cvtss_si32(a)
}

/// Converts the lowest 32 bit float in the input vector to a 32 bit integer
/// with
/// truncation.
///
/// The result is rounded always using truncation (round towards zero). If the
/// result cannot be represented as a 32 bit integer the result will be
/// `0x8000_0000` (`i32::MIN`).
///
/// This corresponds to the `CVTTSS2SI` instruction (with 32 bit output).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvttss_si32)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cvttss2si))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cvttss_si32(a: __m128) -> i32 {
    unsafe { cvttss2si(a) }
}

/// Alias for [`_mm_cvttss_si32`](fn._mm_cvttss_si32.html).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtt_ss2si)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cvttss2si))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cvtt_ss2si(a: __m128) -> i32 {
    _mm_cvttss_si32(a)
}

/// Extracts the lowest 32 bit float from the input vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtss_f32)
#[inline]
#[target_feature(enable = "sse")]
// No point in using assert_instrs. In Unix x86_64 calling convention this is a
// no-op, and on msvc it's just a `mov`.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cvtss_f32(a: __m128) -> f32 {
    unsafe { simd_extract!(a, 0) }
}

/// Converts a 32 bit integer to a 32 bit float. The result vector is the input
/// vector `a` with the lowest 32 bit float replaced by the converted integer.
///
/// This intrinsic corresponds to the `CVTSI2SS` instruction (with 32 bit
/// input).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtsi32_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cvtsi2ss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cvtsi32_ss(a: __m128, b: i32) -> __m128 {
    unsafe { cvtsi2ss(a, b) }
}

/// Alias for [`_mm_cvtsi32_ss`](fn._mm_cvtsi32_ss.html).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvt_si2ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(cvtsi2ss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cvt_si2ss(a: __m128, b: i32) -> __m128 {
    _mm_cvtsi32_ss(a, b)
}

/// Construct a `__m128` with the lowest element set to `a` and the rest set to
/// zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_set_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(movss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_set_ss(a: f32) -> __m128 {
    __m128([a, 0.0, 0.0, 0.0])
}

/// Construct a `__m128` with all element set to `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_set1_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(shufps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_set1_ps(a: f32) -> __m128 {
    __m128([a, a, a, a])
}

/// Alias for [`_mm_set1_ps`](fn._mm_set1_ps.html)
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_set_ps1)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(shufps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_set_ps1(a: f32) -> __m128 {
    _mm_set1_ps(a)
}

/// Construct a `__m128` from four floating point values highest to lowest.
///
/// Note that `a` will be the highest 32 bits of the result, and `d` the
/// lowest. This matches the standard way of writing bit patterns on x86:
///
/// ```text
///  bit    127 .. 96  95 .. 64  63 .. 32  31 .. 0
///        +---------+---------+---------+---------+
///        |    a    |    b    |    c    |    d    |   result
///        +---------+---------+---------+---------+
/// ```
///
/// Alternatively:
///
/// ```text
/// let v = _mm_set_ps(d, c, b, a);
/// ```
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_set_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(unpcklps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_set_ps(a: f32, b: f32, c: f32, d: f32) -> __m128 {
    __m128([d, c, b, a])
}

/// Construct a `__m128` from four floating point values lowest to highest.
///
/// This matches the memory order of `__m128`, i.e., `a` will be the lowest 32
/// bits of the result, and `d` the highest.
///
/// ```text
/// assert_eq!(__m128::new(a, b, c, d), _mm_setr_ps(a, b, c, d));
/// ```
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_setr_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(
    all(test, any(target_env = "msvc", target_arch = "x86_64")),
    assert_instr(unpcklps)
)]
// On a 32-bit architecture on non-msvc it just copies the operands from the stack.
#[cfg_attr(
    all(test, all(not(target_env = "msvc"), target_arch = "x86")),
    assert_instr(movaps)
)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_setr_ps(a: f32, b: f32, c: f32, d: f32) -> __m128 {
    __m128([a, b, c, d])
}

/// Construct a `__m128` with all elements initialized to zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_setzero_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(xorps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_setzero_ps() -> __m128 {
    const { unsafe { mem::zeroed() } }
}

/// A utility function for creating masks to use with Intel shuffle and
/// permute intrinsics.
#[inline]
#[allow(non_snake_case)]
#[unstable(feature = "stdarch_x86_mm_shuffle", issue = "111147")]
pub const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

/// Shuffles packed single-precision (32-bit) floating-point elements in `a` and
/// `b` using `MASK`.
///
/// The lower half of result takes values from `a` and the higher half from
/// `b`. Mask is split to 2 control bits each to index the element from inputs.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shuffle_ps)
///
/// Note that there appears to be a mistake within Intel's Intrinsics Guide.
/// `_mm_shuffle_ps` is supposed to take an `i32` instead of a `u32`
/// as is the case for [other shuffle intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shuffle_).
/// Performing an implicit type conversion between an unsigned integer and a signed integer
/// does not cause a problem in C, however Rust's commitment to strong typing does not allow this.
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(shufps, MASK = 3))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_shuffle_ps<const MASK: i32>(a: __m128, b: __m128) -> __m128 {
    static_assert_uimm_bits!(MASK, 8);
    unsafe {
        simd_shuffle!(
            a,
            b,
            [
                MASK as u32 & 0b11,
                (MASK as u32 >> 2) & 0b11,
                ((MASK as u32 >> 4) & 0b11) + 4,
                ((MASK as u32 >> 6) & 0b11) + 4,
            ],
        )
    }
}

/// Unpacks and interleave single-precision (32-bit) floating-point elements
/// from the higher half of `a` and `b`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_unpackhi_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(unpckhps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_unpackhi_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_shuffle!(a, b, [2, 6, 3, 7]) }
}

/// Unpacks and interleave single-precision (32-bit) floating-point elements
/// from the lower half of `a` and `b`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_unpacklo_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(unpcklps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_unpacklo_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_shuffle!(a, b, [0, 4, 1, 5]) }
}

/// Combine higher half of `a` and `b`. The higher half of `b` occupies the
/// lower half of result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movehl_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(movhlps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_movehl_ps(a: __m128, b: __m128) -> __m128 {
    // TODO; figure why this is a different instruction on msvc?
    unsafe { simd_shuffle!(a, b, [6, 7, 2, 3]) }
}

/// Combine lower half of `a` and `b`. The lower half of `b` occupies the
/// higher half of result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movelh_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(movlhps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_movelh_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_shuffle!(a, b, [0, 1, 4, 5]) }
}

/// Returns a mask of the most significant bit of each element in `a`.
///
/// The mask is stored in the 4 least significant bits of the return value.
/// All other bits are set to `0`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movemask_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(movmskps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_movemask_ps(a: __m128) -> i32 {
    // Propagate the highest bit to the rest, because simd_bitmask
    // requires all-1 or all-0.
    unsafe {
        let mask: i32x4 = simd_lt(transmute(a), i32x4::ZERO);
        simd_bitmask::<i32x4, u8>(mask).into()
    }
}

/// Construct a `__m128` with the lowest element read from `p` and the other
/// elements set to zero.
///
/// This corresponds to instructions `VMOVSS` / `MOVSS`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(movss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_load_ss(p: *const f32) -> __m128 {
    __m128([*p, 0.0, 0.0, 0.0])
}

/// Construct a `__m128` by duplicating the value read from `p` into all
/// elements.
///
/// This corresponds to instructions `VMOVSS` / `MOVSS` followed by some
/// shuffling.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load1_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(movss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_load1_ps(p: *const f32) -> __m128 {
    let a = *p;
    __m128([a, a, a, a])
}

/// Alias for [`_mm_load1_ps`](fn._mm_load1_ps.html)
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load_ps1)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(movss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_load_ps1(p: *const f32) -> __m128 {
    _mm_load1_ps(p)
}

/// Loads four `f32` values from *aligned* memory into a `__m128`. If the
/// pointer is not aligned to a 128-bit boundary (16 bytes) a general
/// protection fault will be triggered (fatal program crash).
///
/// Use [`_mm_loadu_ps`](fn._mm_loadu_ps.html) for potentially unaligned
/// memory.
///
/// This corresponds to instructions `VMOVAPS` / `MOVAPS`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load_ps)
#[inline]
#[target_feature(enable = "sse")]
// FIXME: Rust doesn't emit alignment attributes for MSVC x86-32. Ref https://github.com/rust-lang/rust/pull/139261
// All aligned load/store intrinsics are affected
#[cfg_attr(
    all(test, not(all(target_arch = "x86", target_env = "msvc"))),
    assert_instr(movaps)
)]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow(clippy::cast_ptr_alignment)]
pub unsafe fn _mm_load_ps(p: *const f32) -> __m128 {
    *(p as *const __m128)
}

/// Loads four `f32` values from memory into a `__m128`. There are no
/// restrictions
/// on memory alignment. For aligned memory
/// [`_mm_load_ps`](fn._mm_load_ps.html)
/// may be faster.
///
/// This corresponds to instructions `VMOVUPS` / `MOVUPS`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(movups))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_loadu_ps(p: *const f32) -> __m128 {
    // Note: Using `*p` would require `f32` alignment, but `movups` has no
    // alignment restrictions.
    let mut dst = _mm_undefined_ps();
    ptr::copy_nonoverlapping(
        p as *const u8,
        ptr::addr_of_mut!(dst) as *mut u8,
        mem::size_of::<__m128>(),
    );
    dst
}

/// Loads four `f32` values from aligned memory into a `__m128` in reverse
/// order.
///
/// If the pointer is not aligned to a 128-bit boundary (16 bytes) a general
/// protection fault will be triggered (fatal program crash).
///
/// Functionally equivalent to the following code sequence (assuming `p`
/// satisfies the alignment restrictions):
///
/// ```text
/// let a0 = *p;
/// let a1 = *p.add(1);
/// let a2 = *p.add(2);
/// let a3 = *p.add(3);
/// __m128::new(a3, a2, a1, a0)
/// ```
///
/// This corresponds to instructions `VMOVAPS` / `MOVAPS` followed by some
/// shuffling.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadr_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(
    all(test, not(all(target_arch = "x86", target_env = "msvc"))),
    assert_instr(movaps)
)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_loadr_ps(p: *const f32) -> __m128 {
    let a = _mm_load_ps(p);
    simd_shuffle!(a, a, [3, 2, 1, 0])
}

/// Stores the lowest 32 bit float of `a` into memory.
///
/// This intrinsic corresponds to the `MOVSS` instruction.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_store_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(movss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_store_ss(p: *mut f32, a: __m128) {
    *p = simd_extract!(a, 0);
}

/// Stores the lowest 32 bit float of `a` repeated four times into *aligned*
/// memory.
///
/// If the pointer is not aligned to a 128-bit boundary (16 bytes) a general
/// protection fault will be triggered (fatal program crash).
///
/// Functionally equivalent to the following code sequence (assuming `p`
/// satisfies the alignment restrictions):
///
/// ```text
/// let x = a.extract(0);
/// *p = x;
/// *p.add(1) = x;
/// *p.add(2) = x;
/// *p.add(3) = x;
/// ```
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_store1_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(
    all(test, not(all(target_arch = "x86", target_env = "msvc"))),
    assert_instr(movaps)
)]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow(clippy::cast_ptr_alignment)]
pub unsafe fn _mm_store1_ps(p: *mut f32, a: __m128) {
    let b: __m128 = simd_shuffle!(a, a, [0, 0, 0, 0]);
    *(p as *mut __m128) = b;
}

/// Alias for [`_mm_store1_ps`](fn._mm_store1_ps.html)
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_store_ps1)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(
    all(test, not(all(target_arch = "x86", target_env = "msvc"))),
    assert_instr(movaps)
)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_store_ps1(p: *mut f32, a: __m128) {
    _mm_store1_ps(p, a);
}

/// Stores four 32-bit floats into *aligned* memory.
///
/// If the pointer is not aligned to a 128-bit boundary (16 bytes) a general
/// protection fault will be triggered (fatal program crash).
///
/// Use [`_mm_storeu_ps`](fn._mm_storeu_ps.html) for potentially unaligned
/// memory.
///
/// This corresponds to instructions `VMOVAPS` / `MOVAPS`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_store_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(
    all(test, not(all(target_arch = "x86", target_env = "msvc"))),
    assert_instr(movaps)
)]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow(clippy::cast_ptr_alignment)]
pub unsafe fn _mm_store_ps(p: *mut f32, a: __m128) {
    *(p as *mut __m128) = a;
}

/// Stores four 32-bit floats into memory. There are no restrictions on memory
/// alignment. For aligned memory [`_mm_store_ps`](fn._mm_store_ps.html) may be
/// faster.
///
/// This corresponds to instructions `VMOVUPS` / `MOVUPS`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(movups))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_storeu_ps(p: *mut f32, a: __m128) {
    ptr::copy_nonoverlapping(
        ptr::addr_of!(a) as *const u8,
        p as *mut u8,
        mem::size_of::<__m128>(),
    );
}

/// Stores four 32-bit floats into *aligned* memory in reverse order.
///
/// If the pointer is not aligned to a 128-bit boundary (16 bytes) a general
/// protection fault will be triggered (fatal program crash).
///
/// Functionally equivalent to the following code sequence (assuming `p`
/// satisfies the alignment restrictions):
///
/// ```text
/// *p = a.extract(3);
/// *p.add(1) = a.extract(2);
/// *p.add(2) = a.extract(1);
/// *p.add(3) = a.extract(0);
/// ```
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storer_ps)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(
    all(test, not(all(target_arch = "x86", target_env = "msvc"))),
    assert_instr(movaps)
)]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow(clippy::cast_ptr_alignment)]
pub unsafe fn _mm_storer_ps(p: *mut f32, a: __m128) {
    let b: __m128 = simd_shuffle!(a, a, [3, 2, 1, 0]);
    *(p as *mut __m128) = b;
}

/// Returns a `__m128` with the first component from `b` and the remaining
/// components from `a`.
///
/// In other words for any `a` and `b`:
/// ```text
/// _mm_move_ss(a, b) == a.replace(0, b.extract(0))
/// ```
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_move_ss)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(movss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_move_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { simd_shuffle!(a, b, [4, 1, 2, 3]) }
}

/// Performs a serializing operation on all non-temporal ("streaming") store instructions that
/// were issued by the current thread prior to this instruction.
///
/// Guarantees that every non-temporal store instruction that precedes this fence, in program order, is
/// ordered before any load or store instruction which follows the fence in
/// synchronization order.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sfence)
/// (but note that Intel is only documenting the hardware-level concerns related to this
/// instruction; the Intel documentation does not take into account the extra concerns that arise
/// because the Rust memory model is different from the x86 memory model.)
///
/// # Safety of non-temporal stores
///
/// After using any non-temporal store intrinsic, but before any other access to the memory that the
/// intrinsic mutates, a call to `_mm_sfence` must be performed on the thread that used the
/// intrinsic.
///
/// Non-temporal stores behave very different from regular stores. For the purpose of the Rust
/// memory model, these stores are happening asynchronously in a background thread. This means a
/// non-temporal store can cause data races with other accesses, even other accesses on the same
/// thread. It also means that cross-thread synchronization does not work as expected: let's say the
/// intrinsic is called on thread T1, and T1 performs synchronization with some other thread T2. The
/// non-temporal store acts as if it happened not in T1 but in a different thread T3, and T2 has not
/// synchronized with T3! Calling `_mm_sfence` makes the current thread wait for and synchronize
/// with all the non-temporal stores previously started on this thread, which means in particular
/// that subsequent synchronization with other threads will then work as intended again.
///
/// The general pattern to use non-temporal stores correctly is to call `_mm_sfence` before your
/// code jumps back to code outside your library. This ensures all stores inside your function
/// are synchronized-before the return, and thus transitively synchronized-before everything
/// the caller does after your function returns.
//
// The following is not a doc comment since it's not clear whether we want to put this into the
// docs, but it should be written out somewhere.
//
// Formally, we consider non-temporal stores and sfences to be opaque blobs that the compiler cannot
// inspect, and that behave like the following functions. This explains where the docs above come
// from.
// ```
// #[thread_local]
// static mut PENDING_NONTEMP_WRITES = AtomicUsize::new(0);
//
// pub unsafe fn nontemporal_store<T>(ptr: *mut T, val: T) {
//     PENDING_NONTEMP_WRITES.fetch_add(1, Relaxed);
//     // Spawn a thread that will eventually do our write.
//     // We need to fetch a pointer to this thread's pending-write
//     // counter, so that we can access it from the background thread.
//     let pending_writes = addr_of!(PENDING_NONTEMP_WRITES);
//     // If this was actual Rust code we'd have to do some extra work
//     // because `ptr`, `val`, `pending_writes` are all `!Send`. We skip that here.
//     std::thread::spawn(move || {
//         // Do the write in the background thread.
//         ptr.write(val);
//         // Register the write as done. Crucially, this is `Release`, so it
//         // syncs-with the `Acquire in `sfence`.
//         (&*pending_writes).fetch_sub(1, Release);
//     });
// }
//
// pub fn sfence() {
//     unsafe {
//         // Wait until there are no more pending writes.
//         while PENDING_NONTEMP_WRITES.load(Acquire) > 0 {}
//     }
// }
// ```
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(sfence))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_sfence() {
    sfence()
}

/// Gets the unsigned 32-bit value of the MXCSR control and status register.
///
/// Note that Rust makes no guarantees whatsoever about the contents of this register: Rust
/// floating-point operations may or may not result in this register getting updated with exception
/// state, and the register can change between two invocations of this function even when no
/// floating-point operations appear in the source code (since floating-point operations appearing
/// earlier or later can be reordered).
///
/// If you need to perform some floating-point operations and check whether they raised an
/// exception, use an inline assembly block for the entire sequence of operations.
///
/// For more info see [`_mm_setcsr`](fn._mm_setcsr.html)
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_getcsr)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(stmxcsr))]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[deprecated(
    since = "1.75.0",
    note = "see `_mm_getcsr` documentation - use inline assembly instead"
)]
pub unsafe fn _mm_getcsr() -> u32 {
    unsafe {
        let mut result = 0_i32;
        stmxcsr(ptr::addr_of_mut!(result) as *mut i8);
        result as u32
    }
}

/// Sets the MXCSR register with the 32-bit unsigned integer value.
///
/// This register controls how SIMD instructions handle floating point
/// operations. Modifying this register only affects the current thread.
///
/// It contains several groups of flags:
///
/// * *Exception flags* report which exceptions occurred since last they were reset.
///
/// * *Masking flags* can be used to mask (ignore) certain exceptions. By default
///   these flags are all set to 1, so all exceptions are masked. When
///   an exception is masked, the processor simply sets the exception flag and
///   continues the operation. If the exception is unmasked, the flag is also set
///   but additionally an exception handler is invoked.
///
/// * *Rounding mode flags* control the rounding mode of floating point
///   instructions.
///
/// * The *denormals-are-zero mode flag* turns all numbers which would be
///   denormalized (exponent bits are all zeros) into zeros.
///
/// Note that modifying the masking flags, rounding mode, or denormals-are-zero mode flags leads to
/// **immediate Undefined Behavior**: Rust assumes that these are always in their default state and
/// will optimize accordingly. This even applies when the register is altered and later reset to its
/// original value without any floating-point operations appearing in the source code between those
/// operations (since floating-point operations appearing earlier or later can be reordered).
///
/// If you need to perform some floating-point operations under a different masking flags, rounding
/// mode, or denormals-are-zero mode, use an inline assembly block and make sure to restore the
/// original MXCSR register state before the end of the block.
///
/// ## Exception Flags
///
/// * `_MM_EXCEPT_INVALID`: An invalid operation was performed (e.g., dividing
///   Infinity by Infinity).
///
/// * `_MM_EXCEPT_DENORM`: An operation attempted to operate on a denormalized
///   number. Mainly this can cause loss of precision.
///
/// * `_MM_EXCEPT_DIV_ZERO`: Division by zero occurred.
///
/// * `_MM_EXCEPT_OVERFLOW`: A numeric overflow exception occurred, i.e., a
///   result was too large to be represented (e.g., an `f32` with absolute
///   value greater than `2^128`).
///
/// * `_MM_EXCEPT_UNDERFLOW`: A numeric underflow exception occurred, i.e., a
///   result was too small to be represented in a normalized way (e.g., an
///   `f32` with absolute value smaller than `2^-126`.)
///
/// * `_MM_EXCEPT_INEXACT`: An inexact-result exception occurred (a.k.a.
///   precision exception). This means some precision was lost due to rounding.
///   For example, the fraction `1/3` cannot be represented accurately in a
///   32 or 64 bit float and computing it would cause this exception to be
///   raised. Precision exceptions are very common, so they are usually masked.
///
/// Exception flags can be read and set using the convenience functions
/// `_MM_GET_EXCEPTION_STATE` and `_MM_SET_EXCEPTION_STATE`. For example, to
/// check if an operation caused some overflow:
///
/// ```rust,ignore
/// _MM_SET_EXCEPTION_STATE(0); // clear all exception flags
///                             // perform calculations
/// if _MM_GET_EXCEPTION_STATE() & _MM_EXCEPT_OVERFLOW != 0 {
///     // handle overflow
/// }
/// ```
///
/// ## Masking Flags
///
/// There is one masking flag for each exception flag: `_MM_MASK_INVALID`,
/// `_MM_MASK_DENORM`, `_MM_MASK_DIV_ZERO`, `_MM_MASK_OVERFLOW`,
/// `_MM_MASK_UNDERFLOW`, `_MM_MASK_INEXACT`.
///
/// A single masking bit can be set via
///
/// ```rust,ignore
/// _MM_SET_EXCEPTION_MASK(_MM_MASK_UNDERFLOW);
/// ```
///
/// However, since mask bits are by default all set to 1, it is more common to
/// want to *disable* certain bits. For example, to unmask the underflow
/// exception, use:
///
/// ```rust,ignore
/// _mm_setcsr(_mm_getcsr() & !_MM_MASK_UNDERFLOW); // unmask underflow
/// exception
/// ```
///
/// Warning: an unmasked exception will cause an exception handler to be
/// called.
/// The standard handler will simply terminate the process. So, in this case
/// any underflow exception would terminate the current process with something
/// like `signal: 8, SIGFPE: erroneous arithmetic operation`.
///
/// ## Rounding Mode
///
/// The rounding mode is describe using two bits. It can be read and set using
/// the convenience wrappers `_MM_GET_ROUNDING_MODE()` and
/// `_MM_SET_ROUNDING_MODE(mode)`.
///
/// The rounding modes are:
///
/// * `_MM_ROUND_NEAREST`: (default) Round to closest to the infinite precision
///   value. If two values are equally close, round to even (i.e., least
///   significant bit will be zero).
///
/// * `_MM_ROUND_DOWN`: Round toward negative Infinity.
///
/// * `_MM_ROUND_UP`: Round toward positive Infinity.
///
/// * `_MM_ROUND_TOWARD_ZERO`: Round towards zero (truncate).
///
/// Example:
///
/// ```rust,ignore
/// _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN)
/// ```
///
/// ## Denormals-are-zero/Flush-to-zero Mode
///
/// If this bit is set, values that would be denormalized will be set to zero
/// instead. This is turned off by default.
///
/// You can read and enable/disable this mode via the helper functions
/// `_MM_GET_FLUSH_ZERO_MODE()` and `_MM_SET_FLUSH_ZERO_MODE()`:
///
/// ```rust,ignore
/// _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF); // turn off (default)
/// _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); // turn on
/// ```
///
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_setcsr)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(ldmxcsr))]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[deprecated(
    since = "1.75.0",
    note = "see `_mm_setcsr` documentation - use inline assembly instead"
)]
pub unsafe fn _mm_setcsr(val: u32) {
    ldmxcsr(ptr::addr_of!(val) as *const i8);
}

/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_EXCEPT_INVALID: u32 = 0x0001;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_EXCEPT_DENORM: u32 = 0x0002;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_EXCEPT_DIV_ZERO: u32 = 0x0004;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_EXCEPT_OVERFLOW: u32 = 0x0008;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_EXCEPT_UNDERFLOW: u32 = 0x0010;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_EXCEPT_INEXACT: u32 = 0x0020;
/// See [`_MM_GET_EXCEPTION_STATE`](fn._MM_GET_EXCEPTION_STATE.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_EXCEPT_MASK: u32 = 0x003f;

/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_MASK_INVALID: u32 = 0x0080;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_MASK_DENORM: u32 = 0x0100;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_MASK_DIV_ZERO: u32 = 0x0200;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_MASK_OVERFLOW: u32 = 0x0400;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_MASK_UNDERFLOW: u32 = 0x0800;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_MASK_INEXACT: u32 = 0x1000;
/// See [`_MM_GET_EXCEPTION_MASK`](fn._MM_GET_EXCEPTION_MASK.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_MASK_MASK: u32 = 0x1f80;

/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_ROUND_NEAREST: u32 = 0x0000;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_ROUND_DOWN: u32 = 0x2000;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_ROUND_UP: u32 = 0x4000;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_ROUND_TOWARD_ZERO: u32 = 0x6000;

/// See [`_MM_GET_ROUNDING_MODE`](fn._MM_GET_ROUNDING_MODE.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_ROUND_MASK: u32 = 0x6000;

/// See [`_MM_GET_FLUSH_ZERO_MODE`](fn._MM_GET_FLUSH_ZERO_MODE.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FLUSH_ZERO_MASK: u32 = 0x8000;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FLUSH_ZERO_ON: u32 = 0x8000;
/// See [`_mm_setcsr`](fn._mm_setcsr.html)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FLUSH_ZERO_OFF: u32 = 0x0000;

/// See [`_mm_setcsr`](fn._mm_setcsr.html)
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_MM_GET_EXCEPTION_MASK)
#[inline]
#[allow(deprecated)] // Deprecated function implemented on top of deprecated function
#[allow(non_snake_case)]
#[target_feature(enable = "sse")]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[deprecated(
    since = "1.75.0",
    note = "see `_mm_getcsr` documentation - use inline assembly instead"
)]
pub unsafe fn _MM_GET_EXCEPTION_MASK() -> u32 {
    _mm_getcsr() & _MM_MASK_MASK
}

/// See [`_mm_setcsr`](fn._mm_setcsr.html)
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_MM_GET_EXCEPTION_STATE)
#[inline]
#[allow(deprecated)] // Deprecated function implemented on top of deprecated function
#[allow(non_snake_case)]
#[target_feature(enable = "sse")]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[deprecated(
    since = "1.75.0",
    note = "see `_mm_getcsr` documentation - use inline assembly instead"
)]
pub unsafe fn _MM_GET_EXCEPTION_STATE() -> u32 {
    _mm_getcsr() & _MM_EXCEPT_MASK
}

/// See [`_mm_setcsr`](fn._mm_setcsr.html)
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_MM_GET_FLUSH_ZERO_MODE)
#[inline]
#[allow(deprecated)] // Deprecated function implemented on top of deprecated function
#[allow(non_snake_case)]
#[target_feature(enable = "sse")]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[deprecated(
    since = "1.75.0",
    note = "see `_mm_getcsr` documentation - use inline assembly instead"
)]
pub unsafe fn _MM_GET_FLUSH_ZERO_MODE() -> u32 {
    _mm_getcsr() & _MM_FLUSH_ZERO_MASK
}

/// See [`_mm_setcsr`](fn._mm_setcsr.html)
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_MM_GET_ROUNDING_MODE)
#[inline]
#[allow(deprecated)] // Deprecated function implemented on top of deprecated function
#[allow(non_snake_case)]
#[target_feature(enable = "sse")]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[deprecated(
    since = "1.75.0",
    note = "see `_mm_getcsr` documentation - use inline assembly instead"
)]
pub unsafe fn _MM_GET_ROUNDING_MODE() -> u32 {
    _mm_getcsr() & _MM_ROUND_MASK
}

/// See [`_mm_setcsr`](fn._mm_setcsr.html)
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_MM_SET_EXCEPTION_MASK)
#[inline]
#[allow(deprecated)] // Deprecated function implemented on top of deprecated function
#[allow(non_snake_case)]
#[target_feature(enable = "sse")]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[deprecated(
    since = "1.75.0",
    note = "see `_mm_setcsr` documentation - use inline assembly instead"
)]
pub unsafe fn _MM_SET_EXCEPTION_MASK(x: u32) {
    _mm_setcsr((_mm_getcsr() & !_MM_MASK_MASK) | (x & _MM_MASK_MASK))
}

/// See [`_mm_setcsr`](fn._mm_setcsr.html)
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_MM_SET_EXCEPTION_STATE)
#[inline]
#[allow(deprecated)] // Deprecated function implemented on top of deprecated function
#[allow(non_snake_case)]
#[target_feature(enable = "sse")]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[deprecated(
    since = "1.75.0",
    note = "see `_mm_setcsr` documentation - use inline assembly instead"
)]
pub unsafe fn _MM_SET_EXCEPTION_STATE(x: u32) {
    _mm_setcsr((_mm_getcsr() & !_MM_EXCEPT_MASK) | (x & _MM_EXCEPT_MASK))
}

/// See [`_mm_setcsr`](fn._mm_setcsr.html)
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_MM_SET_FLUSH_ZERO_MODE)
#[inline]
#[allow(deprecated)] // Deprecated function implemented on top of deprecated function
#[allow(non_snake_case)]
#[target_feature(enable = "sse")]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[deprecated(
    since = "1.75.0",
    note = "see `_mm_setcsr` documentation - use inline assembly instead"
)]
pub unsafe fn _MM_SET_FLUSH_ZERO_MODE(x: u32) {
    _mm_setcsr((_mm_getcsr() & !_MM_FLUSH_ZERO_MASK) | (x & _MM_FLUSH_ZERO_MASK))
}

/// See [`_mm_setcsr`](fn._mm_setcsr.html)
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_MM_SET_ROUNDING_MODE)
#[inline]
#[allow(deprecated)] // Deprecated function implemented on top of deprecated function
#[allow(non_snake_case)]
#[target_feature(enable = "sse")]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[deprecated(
    since = "1.75.0",
    note = "see `_mm_setcsr` documentation - use inline assembly instead"
)]
pub unsafe fn _MM_SET_ROUNDING_MODE(x: u32) {
    _mm_setcsr((_mm_getcsr() & !_MM_ROUND_MASK) | (x & _MM_ROUND_MASK))
}

/// See [`_mm_prefetch`](fn._mm_prefetch.html).
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_HINT_T0: i32 = 3;

/// See [`_mm_prefetch`](fn._mm_prefetch.html).
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_HINT_T1: i32 = 2;

/// See [`_mm_prefetch`](fn._mm_prefetch.html).
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_HINT_T2: i32 = 1;

/// See [`_mm_prefetch`](fn._mm_prefetch.html).
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_HINT_NTA: i32 = 0;

/// See [`_mm_prefetch`](fn._mm_prefetch.html).
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_HINT_ET0: i32 = 7;

/// See [`_mm_prefetch`](fn._mm_prefetch.html).
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_HINT_ET1: i32 = 6;

/// Fetch the cache line that contains address `p` using the given `STRATEGY`.
///
/// The `STRATEGY` must be one of:
///
/// * [`_MM_HINT_T0`](constant._MM_HINT_T0.html): Fetch into all levels of the
///   cache hierarchy.
///
/// * [`_MM_HINT_T1`](constant._MM_HINT_T1.html): Fetch into L2 and higher.
///
/// * [`_MM_HINT_T2`](constant._MM_HINT_T2.html): Fetch into L3 and higher or
///   an implementation-specific choice (e.g., L2 if there is no L3).
///
/// * [`_MM_HINT_NTA`](constant._MM_HINT_NTA.html): Fetch data using the
///   non-temporal access (NTA) hint. It may be a place closer than main memory
///   but outside of the cache hierarchy. This is used to reduce access latency
///   without polluting the cache.
///
/// * [`_MM_HINT_ET0`](constant._MM_HINT_ET0.html) and
///   [`_MM_HINT_ET1`](constant._MM_HINT_ET1.html) are similar to `_MM_HINT_T0`
///   and `_MM_HINT_T1` but indicate an anticipation to write to the address.
///
/// The actual implementation depends on the particular CPU. This instruction
/// is considered a hint, so the CPU is also free to simply ignore the request.
///
/// The amount of prefetched data depends on the cache line size of the
/// specific CPU, but it will be at least 32 bytes.
///
/// Common caveats:
///
/// * Most modern CPUs already automatically prefetch data based on predicted
///   access patterns.
///
/// * Data is usually not fetched if this would cause a TLB miss or a page
///   fault.
///
/// * Too much prefetching can cause unnecessary cache evictions.
///
/// * Prefetching may also fail if there are not enough memory-subsystem
///   resources (e.g., request buffers).
///
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_prefetch)
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(prefetcht0, STRATEGY = _MM_HINT_T0))]
#[cfg_attr(test, assert_instr(prefetcht1, STRATEGY = _MM_HINT_T1))]
#[cfg_attr(test, assert_instr(prefetcht2, STRATEGY = _MM_HINT_T2))]
#[cfg_attr(test, assert_instr(prefetchnta, STRATEGY = _MM_HINT_NTA))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_prefetch<const STRATEGY: i32>(p: *const i8) {
    static_assert_uimm_bits!(STRATEGY, 3);
    // We use the `llvm.prefetch` intrinsic with `cache type` = 1 (data cache).
    // `locality` and `rw` are based on our `STRATEGY`.
    prefetch(p, (STRATEGY >> 2) & 1, STRATEGY & 3, 1);
}

/// Returns vector of type __m128 with indeterminate elements.with indetermination elements.
/// Despite using the word "undefined" (following Intel's naming scheme), this non-deterministically
/// picks some valid value and is not equivalent to [`mem::MaybeUninit`].
/// In practice, this is typically equivalent to [`mem::zeroed`].
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_undefined_ps)
#[inline]
#[target_feature(enable = "sse")]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_undefined_ps() -> __m128 {
    const { unsafe { mem::zeroed() } }
}

/// Transpose the 4x4 matrix formed by 4 rows of __m128 in place.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_MM_TRANSPOSE4_PS)
#[inline]
#[allow(non_snake_case)]
#[target_feature(enable = "sse")]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _MM_TRANSPOSE4_PS(
    row0: &mut __m128,
    row1: &mut __m128,
    row2: &mut __m128,
    row3: &mut __m128,
) {
    let tmp0 = _mm_unpacklo_ps(*row0, *row1);
    let tmp2 = _mm_unpacklo_ps(*row2, *row3);
    let tmp1 = _mm_unpackhi_ps(*row0, *row1);
    let tmp3 = _mm_unpackhi_ps(*row2, *row3);

    *row0 = _mm_movelh_ps(tmp0, tmp2);
    *row1 = _mm_movehl_ps(tmp2, tmp0);
    *row2 = _mm_movelh_ps(tmp1, tmp3);
    *row3 = _mm_movehl_ps(tmp3, tmp1);
}

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.x86.sse.rcp.ss"]
    fn rcpss(a: __m128) -> __m128;
    #[link_name = "llvm.x86.sse.rcp.ps"]
    fn rcpps(a: __m128) -> __m128;
    #[link_name = "llvm.x86.sse.rsqrt.ss"]
    fn rsqrtss(a: __m128) -> __m128;
    #[link_name = "llvm.x86.sse.rsqrt.ps"]
    fn rsqrtps(a: __m128) -> __m128;
    #[link_name = "llvm.x86.sse.min.ss"]
    fn minss(a: __m128, b: __m128) -> __m128;
    #[link_name = "llvm.x86.sse.min.ps"]
    fn minps(a: __m128, b: __m128) -> __m128;
    #[link_name = "llvm.x86.sse.max.ss"]
    fn maxss(a: __m128, b: __m128) -> __m128;
    #[link_name = "llvm.x86.sse.max.ps"]
    fn maxps(a: __m128, b: __m128) -> __m128;
    #[link_name = "llvm.x86.sse.cmp.ps"]
    fn cmpps(a: __m128, b: __m128, imm8: i8) -> __m128;
    #[link_name = "llvm.x86.sse.comieq.ss"]
    fn comieq_ss(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.sse.comilt.ss"]
    fn comilt_ss(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.sse.comile.ss"]
    fn comile_ss(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.sse.comigt.ss"]
    fn comigt_ss(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.sse.comige.ss"]
    fn comige_ss(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.sse.comineq.ss"]
    fn comineq_ss(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomieq.ss"]
    fn ucomieq_ss(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomilt.ss"]
    fn ucomilt_ss(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomile.ss"]
    fn ucomile_ss(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomigt.ss"]
    fn ucomigt_ss(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomige.ss"]
    fn ucomige_ss(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.sse.ucomineq.ss"]
    fn ucomineq_ss(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.sse.cvtss2si"]
    fn cvtss2si(a: __m128) -> i32;
    #[link_name = "llvm.x86.sse.cvttss2si"]
    fn cvttss2si(a: __m128) -> i32;
    #[link_name = "llvm.x86.sse.cvtsi2ss"]
    fn cvtsi2ss(a: __m128, b: i32) -> __m128;
    #[link_name = "llvm.x86.sse.sfence"]
    fn sfence();
    #[link_name = "llvm.x86.sse.stmxcsr"]
    fn stmxcsr(p: *mut i8);
    #[link_name = "llvm.x86.sse.ldmxcsr"]
    fn ldmxcsr(p: *const i8);
    #[link_name = "llvm.prefetch"]
    fn prefetch(p: *const i8, rw: i32, loc: i32, ty: i32);
    #[link_name = "llvm.x86.sse.cmp.ss"]
    fn cmpss(a: __m128, b: __m128, imm8: i8) -> __m128;
}

/// Stores `a` into the memory at `mem_addr` using a non-temporal memory hint.
///
/// `mem_addr` must be aligned on a 16-byte boundary or a general-protection
/// exception _may_ be generated.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_stream_ps)
///
/// # Safety of non-temporal stores
///
/// After using this intrinsic, but before any other access to the memory that this intrinsic
/// mutates, a call to [`_mm_sfence`] must be performed by the thread that used the intrinsic. In
/// particular, functions that call this intrinsic should generally call `_mm_sfence` before they
/// return.
///
/// See [`_mm_sfence`] for details.
#[inline]
#[target_feature(enable = "sse")]
#[cfg_attr(test, assert_instr(movntps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow(clippy::cast_ptr_alignment)]
pub unsafe fn _mm_stream_ps(mem_addr: *mut f32, a: __m128) {
    crate::arch::asm!(
        vps!("movntps", ",{a}"),
        p = in(reg) mem_addr,
        a = in(xmm_reg) a,
        options(nostack, preserves_flags),
    );
}

#[cfg(test)]
mod tests {
    use crate::{hint::black_box, mem::transmute, ptr};
    use std::boxed;
    use stdarch_test::simd_test;

    use crate::core_arch::{simd::*, x86::*};

    const NAN: f32 = f32::NAN;

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_add_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_add_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(-101.0, 25.0, 0.0, -15.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_add_ss() {
        let a = _mm_set_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_set_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_add_ss(a, b);
        assert_eq_m128(r, _mm_set_ps(-1.0, 5.0, 0.0, -15.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_sub_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_sub_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(99.0, -15.0, 0.0, -5.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_sub_ss() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_sub_ss(a, b);
        assert_eq_m128(r, _mm_setr_ps(99.0, 5.0, 0.0, -10.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_mul_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_mul_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(100.0, 100.0, 0.0, 50.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_mul_ss() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_mul_ss(a, b);
        assert_eq_m128(r, _mm_setr_ps(100.0, 5.0, 0.0, -10.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_div_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 2.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.2, -5.0);
        let r = _mm_div_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(0.01, 0.25, 10.0, 2.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_div_ss() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_div_ss(a, b);
        assert_eq_m128(r, _mm_setr_ps(0.01, 5.0, 0.0, -10.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_sqrt_ss() {
        let a = _mm_setr_ps(4.0, 13.0, 16.0, 100.0);
        let r = _mm_sqrt_ss(a);
        let e = _mm_setr_ps(2.0, 13.0, 16.0, 100.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_sqrt_ps() {
        let a = _mm_setr_ps(4.0, 13.0, 16.0, 100.0);
        let r = _mm_sqrt_ps(a);
        let e = _mm_setr_ps(2.0, 3.6055512, 4.0, 10.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_rcp_ss() {
        let a = _mm_setr_ps(4.0, 13.0, 16.0, 100.0);
        let r = _mm_rcp_ss(a);
        let e = _mm_setr_ps(0.24993896, 13.0, 16.0, 100.0);
        let rel_err = 0.00048828125;
        assert_approx_eq!(get_m128(r, 0), get_m128(e, 0), 2. * rel_err);
        for i in 1..4 {
            assert_eq!(get_m128(r, i), get_m128(e, i));
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_rcp_ps() {
        let a = _mm_setr_ps(4.0, 13.0, 16.0, 100.0);
        let r = _mm_rcp_ps(a);
        let e = _mm_setr_ps(0.24993896, 0.0769043, 0.06248474, 0.0099983215);
        let rel_err = 0.00048828125;
        for i in 0..4 {
            assert_approx_eq!(get_m128(r, i), get_m128(e, i), 2. * rel_err);
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_rsqrt_ss() {
        let a = _mm_setr_ps(4.0, 13.0, 16.0, 100.0);
        let r = _mm_rsqrt_ss(a);
        let e = _mm_setr_ps(0.49987793, 13.0, 16.0, 100.0);
        let rel_err = 0.00048828125;
        for i in 0..4 {
            assert_approx_eq!(get_m128(r, i), get_m128(e, i), 2. * rel_err);
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_rsqrt_ps() {
        let a = _mm_setr_ps(4.0, 13.0, 16.0, 100.0);
        let r = _mm_rsqrt_ps(a);
        let e = _mm_setr_ps(0.49987793, 0.2772827, 0.24993896, 0.099990845);
        let rel_err = 0.00048828125;
        for i in 0..4 {
            assert_approx_eq!(get_m128(r, i), get_m128(e, i), 2. * rel_err);
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_min_ss() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_min_ss(a, b);
        assert_eq_m128(r, _mm_setr_ps(-100.0, 5.0, 0.0, -10.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_min_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_min_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(-100.0, 5.0, 0.0, -10.0));

        // `_mm_min_ps` can **not** be implemented using the `simd_min` rust intrinsic. `simd_min`
        // is lowered by the llvm codegen backend to `llvm.minnum.v*` llvm intrinsic. This intrinsic
        // doesn't specify how -0.0 is handled. Unfortunately it happens to behave different from
        // the `minps` x86 instruction on x86. The `llvm.minnum.v*` llvm intrinsic equals
        // `r1` to `a` and `r2` to `b`.
        let a = _mm_setr_ps(-0.0, 0.0, 0.0, 0.0);
        let b = _mm_setr_ps(0.0, 0.0, 0.0, 0.0);
        let r1: [u8; 16] = transmute(_mm_min_ps(a, b));
        let r2: [u8; 16] = transmute(_mm_min_ps(b, a));
        let a: [u8; 16] = transmute(a);
        let b: [u8; 16] = transmute(b);
        assert_eq!(r1, b);
        assert_eq!(r2, a);
        assert_ne!(a, b); // sanity check that -0.0 is actually present
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_max_ss() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_max_ss(a, b);
        assert_eq_m128(r, _mm_setr_ps(-1.0, 5.0, 0.0, -10.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_max_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_max_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(-1.0, 20.0, 0.0, -5.0));

        // Check SSE-specific semantics for -0.0 handling.
        let a = _mm_setr_ps(-0.0, 0.0, 0.0, 0.0);
        let b = _mm_setr_ps(0.0, 0.0, 0.0, 0.0);
        let r1: [u8; 16] = transmute(_mm_max_ps(a, b));
        let r2: [u8; 16] = transmute(_mm_max_ps(b, a));
        let a: [u8; 16] = transmute(a);
        let b: [u8; 16] = transmute(b);
        assert_eq!(r1, b);
        assert_eq!(r2, a);
        assert_ne!(a, b); // sanity check that -0.0 is actually present
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_and_ps() {
        let a = transmute(u32x4::splat(0b0011));
        let b = transmute(u32x4::splat(0b0101));
        let r = _mm_and_ps(*black_box(&a), *black_box(&b));
        let e = transmute(u32x4::splat(0b0001));
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_andnot_ps() {
        let a = transmute(u32x4::splat(0b0011));
        let b = transmute(u32x4::splat(0b0101));
        let r = _mm_andnot_ps(*black_box(&a), *black_box(&b));
        let e = transmute(u32x4::splat(0b0100));
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_or_ps() {
        let a = transmute(u32x4::splat(0b0011));
        let b = transmute(u32x4::splat(0b0101));
        let r = _mm_or_ps(*black_box(&a), *black_box(&b));
        let e = transmute(u32x4::splat(0b0111));
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_xor_ps() {
        let a = transmute(u32x4::splat(0b0011));
        let b = transmute(u32x4::splat(0b0101));
        let r = _mm_xor_ps(*black_box(&a), *black_box(&b));
        let e = transmute(u32x4::splat(0b0110));
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpeq_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(-1.0, 5.0, 6.0, 7.0);
        let r: u32x4 = transmute(_mm_cmpeq_ss(a, b));
        let e: u32x4 = transmute(_mm_setr_ps(f32::from_bits(0), 2.0, 3.0, 4.0));
        assert_eq!(r, e);

        let b2 = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let r2: u32x4 = transmute(_mm_cmpeq_ss(a, b2));
        let e2: u32x4 = transmute(_mm_setr_ps(f32::from_bits(0xffffffff), 2.0, 3.0, 4.0));
        assert_eq!(r2, e2);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmplt_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = 0u32; // a.extract(0) < b.extract(0)
        let c1 = 0u32; // a.extract(0) < c.extract(0)
        let d1 = !0u32; // a.extract(0) < d.extract(0)

        let rb: u32x4 = transmute(_mm_cmplt_ss(a, b));
        let eb: u32x4 = transmute(_mm_setr_ps(f32::from_bits(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: u32x4 = transmute(_mm_cmplt_ss(a, c));
        let ec: u32x4 = transmute(_mm_setr_ps(f32::from_bits(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: u32x4 = transmute(_mm_cmplt_ss(a, d));
        let ed: u32x4 = transmute(_mm_setr_ps(f32::from_bits(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmple_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = 0u32; // a.extract(0) <= b.extract(0)
        let c1 = !0u32; // a.extract(0) <= c.extract(0)
        let d1 = !0u32; // a.extract(0) <= d.extract(0)

        let rb: u32x4 = transmute(_mm_cmple_ss(a, b));
        let eb: u32x4 = transmute(_mm_setr_ps(f32::from_bits(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: u32x4 = transmute(_mm_cmple_ss(a, c));
        let ec: u32x4 = transmute(_mm_setr_ps(f32::from_bits(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: u32x4 = transmute(_mm_cmple_ss(a, d));
        let ed: u32x4 = transmute(_mm_setr_ps(f32::from_bits(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpgt_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = !0u32; // a.extract(0) > b.extract(0)
        let c1 = 0u32; // a.extract(0) > c.extract(0)
        let d1 = 0u32; // a.extract(0) > d.extract(0)

        let rb: u32x4 = transmute(_mm_cmpgt_ss(a, b));
        let eb: u32x4 = transmute(_mm_setr_ps(f32::from_bits(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: u32x4 = transmute(_mm_cmpgt_ss(a, c));
        let ec: u32x4 = transmute(_mm_setr_ps(f32::from_bits(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: u32x4 = transmute(_mm_cmpgt_ss(a, d));
        let ed: u32x4 = transmute(_mm_setr_ps(f32::from_bits(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpge_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = !0u32; // a.extract(0) >= b.extract(0)
        let c1 = !0u32; // a.extract(0) >= c.extract(0)
        let d1 = 0u32; // a.extract(0) >= d.extract(0)

        let rb: u32x4 = transmute(_mm_cmpge_ss(a, b));
        let eb: u32x4 = transmute(_mm_setr_ps(f32::from_bits(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: u32x4 = transmute(_mm_cmpge_ss(a, c));
        let ec: u32x4 = transmute(_mm_setr_ps(f32::from_bits(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: u32x4 = transmute(_mm_cmpge_ss(a, d));
        let ed: u32x4 = transmute(_mm_setr_ps(f32::from_bits(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpneq_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = !0u32; // a.extract(0) != b.extract(0)
        let c1 = 0u32; // a.extract(0) != c.extract(0)
        let d1 = !0u32; // a.extract(0) != d.extract(0)

        let rb: u32x4 = transmute(_mm_cmpneq_ss(a, b));
        let eb: u32x4 = transmute(_mm_setr_ps(f32::from_bits(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: u32x4 = transmute(_mm_cmpneq_ss(a, c));
        let ec: u32x4 = transmute(_mm_setr_ps(f32::from_bits(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: u32x4 = transmute(_mm_cmpneq_ss(a, d));
        let ed: u32x4 = transmute(_mm_setr_ps(f32::from_bits(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpnlt_ss() {
        // TODO: this test is exactly the same as for `_mm_cmpge_ss`, but there
        // must be a difference. It may have to do with behavior in the
        // presence of NaNs (signaling or quiet). If so, we should add tests
        // for those.

        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = !0u32; // a.extract(0) >= b.extract(0)
        let c1 = !0u32; // a.extract(0) >= c.extract(0)
        let d1 = 0u32; // a.extract(0) >= d.extract(0)

        let rb: u32x4 = transmute(_mm_cmpnlt_ss(a, b));
        let eb: u32x4 = transmute(_mm_setr_ps(f32::from_bits(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: u32x4 = transmute(_mm_cmpnlt_ss(a, c));
        let ec: u32x4 = transmute(_mm_setr_ps(f32::from_bits(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: u32x4 = transmute(_mm_cmpnlt_ss(a, d));
        let ed: u32x4 = transmute(_mm_setr_ps(f32::from_bits(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpnle_ss() {
        // TODO: this test is exactly the same as for `_mm_cmpgt_ss`, but there
        // must be a difference. It may have to do with behavior in the
        // presence
        // of NaNs (signaling or quiet). If so, we should add tests for those.

        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = !0u32; // a.extract(0) > b.extract(0)
        let c1 = 0u32; // a.extract(0) > c.extract(0)
        let d1 = 0u32; // a.extract(0) > d.extract(0)

        let rb: u32x4 = transmute(_mm_cmpnle_ss(a, b));
        let eb: u32x4 = transmute(_mm_setr_ps(f32::from_bits(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: u32x4 = transmute(_mm_cmpnle_ss(a, c));
        let ec: u32x4 = transmute(_mm_setr_ps(f32::from_bits(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: u32x4 = transmute(_mm_cmpnle_ss(a, d));
        let ed: u32x4 = transmute(_mm_setr_ps(f32::from_bits(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpngt_ss() {
        // TODO: this test is exactly the same as for `_mm_cmple_ss`, but there
        // must be a difference. It may have to do with behavior in the
        // presence of NaNs (signaling or quiet). If so, we should add tests
        // for those.

        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = 0u32; // a.extract(0) <= b.extract(0)
        let c1 = !0u32; // a.extract(0) <= c.extract(0)
        let d1 = !0u32; // a.extract(0) <= d.extract(0)

        let rb: u32x4 = transmute(_mm_cmpngt_ss(a, b));
        let eb: u32x4 = transmute(_mm_setr_ps(f32::from_bits(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: u32x4 = transmute(_mm_cmpngt_ss(a, c));
        let ec: u32x4 = transmute(_mm_setr_ps(f32::from_bits(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: u32x4 = transmute(_mm_cmpngt_ss(a, d));
        let ed: u32x4 = transmute(_mm_setr_ps(f32::from_bits(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpnge_ss() {
        // TODO: this test is exactly the same as for `_mm_cmplt_ss`, but there
        // must be a difference. It may have to do with behavior in the
        // presence of NaNs (signaling or quiet). If so, we should add tests
        // for those.

        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = 0u32; // a.extract(0) < b.extract(0)
        let c1 = 0u32; // a.extract(0) < c.extract(0)
        let d1 = !0u32; // a.extract(0) < d.extract(0)

        let rb: u32x4 = transmute(_mm_cmpnge_ss(a, b));
        let eb: u32x4 = transmute(_mm_setr_ps(f32::from_bits(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: u32x4 = transmute(_mm_cmpnge_ss(a, c));
        let ec: u32x4 = transmute(_mm_setr_ps(f32::from_bits(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: u32x4 = transmute(_mm_cmpnge_ss(a, d));
        let ed: u32x4 = transmute(_mm_setr_ps(f32::from_bits(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpord_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(NAN, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = !0u32; // a.extract(0) ord b.extract(0)
        let c1 = 0u32; // a.extract(0) ord c.extract(0)
        let d1 = !0u32; // a.extract(0) ord d.extract(0)

        let rb: u32x4 = transmute(_mm_cmpord_ss(a, b));
        let eb: u32x4 = transmute(_mm_setr_ps(f32::from_bits(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: u32x4 = transmute(_mm_cmpord_ss(a, c));
        let ec: u32x4 = transmute(_mm_setr_ps(f32::from_bits(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: u32x4 = transmute(_mm_cmpord_ss(a, d));
        let ed: u32x4 = transmute(_mm_setr_ps(f32::from_bits(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpunord_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(NAN, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = 0u32; // a.extract(0) unord b.extract(0)
        let c1 = !0u32; // a.extract(0) unord c.extract(0)
        let d1 = 0u32; // a.extract(0) unord d.extract(0)

        let rb: u32x4 = transmute(_mm_cmpunord_ss(a, b));
        let eb: u32x4 = transmute(_mm_setr_ps(f32::from_bits(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: u32x4 = transmute(_mm_cmpunord_ss(a, c));
        let ec: u32x4 = transmute(_mm_setr_ps(f32::from_bits(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: u32x4 = transmute(_mm_cmpunord_ss(a, d));
        let ed: u32x4 = transmute(_mm_setr_ps(f32::from_bits(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpeq_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, NAN);
        let tru = !0u32;
        let fls = 0u32;

        let e = u32x4::new(fls, fls, tru, fls);
        let r: u32x4 = transmute(_mm_cmpeq_ps(a, b));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmplt_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, NAN);
        let tru = !0u32;
        let fls = 0u32;

        let e = u32x4::new(tru, fls, fls, fls);
        let r: u32x4 = transmute(_mm_cmplt_ps(a, b));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmple_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, 4.0);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, NAN);
        let tru = !0u32;
        let fls = 0u32;

        let e = u32x4::new(tru, fls, tru, fls);
        let r: u32x4 = transmute(_mm_cmple_ps(a, b));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpgt_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, 42.0);
        let tru = !0u32;
        let fls = 0u32;

        let e = u32x4::new(fls, tru, fls, fls);
        let r: u32x4 = transmute(_mm_cmpgt_ps(a, b));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpge_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, 42.0);
        let tru = !0u32;
        let fls = 0u32;

        let e = u32x4::new(fls, tru, tru, fls);
        let r: u32x4 = transmute(_mm_cmpge_ps(a, b));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpneq_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, NAN);
        let tru = !0u32;
        let fls = 0u32;

        let e = u32x4::new(tru, tru, fls, tru);
        let r: u32x4 = transmute(_mm_cmpneq_ps(a, b));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpnlt_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, 5.0);
        let tru = !0u32;
        let fls = 0u32;

        let e = u32x4::new(fls, tru, tru, tru);
        let r: u32x4 = transmute(_mm_cmpnlt_ps(a, b));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpnle_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, 5.0);
        let tru = !0u32;
        let fls = 0u32;

        let e = u32x4::new(fls, tru, fls, tru);
        let r: u32x4 = transmute(_mm_cmpnle_ps(a, b));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpngt_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, 5.0);
        let tru = !0u32;
        let fls = 0u32;

        let e = u32x4::new(tru, fls, tru, tru);
        let r: u32x4 = transmute(_mm_cmpngt_ps(a, b));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpnge_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, 5.0);
        let tru = !0u32;
        let fls = 0u32;

        let e = u32x4::new(tru, fls, fls, tru);
        let r: u32x4 = transmute(_mm_cmpnge_ps(a, b));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpord_ps() {
        let a = _mm_setr_ps(10.0, 50.0, NAN, NAN);
        let b = _mm_setr_ps(15.0, NAN, 1.0, NAN);
        let tru = !0u32;
        let fls = 0u32;

        let e = u32x4::new(tru, fls, fls, fls);
        let r: u32x4 = transmute(_mm_cmpord_ps(a, b));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cmpunord_ps() {
        let a = _mm_setr_ps(10.0, 50.0, NAN, NAN);
        let b = _mm_setr_ps(15.0, NAN, 1.0, NAN);
        let tru = !0u32;
        let fls = 0u32;

        let e = u32x4::new(fls, tru, tru, tru);
        let r: u32x4 = transmute(_mm_cmpunord_ps(a, b));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_comieq_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[1i32, 0, 0, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_comieq_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_comieq_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_comilt_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[0i32, 1, 0, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_comilt_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_comilt_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_comile_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[1i32, 1, 0, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_comile_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_comile_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_comigt_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[1i32, 0, 1, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_comige_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_comige_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_comineq_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[0i32, 1, 1, 1];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_comineq_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_comineq_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_ucomieq_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[1i32, 0, 0, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_ucomieq_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_ucomieq_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_ucomilt_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[0i32, 1, 0, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_ucomilt_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_ucomilt_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_ucomile_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[1i32, 1, 0, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_ucomile_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_ucomile_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_ucomigt_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[0i32, 0, 1, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_ucomigt_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_ucomigt_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_ucomige_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[1i32, 0, 1, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_ucomige_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_ucomige_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_ucomineq_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[0i32, 1, 1, 1];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_ucomineq_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_ucomineq_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cvtss_si32() {
        let inputs = &[42.0f32, -3.1, 4.0e10, 4.0e-20, NAN, 2147483500.1];
        let result = &[42i32, -3, i32::MIN, 0, i32::MIN, 2147483520];
        for i in 0..inputs.len() {
            let x = _mm_setr_ps(inputs[i], 1.0, 3.0, 4.0);
            let e = result[i];
            let r = _mm_cvtss_si32(x);
            assert_eq!(
                e, r,
                "TestCase #{} _mm_cvtss_si32({:?}) = {}, expected: {}",
                i, x, r, e
            );
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cvttss_si32() {
        let inputs = &[
            (42.0f32, 42i32),
            (-31.4, -31),
            (-33.5, -33),
            (-34.5, -34),
            (10.999, 10),
            (-5.99, -5),
            (4.0e10, i32::MIN),
            (4.0e-10, 0),
            (NAN, i32::MIN),
            (2147483500.1, 2147483520),
        ];
        for (i, &(xi, e)) in inputs.iter().enumerate() {
            let x = _mm_setr_ps(xi, 1.0, 3.0, 4.0);
            let r = _mm_cvttss_si32(x);
            assert_eq!(
                e, r,
                "TestCase #{} _mm_cvttss_si32({:?}) = {}, expected: {}",
                i, x, r, e
            );
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cvtsi32_ss() {
        let inputs = &[
            (4555i32, 4555.0f32),
            (322223333, 322223330.0),
            (-432, -432.0),
            (-322223333, -322223330.0),
        ];

        for &(x, f) in inputs.iter() {
            let a = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);
            let r = _mm_cvtsi32_ss(a, x);
            let e = _mm_setr_ps(f, 6.0, 7.0, 8.0);
            assert_eq_m128(e, r);
        }
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_cvtss_f32() {
        let a = _mm_setr_ps(312.0134, 5.0, 6.0, 7.0);
        assert_eq!(_mm_cvtss_f32(a), 312.0134);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_set_ss() {
        let r = _mm_set_ss(black_box(4.25));
        assert_eq_m128(r, _mm_setr_ps(4.25, 0.0, 0.0, 0.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_set1_ps() {
        let r1 = _mm_set1_ps(black_box(4.25));
        let r2 = _mm_set_ps1(black_box(4.25));
        assert_eq!(get_m128(r1, 0), 4.25);
        assert_eq!(get_m128(r1, 1), 4.25);
        assert_eq!(get_m128(r1, 2), 4.25);
        assert_eq!(get_m128(r1, 3), 4.25);
        assert_eq!(get_m128(r2, 0), 4.25);
        assert_eq!(get_m128(r2, 1), 4.25);
        assert_eq!(get_m128(r2, 2), 4.25);
        assert_eq!(get_m128(r2, 3), 4.25);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_set_ps() {
        let r = _mm_set_ps(
            black_box(1.0),
            black_box(2.0),
            black_box(3.0),
            black_box(4.0),
        );
        assert_eq!(get_m128(r, 0), 4.0);
        assert_eq!(get_m128(r, 1), 3.0);
        assert_eq!(get_m128(r, 2), 2.0);
        assert_eq!(get_m128(r, 3), 1.0);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_setr_ps() {
        let r = _mm_setr_ps(
            black_box(1.0),
            black_box(2.0),
            black_box(3.0),
            black_box(4.0),
        );
        assert_eq_m128(r, _mm_setr_ps(1.0, 2.0, 3.0, 4.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_setzero_ps() {
        let r = *black_box(&_mm_setzero_ps());
        assert_eq_m128(r, _mm_set1_ps(0.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_MM_SHUFFLE() {
        assert_eq!(_MM_SHUFFLE(0, 1, 1, 3), 0b00_01_01_11);
        assert_eq!(_MM_SHUFFLE(3, 1, 1, 0), 0b11_01_01_00);
        assert_eq!(_MM_SHUFFLE(1, 2, 2, 1), 0b01_10_10_01);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_shuffle_ps() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);
        let r = _mm_shuffle_ps::<0b00_01_01_11>(a, b);
        assert_eq_m128(r, _mm_setr_ps(4.0, 2.0, 6.0, 5.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_unpackhi_ps() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);
        let r = _mm_unpackhi_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(3.0, 7.0, 4.0, 8.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_unpacklo_ps() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);
        let r = _mm_unpacklo_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(1.0, 5.0, 2.0, 6.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_movehl_ps() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);
        let r = _mm_movehl_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(7.0, 8.0, 3.0, 4.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_movelh_ps() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);
        let r = _mm_movelh_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(1.0, 2.0, 5.0, 6.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_load_ss() {
        let a = 42.0f32;
        let r = _mm_load_ss(ptr::addr_of!(a));
        assert_eq_m128(r, _mm_setr_ps(42.0, 0.0, 0.0, 0.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_load1_ps() {
        let a = 42.0f32;
        let r = _mm_load1_ps(ptr::addr_of!(a));
        assert_eq_m128(r, _mm_setr_ps(42.0, 42.0, 42.0, 42.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_load_ps() {
        let vals = &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut p = vals.as_ptr();
        let mut fixup = 0.0f32;

        // Make sure p is aligned, otherwise we might get a
        // (signal: 11, SIGSEGV: invalid memory reference)

        let unalignment = (p as usize) & 0xf;
        if unalignment != 0 {
            let delta = (16 - unalignment) >> 2;
            fixup = delta as f32;
            p = p.add(delta);
        }

        let r = _mm_load_ps(p);
        let e = _mm_add_ps(_mm_setr_ps(1.0, 2.0, 3.0, 4.0), _mm_set1_ps(fixup));
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_loadu_ps() {
        let vals = &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let p = vals.as_ptr().add(3);
        let r = _mm_loadu_ps(black_box(p));
        assert_eq_m128(r, _mm_setr_ps(4.0, 5.0, 6.0, 7.0));
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_loadr_ps() {
        let vals = &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut p = vals.as_ptr();
        let mut fixup = 0.0f32;

        // Make sure p is aligned, otherwise we might get a
        // (signal: 11, SIGSEGV: invalid memory reference)

        let unalignment = (p as usize) & 0xf;
        if unalignment != 0 {
            let delta = (16 - unalignment) >> 2;
            fixup = delta as f32;
            p = p.add(delta);
        }

        let r = _mm_loadr_ps(p);
        let e = _mm_add_ps(_mm_setr_ps(4.0, 3.0, 2.0, 1.0), _mm_set1_ps(fixup));
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_store_ss() {
        let mut vals = [0.0f32; 8];
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        _mm_store_ss(vals.as_mut_ptr().add(1), a);

        assert_eq!(vals[0], 0.0);
        assert_eq!(vals[1], 1.0);
        assert_eq!(vals[2], 0.0);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_store1_ps() {
        let mut vals = [0.0f32; 8];
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);

        let mut ofs = 0;
        let mut p = vals.as_mut_ptr();

        if (p as usize) & 0xf != 0 {
            ofs = (16 - ((p as usize) & 0xf)) >> 2;
            p = p.add(ofs);
        }

        _mm_store1_ps(p, *black_box(&a));

        if ofs > 0 {
            assert_eq!(vals[ofs - 1], 0.0);
        }
        assert_eq!(vals[ofs + 0], 1.0);
        assert_eq!(vals[ofs + 1], 1.0);
        assert_eq!(vals[ofs + 2], 1.0);
        assert_eq!(vals[ofs + 3], 1.0);
        assert_eq!(vals[ofs + 4], 0.0);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_store_ps() {
        let mut vals = [0.0f32; 8];
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);

        let mut ofs = 0;
        let mut p = vals.as_mut_ptr();

        // Align p to 16-byte boundary
        if (p as usize) & 0xf != 0 {
            ofs = (16 - ((p as usize) & 0xf)) >> 2;
            p = p.add(ofs);
        }

        _mm_store_ps(p, *black_box(&a));

        if ofs > 0 {
            assert_eq!(vals[ofs - 1], 0.0);
        }
        assert_eq!(vals[ofs + 0], 1.0);
        assert_eq!(vals[ofs + 1], 2.0);
        assert_eq!(vals[ofs + 2], 3.0);
        assert_eq!(vals[ofs + 3], 4.0);
        assert_eq!(vals[ofs + 4], 0.0);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_storer_ps() {
        let mut vals = [0.0f32; 8];
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);

        let mut ofs = 0;
        let mut p = vals.as_mut_ptr();

        // Align p to 16-byte boundary
        if (p as usize) & 0xf != 0 {
            ofs = (16 - ((p as usize) & 0xf)) >> 2;
            p = p.add(ofs);
        }

        _mm_storer_ps(p, *black_box(&a));

        if ofs > 0 {
            assert_eq!(vals[ofs - 1], 0.0);
        }
        assert_eq!(vals[ofs + 0], 4.0);
        assert_eq!(vals[ofs + 1], 3.0);
        assert_eq!(vals[ofs + 2], 2.0);
        assert_eq!(vals[ofs + 3], 1.0);
        assert_eq!(vals[ofs + 4], 0.0);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_storeu_ps() {
        let mut vals = [0.0f32; 8];
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);

        let mut ofs = 0;
        let mut p = vals.as_mut_ptr();

        // Make sure p is **not** aligned to 16-byte boundary
        if (p as usize) & 0xf == 0 {
            ofs = 1;
            p = p.add(1);
        }

        _mm_storeu_ps(p, *black_box(&a));

        if ofs > 0 {
            assert_eq!(vals[ofs - 1], 0.0);
        }
        assert_eq!(vals[ofs + 0], 1.0);
        assert_eq!(vals[ofs + 1], 2.0);
        assert_eq!(vals[ofs + 2], 3.0);
        assert_eq!(vals[ofs + 3], 4.0);
        assert_eq!(vals[ofs + 4], 0.0);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_move_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);

        let r = _mm_move_ss(a, b);
        let e = _mm_setr_ps(5.0, 2.0, 3.0, 4.0);
        assert_eq_m128(e, r);
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_mm_movemask_ps() {
        let r = _mm_movemask_ps(_mm_setr_ps(-1.0, 5.0, -5.0, 0.0));
        assert_eq!(r, 0b0101);

        let r = _mm_movemask_ps(_mm_setr_ps(-1.0, -5.0, -5.0, 0.0));
        assert_eq!(r, 0b0111);
    }

    #[simd_test(enable = "sse")]
    // Miri cannot support this until it is clear how it fits in the Rust memory model
    #[cfg_attr(miri, ignore)]
    unsafe fn test_mm_sfence() {
        _mm_sfence();
    }

    #[simd_test(enable = "sse")]
    unsafe fn test_MM_TRANSPOSE4_PS() {
        let mut a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let mut b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);
        let mut c = _mm_setr_ps(9.0, 10.0, 11.0, 12.0);
        let mut d = _mm_setr_ps(13.0, 14.0, 15.0, 16.0);

        _MM_TRANSPOSE4_PS(&mut a, &mut b, &mut c, &mut d);

        assert_eq_m128(a, _mm_setr_ps(1.0, 5.0, 9.0, 13.0));
        assert_eq_m128(b, _mm_setr_ps(2.0, 6.0, 10.0, 14.0));
        assert_eq_m128(c, _mm_setr_ps(3.0, 7.0, 11.0, 15.0));
        assert_eq_m128(d, _mm_setr_ps(4.0, 8.0, 12.0, 16.0));
    }

    #[repr(align(16))]
    struct Memory {
        pub data: [f32; 4],
    }

    #[simd_test(enable = "sse")]
    // Miri cannot support this until it is clear how it fits in the Rust memory model
    // (non-temporal store)
    #[cfg_attr(miri, ignore)]
    unsafe fn test_mm_stream_ps() {
        let a = _mm_set1_ps(7.0);
        let mut mem = Memory { data: [-1.0; 4] };

        _mm_stream_ps(ptr::addr_of_mut!(mem.data[0]), a);
        for i in 0..4 {
            assert_eq!(mem.data[i], get_m128(a, i));
        }
    }
}
