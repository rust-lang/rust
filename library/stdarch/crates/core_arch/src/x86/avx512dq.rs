use crate::{
    core_arch::{simd::*, x86::*},
    intrinsics::simd::*,
    mem::transmute,
};

// And //

/// Compute the bitwise AND of packed double-precision (64-bit) floating point numbers in a and b
/// and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_and_pd&ig_expand=288)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_and_pd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let and = _mm_and_pd(a, b).as_f64x2();
    transmute(simd_select_bitmask(k, and, src.as_f64x2()))
}

/// Compute the bitwise AND of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_and_pd&ig_expand=289)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_and_pd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let and = _mm_and_pd(a, b).as_f64x2();
    let zero = _mm_setzero_pd().as_f64x2();
    transmute(simd_select_bitmask(k, and, zero))
}

/// Compute the bitwise AND of packed double-precision (64-bit) floating point numbers in a and b
/// and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_and_pd&ig_expand=291)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_and_pd(src: __m256d, k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let and = _mm256_and_pd(a, b).as_f64x4();
    transmute(simd_select_bitmask(k, and, src.as_f64x4()))
}

/// Compute the bitwise AND of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_and_pd&ig_expand=292)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_and_pd(k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let and = _mm256_and_pd(a, b).as_f64x4();
    let zero = _mm256_setzero_pd().as_f64x4();
    transmute(simd_select_bitmask(k, and, zero))
}

/// Compute the bitwise AND of packed double-precision (64-bit) floating point numbers in a and b
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_and_pd&ig_expand=293)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandps))] // FIXME: should be `vandpd` instruction.
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_and_pd(a: __m512d, b: __m512d) -> __m512d {
    transmute(simd_and(transmute::<_, u64x8>(a), transmute::<_, u64x8>(b)))
}

/// Compute the bitwise AND of packed double-precision (64-bit) floating point numbers in a and b
/// and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_and_pd&ig_expand=294)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_and_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let and = _mm512_and_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, and, src.as_f64x8()))
}

/// Compute the bitwise AND of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_and_pd&ig_expand=295)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_and_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let and = _mm512_and_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, and, zero))
}

/// Compute the bitwise AND of packed single-precision (32-bit) floating point numbers in a and b
/// and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_and_ps&ig_expand=297)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_and_ps(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let and = _mm_and_ps(a, b).as_f32x4();
    transmute(simd_select_bitmask(k, and, src.as_f32x4()))
}

/// Compute the bitwise AND of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_and_ps&ig_expand=298)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_and_ps(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let and = _mm_and_ps(a, b).as_f32x4();
    let zero = _mm_setzero_ps().as_f32x4();
    transmute(simd_select_bitmask(k, and, zero))
}

/// Compute the bitwise AND of packed single-precision (32-bit) floating point numbers in a and b
/// and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_and_ps&ig_expand=300)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_and_ps(src: __m256, k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let and = _mm256_and_ps(a, b).as_f32x8();
    transmute(simd_select_bitmask(k, and, src.as_f32x8()))
}

/// Compute the bitwise AND of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_and_ps&ig_expand=301)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_and_ps(k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let and = _mm256_and_ps(a, b).as_f32x8();
    let zero = _mm256_setzero_ps().as_f32x8();
    transmute(simd_select_bitmask(k, and, zero))
}

/// Compute the bitwise AND of packed single-precision (32-bit) floating point numbers in a and b
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_and_ps&ig_expand=303)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_and_ps(a: __m512, b: __m512) -> __m512 {
    transmute(simd_and(
        transmute::<_, u32x16>(a),
        transmute::<_, u32x16>(b),
    ))
}

/// Compute the bitwise AND of packed single-precision (32-bit) floating point numbers in a and b
/// and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_and_ps&ig_expand=304)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_and_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let and = _mm512_and_ps(a, b).as_f32x16();
    transmute(simd_select_bitmask(k, and, src.as_f32x16()))
}

/// Compute the bitwise AND of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_and_ps&ig_expand=305)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_and_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let and = _mm512_and_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, and, zero))
}

// Andnot

/// Compute the bitwise NOT of packed double-precision (64-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_andnot_pd&ig_expand=326)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_andnot_pd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let andnot = _mm_andnot_pd(a, b).as_f64x2();
    transmute(simd_select_bitmask(k, andnot, src.as_f64x2()))
}

/// Compute the bitwise NOT of packed double-precision (64-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst using zeromask k (elements are zeroed out if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_andnot_pd&ig_expand=327)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_andnot_pd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let andnot = _mm_andnot_pd(a, b).as_f64x2();
    let zero = _mm_setzero_pd().as_f64x2();
    transmute(simd_select_bitmask(k, andnot, zero))
}

/// Compute the bitwise NOT of packed double-precision (64-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_andnot_pd&ig_expand=329)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_andnot_pd(src: __m256d, k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let andnot = _mm256_andnot_pd(a, b).as_f64x4();
    transmute(simd_select_bitmask(k, andnot, src.as_f64x4()))
}

/// Compute the bitwise NOT of packed double-precision (64-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst using zeromask k (elements are zeroed out if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_andnot_pd&ig_expand=330)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_andnot_pd(k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let andnot = _mm256_andnot_pd(a, b).as_f64x4();
    let zero = _mm256_setzero_pd().as_f64x4();
    transmute(simd_select_bitmask(k, andnot, zero))
}

/// Compute the bitwise NOT of packed double-precision (64-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_andnot_pd&ig_expand=331)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnps))] // FIXME: should be `vandnpd` instruction.
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_andnot_pd(a: __m512d, b: __m512d) -> __m512d {
    _mm512_and_pd(_mm512_xor_pd(a, transmute(_mm512_set1_epi64(-1))), b)
}

/// Compute the bitwise NOT of packed double-precision (64-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_andnot_pd&ig_expand=332)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_andnot_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let andnot = _mm512_andnot_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, andnot, src.as_f64x8()))
}

/// Compute the bitwise NOT of packed double-precision (64-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst using zeromask k (elements are zeroed out if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_andnot_pd&ig_expand=333)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_andnot_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let andnot = _mm512_andnot_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, andnot, zero))
}

/// Compute the bitwise NOT of packed single-precision (32-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_andnot_ps&ig_expand=335)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_andnot_ps(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let andnot = _mm_andnot_ps(a, b).as_f32x4();
    transmute(simd_select_bitmask(k, andnot, src.as_f32x4()))
}

/// Compute the bitwise NOT of packed single-precision (32-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst using zeromask k (elements are zeroed out if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_andnot_ps&ig_expand=336)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_andnot_ps(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let andnot = _mm_andnot_ps(a, b).as_f32x4();
    let zero = _mm_setzero_ps().as_f32x4();
    transmute(simd_select_bitmask(k, andnot, zero))
}

/// Compute the bitwise NOT of packed single-precision (32-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_andnot_ps&ig_expand=338)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_andnot_ps(src: __m256, k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let andnot = _mm256_andnot_ps(a, b).as_f32x8();
    transmute(simd_select_bitmask(k, andnot, src.as_f32x8()))
}

/// Compute the bitwise NOT of packed single-precision (32-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst using zeromask k (elements are zeroed out if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_andnot_ps&ig_expand=339)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandnps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_andnot_ps(k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let andnot = _mm256_andnot_ps(a, b).as_f32x8();
    let zero = _mm256_setzero_ps().as_f32x8();
    transmute(simd_select_bitmask(k, andnot, zero))
}

/// Compute the bitwise NOT of packed single-precision (32-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_andnot_ps&ig_expand=340)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_andnot_ps(a: __m512, b: __m512) -> __m512 {
    _mm512_and_ps(_mm512_xor_ps(a, transmute(_mm512_set1_epi32(-1))), b)
}

/// Compute the bitwise NOT of packed single-precision (32-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_andnot_ps&ig_expand=341)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_andnot_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let andnot = _mm512_andnot_ps(a, b).as_f32x16();
    transmute(simd_select_bitmask(k, andnot, src.as_f32x16()))
}

/// Compute the bitwise NOT of packed single-precision (32-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst using zeromask k (elements are zeroed out if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_andnot_ps&ig_expand=342)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_andnot_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let andnot = _mm512_andnot_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, andnot, zero))
}

// Or

/// Compute the bitwise OR of packed double-precision (64-bit) floating point numbers in a and b
/// and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_or_pd&ig_expand=4824)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_or_pd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let or = _mm_or_pd(a, b).as_f64x2();
    transmute(simd_select_bitmask(k, or, src.as_f64x2()))
}

/// Compute the bitwise OR of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_or_pd&ig_expand=4825)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_or_pd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let or = _mm_or_pd(a, b).as_f64x2();
    let zero = _mm_setzero_pd().as_f64x2();
    transmute(simd_select_bitmask(k, or, zero))
}

/// Compute the bitwise OR of packed double-precision (64-bit) floating point numbers in a and b
/// and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_or_pd&ig_expand=4827)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_or_pd(src: __m256d, k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let or = _mm256_or_pd(a, b).as_f64x4();
    transmute(simd_select_bitmask(k, or, src.as_f64x4()))
}

/// Compute the bitwise OR of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_or_pd&ig_expand=4828)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_or_pd(k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let or = _mm256_or_pd(a, b).as_f64x4();
    let zero = _mm256_setzero_pd().as_f64x4();
    transmute(simd_select_bitmask(k, or, zero))
}

/// Compute the bitwise OR of packed double-precision (64-bit) floating point numbers in a and b
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_or_pd&ig_expand=4829)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorps))] // FIXME: should be `vorpd` instruction.
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_or_pd(a: __m512d, b: __m512d) -> __m512d {
    transmute(simd_or(transmute::<_, u64x8>(a), transmute::<_, u64x8>(b)))
}

/// Compute the bitwise OR of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_or_pd&ig_expand=4830)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_or_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let or = _mm512_or_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, or, src.as_f64x8()))
}

/// Compute the bitwise OR of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_or_pd&ig_expand=4831)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_or_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let or = _mm512_or_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, or, zero))
}

/// Compute the bitwise OR of packed single-precision (32-bit) floating point numbers in a and b
/// and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_or_ps&ig_expand=4833)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_or_ps(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let or = _mm_or_ps(a, b).as_f32x4();
    transmute(simd_select_bitmask(k, or, src.as_f32x4()))
}

/// Compute the bitwise OR of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_or_ps&ig_expand=4834)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_or_ps(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let or = _mm_or_ps(a, b).as_f32x4();
    let zero = _mm_setzero_ps().as_f32x4();
    transmute(simd_select_bitmask(k, or, zero))
}

/// Compute the bitwise OR of packed single-precision (32-bit) floating point numbers in a and b
/// and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_or_ps&ig_expand=4836)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_or_ps(src: __m256, k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let or = _mm256_or_ps(a, b).as_f32x8();
    transmute(simd_select_bitmask(k, or, src.as_f32x8()))
}

/// Compute the bitwise OR of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_or_ps&ig_expand=4837)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_or_ps(k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let or = _mm256_or_ps(a, b).as_f32x8();
    let zero = _mm256_setzero_ps().as_f32x8();
    transmute(simd_select_bitmask(k, or, zero))
}

/// Compute the bitwise OR of packed single-precision (32-bit) floating point numbers in a and b
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_or_ps&ig_expand=4838)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_or_ps(a: __m512, b: __m512) -> __m512 {
    transmute(simd_or(
        transmute::<_, u32x16>(a),
        transmute::<_, u32x16>(b),
    ))
}

/// Compute the bitwise OR of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_or_ps&ig_expand=4839)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_or_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let or = _mm512_or_ps(a, b).as_f32x16();
    transmute(simd_select_bitmask(k, or, src.as_f32x16()))
}

/// Compute the bitwise OR of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_or_ps&ig_expand=4840)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_or_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let or = _mm512_or_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, or, zero))
}

// Xor

/// Compute the bitwise XOR of packed double-precision (64-bit) floating point numbers in a and b
/// and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_xor_pd&ig_expand=7094)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_xor_pd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let xor = _mm_xor_pd(a, b).as_f64x2();
    transmute(simd_select_bitmask(k, xor, src.as_f64x2()))
}

/// Compute the bitwise XOR of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_xor_pd&ig_expand=7095)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_xor_pd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let xor = _mm_xor_pd(a, b).as_f64x2();
    let zero = _mm_setzero_pd().as_f64x2();
    transmute(simd_select_bitmask(k, xor, zero))
}

/// Compute the bitwise XOR of packed double-precision (64-bit) floating point numbers in a and b
/// and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_xor_pd&ig_expand=7097)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_xor_pd(src: __m256d, k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let xor = _mm256_xor_pd(a, b).as_f64x4();
    transmute(simd_select_bitmask(k, xor, src.as_f64x4()))
}

/// Compute the bitwise XOR of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_xor_pd&ig_expand=7098)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_xor_pd(k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    let xor = _mm256_xor_pd(a, b).as_f64x4();
    let zero = _mm256_setzero_pd().as_f64x4();
    transmute(simd_select_bitmask(k, xor, zero))
}

/// Compute the bitwise XOR of packed double-precision (64-bit) floating point numbers in a and b
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_xor_pd&ig_expand=7102)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorps))] // FIXME: should be `vxorpd` instruction.
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_xor_pd(a: __m512d, b: __m512d) -> __m512d {
    transmute(simd_xor(transmute::<_, u64x8>(a), transmute::<_, u64x8>(b)))
}

/// Compute the bitwise XOR of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_xor_pd&ig_expand=7100)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_xor_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let xor = _mm512_xor_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, xor, src.as_f64x8()))
}

/// Compute the bitwise XOR of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_xor_pd&ig_expand=7101)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_xor_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let xor = _mm512_xor_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, xor, zero))
}

/// Compute the bitwise XOR of packed single-precision (32-bit) floating point numbers in a and b
/// and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_xor_ps&ig_expand=7103)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_xor_ps(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let xor = _mm_xor_ps(a, b).as_f32x4();
    transmute(simd_select_bitmask(k, xor, src.as_f32x4()))
}

/// Compute the bitwise XOR of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_xor_ps&ig_expand=7104)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_xor_ps(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let xor = _mm_xor_ps(a, b).as_f32x4();
    let zero = _mm_setzero_ps().as_f32x4();
    transmute(simd_select_bitmask(k, xor, zero))
}

/// Compute the bitwise XOR of packed single-precision (32-bit) floating point numbers in a and b
/// and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_xor_ps&ig_expand=7106)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_xor_ps(src: __m256, k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let xor = _mm256_xor_ps(a, b).as_f32x8();
    transmute(simd_select_bitmask(k, xor, src.as_f32x8()))
}

/// Compute the bitwise XOR of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_xor_ps&ig_expand=7107)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_xor_ps(k: __mmask8, a: __m256, b: __m256) -> __m256 {
    let xor = _mm256_xor_ps(a, b).as_f32x8();
    let zero = _mm256_setzero_ps().as_f32x8();
    transmute(simd_select_bitmask(k, xor, zero))
}

/// Compute the bitwise XOR of packed single-precision (32-bit) floating point numbers in a and b
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_xor_ps&ig_expand=7111)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_xor_ps(a: __m512, b: __m512) -> __m512 {
    transmute(simd_xor(
        transmute::<_, u32x16>(a),
        transmute::<_, u32x16>(b),
    ))
}

/// Compute the bitwise XOR of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_xor_ps&ig_expand=7109)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_xor_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let xor = _mm512_xor_ps(a, b).as_f32x16();
    transmute(simd_select_bitmask(k, xor, src.as_f32x16()))
}

/// Compute the bitwise XOR of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_xor_ps&ig_expand=7110)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_xor_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let xor = _mm512_xor_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, xor, zero))
}

// Broadcast

/// Broadcasts the lower 2 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_f32x2&ig_expand=509)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_broadcast_f32x2(a: __m128) -> __m256 {
    let b: f32x8 = simd_shuffle!(a, a, [0, 1, 0, 1, 0, 1, 0, 1]);
    transmute(b)
}

/// Broadcasts the lower 2 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst using writemask k (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_broadcast_f32x2&ig_expand=510)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vbroadcastf32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_broadcast_f32x2(src: __m256, k: __mmask8, a: __m128) -> __m256 {
    let b = _mm256_broadcast_f32x2(a).as_f32x8();
    transmute(simd_select_bitmask(k, b, src.as_f32x8()))
}

/// Broadcasts the lower 2 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_broadcast_f32x2&ig_expand=511)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vbroadcastf32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_broadcast_f32x2(k: __mmask8, a: __m128) -> __m256 {
    let b = _mm256_broadcast_f32x2(a).as_f32x8();
    let zero = _mm256_setzero_ps().as_f32x8();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Broadcasts the lower 2 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcast_f32x2&ig_expand=512)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_broadcast_f32x2(a: __m128) -> __m512 {
    let b: f32x16 = simd_shuffle!(a, a, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
    transmute(b)
}

/// Broadcasts the lower 2 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst using writemask k (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_broadcast_f32x2&ig_expand=513)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vbroadcastf32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_broadcast_f32x2(src: __m512, k: __mmask16, a: __m128) -> __m512 {
    let b = _mm512_broadcast_f32x2(a).as_f32x16();
    transmute(simd_select_bitmask(k, b, src.as_f32x16()))
}

/// Broadcasts the lower 2 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_broadcast_f32x2&ig_expand=514)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vbroadcastf32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_broadcast_f32x2(k: __mmask16, a: __m128) -> __m512 {
    let b = _mm512_broadcast_f32x2(a).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Broadcasts the 8 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcast_f32x8&ig_expand=521)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_broadcast_f32x8(a: __m256) -> __m512 {
    let b: f32x16 = simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
    transmute(b)
}

/// Broadcasts the 8 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst using writemask k (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_broadcast_f32x8&ig_expand=522)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_broadcast_f32x8(src: __m512, k: __mmask16, a: __m256) -> __m512 {
    let b = _mm512_broadcast_f32x8(a).as_f32x16();
    transmute(simd_select_bitmask(k, b, src.as_f32x16()))
}

/// Broadcasts the 8 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_broadcast_f32x8&ig_expand=523)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_broadcast_f32x8(k: __mmask16, a: __m256) -> __m512 {
    let b = _mm512_broadcast_f32x8(a).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Broadcasts the 2 packed double-precision (64-bit) floating-point elements from a to all
/// elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_f64x2&ig_expand=524)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_broadcast_f64x2(a: __m128d) -> __m256d {
    let b: f64x4 = simd_shuffle!(a, a, [0, 1, 0, 1]);
    transmute(b)
}

/// Broadcasts the 2 packed double-precision (64-bit) floating-point elements from a to all
/// elements of dst using writemask k (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_broadcast_f64x2&ig_expand=525)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_broadcast_f64x2(src: __m256d, k: __mmask8, a: __m128d) -> __m256d {
    let b = _mm256_broadcast_f64x2(a).as_f64x4();
    transmute(simd_select_bitmask(k, b, src.as_f64x4()))
}

/// Broadcasts the 2 packed double-precision (64-bit) floating-point elements from a to all
/// elements of dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_broadcast_f64x2&ig_expand=526)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_broadcast_f64x2(k: __mmask8, a: __m128d) -> __m256d {
    let b = _mm256_broadcast_f64x2(a).as_f64x4();
    let zero = _mm256_setzero_pd().as_f64x4();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Broadcasts the 2 packed double-precision (64-bit) floating-point elements from a to all
/// elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcast_f64x2&ig_expand=527)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_broadcast_f64x2(a: __m128d) -> __m512d {
    let b: f64x8 = simd_shuffle!(a, a, [0, 1, 0, 1, 0, 1, 0, 1]);
    transmute(b)
}

/// Broadcasts the 2 packed double-precision (64-bit) floating-point elements from a to all
/// elements of dst using writemask k (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_broadcast_f64x2&ig_expand=528)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_broadcast_f64x2(src: __m512d, k: __mmask8, a: __m128d) -> __m512d {
    let b = _mm512_broadcast_f64x2(a).as_f64x8();
    transmute(simd_select_bitmask(k, b, src.as_f64x8()))
}

/// Broadcasts the 2 packed double-precision (64-bit) floating-point elements from a to all
/// elements of dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_broadcast_f64x2&ig_expand=529)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_broadcast_f64x2(k: __mmask8, a: __m128d) -> __m512d {
    let b = _mm512_broadcast_f64x2(a).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_broadcast_i32x2&ig_expand=533)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_broadcast_i32x2(a: __m128i) -> __m128i {
    let a = a.as_i32x4();
    let b: i32x4 = simd_shuffle!(a, a, [0, 1, 0, 1]);
    transmute(b)
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_broadcast_i32x2&ig_expand=534)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vbroadcasti32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_broadcast_i32x2(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    let b = _mm_broadcast_i32x2(a).as_i32x4();
    transmute(simd_select_bitmask(k, b, src.as_i32x4()))
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_broadcast_i32x2&ig_expand=535)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vbroadcasti32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_broadcast_i32x2(k: __mmask8, a: __m128i) -> __m128i {
    let b = _mm_broadcast_i32x2(a).as_i32x4();
    let zero = _mm_setzero_si128().as_i32x4();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_i32x2&ig_expand=536)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_broadcast_i32x2(a: __m128i) -> __m256i {
    let a = a.as_i32x4();
    let b: i32x8 = simd_shuffle!(a, a, [0, 1, 0, 1, 0, 1, 0, 1]);
    transmute(b)
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_broadcast_i32x2&ig_expand=537)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vbroadcasti32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_broadcast_i32x2(src: __m256i, k: __mmask8, a: __m128i) -> __m256i {
    let b = _mm256_broadcast_i32x2(a).as_i32x8();
    transmute(simd_select_bitmask(k, b, src.as_i32x8()))
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_broadcast_i32x2&ig_expand=538)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vbroadcasti32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_broadcast_i32x2(k: __mmask8, a: __m128i) -> __m256i {
    let b = _mm256_broadcast_i32x2(a).as_i32x8();
    let zero = _mm256_setzero_si256().as_i32x8();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcast_i32x2&ig_expand=539)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_broadcast_i32x2(a: __m128i) -> __m512i {
    let a = a.as_i32x4();
    let b: i32x16 = simd_shuffle!(a, a, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
    transmute(b)
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_broadcast_i32x2&ig_expand=540)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vbroadcasti32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_broadcast_i32x2(src: __m512i, k: __mmask16, a: __m128i) -> __m512i {
    let b = _mm512_broadcast_i32x2(a).as_i32x16();
    transmute(simd_select_bitmask(k, b, src.as_i32x16()))
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_broadcast_i32x2&ig_expand=541)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vbroadcasti32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_broadcast_i32x2(k: __mmask16, a: __m128i) -> __m512i {
    let b = _mm512_broadcast_i32x2(a).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Broadcasts the 8 packed 32-bit integers from a to all elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcast_i32x8&ig_expand=548)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_broadcast_i32x8(a: __m256i) -> __m512i {
    let a = a.as_i32x8();
    let b: i32x16 = simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
    transmute(b)
}

/// Broadcasts the 8 packed 32-bit integers from a to all elements of dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_broadcast_i32x8&ig_expand=549)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_broadcast_i32x8(src: __m512i, k: __mmask16, a: __m256i) -> __m512i {
    let b = _mm512_broadcast_i32x8(a).as_i32x16();
    transmute(simd_select_bitmask(k, b, src.as_i32x16()))
}

/// Broadcasts the 8 packed 32-bit integers from a to all elements of dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_broadcast_i32x8&ig_expand=550)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_broadcast_i32x8(k: __mmask16, a: __m256i) -> __m512i {
    let b = _mm512_broadcast_i32x8(a).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Broadcasts the 2 packed 64-bit integers from a to all elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_i64x2&ig_expand=551)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_broadcast_i64x2(a: __m128i) -> __m256i {
    let a = a.as_i64x2();
    let b: i64x4 = simd_shuffle!(a, a, [0, 1, 0, 1]);
    transmute(b)
}

/// Broadcasts the 2 packed 64-bit integers from a to all elements of dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_broadcast_i64x2&ig_expand=552)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_broadcast_i64x2(src: __m256i, k: __mmask8, a: __m128i) -> __m256i {
    let b = _mm256_broadcast_i64x2(a).as_i64x4();
    transmute(simd_select_bitmask(k, b, src.as_i64x4()))
}

/// Broadcasts the 2 packed 64-bit integers from a to all elements of dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_broadcast_i64x2&ig_expand=553)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_broadcast_i64x2(k: __mmask8, a: __m128i) -> __m256i {
    let b = _mm256_broadcast_i64x2(a).as_i64x4();
    let zero = _mm256_setzero_si256().as_i64x4();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Broadcasts the 2 packed 64-bit integers from a to all elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcast_i64x2&ig_expand=554)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_broadcast_i64x2(a: __m128i) -> __m512i {
    let a = a.as_i64x2();
    let b: i64x8 = simd_shuffle!(a, a, [0, 1, 0, 1, 0, 1, 0, 1]);
    transmute(b)
}

/// Broadcasts the 2 packed 64-bit integers from a to all elements of dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_broadcast_i64x2&ig_expand=555)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_broadcast_i64x2(src: __m512i, k: __mmask8, a: __m128i) -> __m512i {
    let b = _mm512_broadcast_i64x2(a).as_i64x8();
    transmute(simd_select_bitmask(k, b, src.as_i64x8()))
}

/// Broadcasts the 2 packed 64-bit integers from a to all elements of dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_broadcast_i64x2&ig_expand=556)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_broadcast_i64x2(k: __mmask8, a: __m128i) -> __m512i {
    let b = _mm512_broadcast_i64x2(a).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, b, zero))
}

// Extract

/// Extracts 256 bits (composed of 8 packed single-precision (32-bit) floating-point elements) from a,
/// selected with IMM8, and stores the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_extractf32x8_ps&ig_expand=2946)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_extractf32x8_ps<const IMM8: i32>(a: __m512) -> __m256 {
    static_assert_uimm_bits!(IMM8, 1);
    match IMM8 & 1 {
        0 => simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7]),
        _ => simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]),
    }
}

/// Extracts 256 bits (composed of 8 packed single-precision (32-bit) floating-point elements) from a,
/// selected with IMM8, and stores the result in dst using writemask k (elements are copied from src
/// if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_extractf32x8_ps&ig_expand=2947)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vextractf32x8, IMM8 = 1))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_extractf32x8_ps<const IMM8: i32>(
    src: __m256,
    k: __mmask8,
    a: __m512,
) -> __m256 {
    static_assert_uimm_bits!(IMM8, 1);
    let b = _mm512_extractf32x8_ps::<IMM8>(a);
    transmute(simd_select_bitmask(k, b.as_f32x8(), src.as_f32x8()))
}

/// Extracts 256 bits (composed of 8 packed single-precision (32-bit) floating-point elements) from a,
/// selected with IMM8, and stores the result in dst using zeromask k (elements are zeroed out if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_extractf32x8_ps&ig_expand=2948)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vextractf32x8, IMM8 = 1))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_extractf32x8_ps<const IMM8: i32>(k: __mmask8, a: __m512) -> __m256 {
    static_assert_uimm_bits!(IMM8, 1);
    let b = _mm512_extractf32x8_ps::<IMM8>(a);
    let zero = _mm256_setzero_ps().as_f32x8();
    transmute(simd_select_bitmask(k, b.as_f32x8(), zero))
}

/// Extracts 128 bits (composed of 2 packed double-precision (64-bit) floating-point elements) from a,
/// selected with IMM8, and stores the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_extractf64x2_pd&ig_expand=2949)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_extractf64x2_pd<const IMM8: i32>(a: __m256d) -> __m128d {
    static_assert_uimm_bits!(IMM8, 1);
    match IMM8 & 1 {
        0 => simd_shuffle!(a, a, [0, 1]),
        _ => simd_shuffle!(a, a, [2, 3]),
    }
}

/// Extracts 128 bits (composed of 2 packed double-precision (64-bit) floating-point elements) from a,
/// selected with IMM8, and stores the result in dst using writemask k (elements are copied from src
/// if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_extractf64x2_pd&ig_expand=2950)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vextractf64x2, IMM8 = 1))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_extractf64x2_pd<const IMM8: i32>(
    src: __m128d,
    k: __mmask8,
    a: __m256d,
) -> __m128d {
    static_assert_uimm_bits!(IMM8, 1);
    let b = _mm256_extractf64x2_pd::<IMM8>(a);
    transmute(simd_select_bitmask(k, b.as_f64x2(), src.as_f64x2()))
}

/// Extracts 128 bits (composed of 2 packed double-precision (64-bit) floating-point elements) from a,
/// selected with IMM8, and stores the result in dst using zeromask k (elements are zeroed out if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_extractf64x2_pd&ig_expand=2951)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vextractf64x2, IMM8 = 1))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_extractf64x2_pd<const IMM8: i32>(k: __mmask8, a: __m256d) -> __m128d {
    static_assert_uimm_bits!(IMM8, 1);
    let b = _mm256_extractf64x2_pd::<IMM8>(a);
    let zero = _mm_setzero_pd().as_f64x2();
    transmute(simd_select_bitmask(k, b.as_f64x2(), zero))
}

/// Extracts 128 bits (composed of 2 packed double-precision (64-bit) floating-point elements) from a,
/// selected with IMM8, and stores the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_extractf64x2_pd&ig_expand=2952)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_extractf64x2_pd<const IMM8: i32>(a: __m512d) -> __m128d {
    static_assert_uimm_bits!(IMM8, 2);
    match IMM8 & 3 {
        0 => simd_shuffle!(a, a, [0, 1]),
        1 => simd_shuffle!(a, a, [2, 3]),
        2 => simd_shuffle!(a, a, [4, 5]),
        _ => simd_shuffle!(a, a, [6, 7]),
    }
}

/// Extracts 128 bits (composed of 2 packed double-precision (64-bit) floating-point elements) from a,
/// selected with IMM8, and stores the result in dst using writemask k (elements are copied from src
/// if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_extractf64x2_pd&ig_expand=2953)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vextractf64x2, IMM8 = 3))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_extractf64x2_pd<const IMM8: i32>(
    src: __m128d,
    k: __mmask8,
    a: __m512d,
) -> __m128d {
    static_assert_uimm_bits!(IMM8, 2);
    let b = _mm512_extractf64x2_pd::<IMM8>(a).as_f64x2();
    transmute(simd_select_bitmask(k, b, src.as_f64x2()))
}

/// Extracts 128 bits (composed of 2 packed double-precision (64-bit) floating-point elements) from a,
/// selected with IMM8, and stores the result in dst using zeromask k (elements are zeroed out if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_extractf64x2_pd&ig_expand=2954)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vextractf64x2, IMM8 = 3))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_extractf64x2_pd<const IMM8: i32>(k: __mmask8, a: __m512d) -> __m128d {
    static_assert_uimm_bits!(IMM8, 2);
    let b = _mm512_extractf64x2_pd::<IMM8>(a).as_f64x2();
    let zero = _mm_setzero_pd().as_f64x2();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Extracts 256 bits (composed of 8 packed 32-bit integers) from a, selected with IMM8, and stores
/// the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_extracti32x8_epi32&ig_expand=2965)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_extracti32x8_epi32<const IMM8: i32>(a: __m512i) -> __m256i {
    static_assert_uimm_bits!(IMM8, 1);
    let a = a.as_i32x16();
    let b: i32x8 = match IMM8 & 1 {
        0 => simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7]),
        _ => simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]),
    };
    transmute(b)
}

/// Extracts 256 bits (composed of 8 packed 32-bit integers) from a, selected with IMM8, and stores
/// the result in dst using writemask k (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_extracti32x8_epi32&ig_expand=2966)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vextracti32x8, IMM8 = 1))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_extracti32x8_epi32<const IMM8: i32>(
    src: __m256i,
    k: __mmask8,
    a: __m512i,
) -> __m256i {
    static_assert_uimm_bits!(IMM8, 1);
    let b = _mm512_extracti32x8_epi32::<IMM8>(a).as_i32x8();
    transmute(simd_select_bitmask(k, b, src.as_i32x8()))
}

/// Extracts 256 bits (composed of 8 packed 32-bit integers) from a, selected with IMM8, and stores
/// the result in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_extracti32x8_epi32&ig_expand=2967)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vextracti32x8, IMM8 = 1))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_extracti32x8_epi32<const IMM8: i32>(k: __mmask8, a: __m512i) -> __m256i {
    static_assert_uimm_bits!(IMM8, 1);
    let b = _mm512_extracti32x8_epi32::<IMM8>(a).as_i32x8();
    let zero = _mm256_setzero_si256().as_i32x8();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Extracts 128 bits (composed of 2 packed 64-bit integers) from a, selected with IMM8, and stores
/// the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_extracti64x2_epi64&ig_expand=2968)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_extracti64x2_epi64<const IMM8: i32>(a: __m256i) -> __m128i {
    static_assert_uimm_bits!(IMM8, 1);
    match IMM8 & 1 {
        0 => simd_shuffle!(a, a, [0, 1]),
        _ => simd_shuffle!(a, a, [2, 3]),
    }
}

/// Extracts 128 bits (composed of 2 packed 64-bit integers) from a, selected with IMM8, and stores
/// the result in dst using writemask k (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_extracti64x2_epi64&ig_expand=2969)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vextracti64x2, IMM8 = 1))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_extracti64x2_epi64<const IMM8: i32>(
    src: __m128i,
    k: __mmask8,
    a: __m256i,
) -> __m128i {
    static_assert_uimm_bits!(IMM8, 1);
    let b = _mm256_extracti64x2_epi64::<IMM8>(a).as_i64x2();
    transmute(simd_select_bitmask(k, b, src.as_i64x2()))
}

/// Extracts 128 bits (composed of 2 packed 64-bit integers) from a, selected with IMM8, and stores
/// the result in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_extracti64x2_epi64&ig_expand=2970)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vextracti64x2, IMM8 = 1))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_extracti64x2_epi64<const IMM8: i32>(k: __mmask8, a: __m256i) -> __m128i {
    static_assert_uimm_bits!(IMM8, 1);
    let b = _mm256_extracti64x2_epi64::<IMM8>(a).as_i64x2();
    let zero = _mm_setzero_si128().as_i64x2();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Extracts 128 bits (composed of 2 packed 64-bit integers) from a, selected with IMM8, and stores
/// the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_extracti64x2_epi64&ig_expand=2971)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_extracti64x2_epi64<const IMM8: i32>(a: __m512i) -> __m128i {
    static_assert_uimm_bits!(IMM8, 2);
    match IMM8 & 3 {
        0 => simd_shuffle!(a, a, [0, 1]),
        1 => simd_shuffle!(a, a, [2, 3]),
        2 => simd_shuffle!(a, a, [4, 5]),
        _ => simd_shuffle!(a, a, [6, 7]),
    }
}

/// Extracts 128 bits (composed of 2 packed 64-bit integers) from a, selected with IMM8, and stores
/// the result in dst using writemask k (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_extracti64x2_epi64&ig_expand=2972)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vextracti64x2, IMM8 = 3))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_extracti64x2_epi64<const IMM8: i32>(
    src: __m128i,
    k: __mmask8,
    a: __m512i,
) -> __m128i {
    static_assert_uimm_bits!(IMM8, 2);
    let b = _mm512_extracti64x2_epi64::<IMM8>(a).as_i64x2();
    transmute(simd_select_bitmask(k, b, src.as_i64x2()))
}

/// Extracts 128 bits (composed of 2 packed 64-bit integers) from a, selected with IMM8, and stores
/// the result in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_extracti64x2_epi64&ig_expand=2973)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vextracti64x2, IMM8 = 3))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_extracti64x2_epi64<const IMM8: i32>(k: __mmask8, a: __m512i) -> __m128i {
    static_assert_uimm_bits!(IMM8, 2);
    let b = _mm512_extracti64x2_epi64::<IMM8>(a).as_i64x2();
    let zero = _mm_setzero_si128().as_i64x2();
    transmute(simd_select_bitmask(k, b, zero))
}

// Insert

/// Copy a to dst, then insert 256 bits (composed of 8 packed single-precision (32-bit) floating-point
/// elements) from b into dst at the location specified by IMM8.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_insertf32x8&ig_expand=3850)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_insertf32x8<const IMM8: i32>(a: __m512, b: __m256) -> __m512 {
    static_assert_uimm_bits!(IMM8, 1);
    let b = _mm512_castps256_ps512(b);
    match IMM8 & 1 {
        0 => simd_shuffle!(
            a,
            b,
            [16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15]
        ),
        _ => simd_shuffle!(
            a,
            b,
            [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]
        ),
    }
}

/// Copy a to tmp, then insert 256 bits (composed of 8 packed single-precision (32-bit) floating-point
/// elements) from b into tmp at the location specified by IMM8, and copy tmp to dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_insertf32x8&ig_expand=3851)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vinsertf32x8, IMM8 = 1))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_insertf32x8<const IMM8: i32>(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m256,
) -> __m512 {
    static_assert_uimm_bits!(IMM8, 1);
    let c = _mm512_insertf32x8::<IMM8>(a, b);
    transmute(simd_select_bitmask(k, c.as_f32x16(), src.as_f32x16()))
}

/// Copy a to tmp, then insert 256 bits (composed of 8 packed single-precision (32-bit) floating-point
/// elements) from b into tmp at the location specified by IMM8, and copy tmp to dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_insertf32x8&ig_expand=3852)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vinsertf32x8, IMM8 = 1))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_insertf32x8<const IMM8: i32>(
    k: __mmask16,
    a: __m512,
    b: __m256,
) -> __m512 {
    static_assert_uimm_bits!(IMM8, 1);
    let c = _mm512_insertf32x8::<IMM8>(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, c, zero))
}

/// Copy a to dst, then insert 128 bits (composed of 2 packed double-precision (64-bit) floating-point
/// elements) from b into dst at the location specified by IMM8.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_insertf64x2&ig_expand=3853)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_insertf64x2<const IMM8: i32>(a: __m256d, b: __m128d) -> __m256d {
    static_assert_uimm_bits!(IMM8, 1);
    let b = _mm256_castpd128_pd256(b);
    match IMM8 & 1 {
        0 => simd_shuffle!(a, b, [4, 5, 2, 3]),
        _ => simd_shuffle!(a, b, [0, 1, 4, 5]),
    }
}

/// Copy a to tmp, then insert 128 bits (composed of 2 packed double-precision (64-bit) floating-point
/// elements) from b into tmp at the location specified by IMM8, and copy tmp to dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_insertf64x2&ig_expand=3854)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vinsertf64x2, IMM8 = 1))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_insertf64x2<const IMM8: i32>(
    src: __m256d,
    k: __mmask8,
    a: __m256d,
    b: __m128d,
) -> __m256d {
    static_assert_uimm_bits!(IMM8, 1);
    let c = _mm256_insertf64x2::<IMM8>(a, b);
    transmute(simd_select_bitmask(k, c.as_f64x4(), src.as_f64x4()))
}

/// Copy a to tmp, then insert 128 bits (composed of 2 packed double-precision (64-bit) floating-point
/// elements) from b into tmp at the location specified by IMM8, and copy tmp to dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_insertf64x2&ig_expand=3855)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vinsertf64x2, IMM8 = 1))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_insertf64x2<const IMM8: i32>(
    k: __mmask8,
    a: __m256d,
    b: __m128d,
) -> __m256d {
    static_assert_uimm_bits!(IMM8, 1);
    let c = _mm256_insertf64x2::<IMM8>(a, b).as_f64x4();
    let zero = _mm256_setzero_pd().as_f64x4();
    transmute(simd_select_bitmask(k, c, zero))
}

/// Copy a to dst, then insert 128 bits (composed of 2 packed double-precision (64-bit) floating-point
/// elements) from b into dst at the location specified by IMM8.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_insertf64x2&ig_expand=3856)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_insertf64x2<const IMM8: i32>(a: __m512d, b: __m128d) -> __m512d {
    static_assert_uimm_bits!(IMM8, 2);
    let b = _mm512_castpd128_pd512(b);
    match IMM8 & 3 {
        0 => simd_shuffle!(a, b, [8, 9, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle!(a, b, [0, 1, 8, 9, 4, 5, 6, 7]),
        2 => simd_shuffle!(a, b, [0, 1, 2, 3, 8, 9, 6, 7]),
        _ => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 8, 9]),
    }
}

/// Copy a to tmp, then insert 128 bits (composed of 2 packed double-precision (64-bit) floating-point
/// elements) from b into tmp at the location specified by IMM8, and copy tmp to dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_insertf64x2&ig_expand=3857)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vinsertf64x2, IMM8 = 3))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_insertf64x2<const IMM8: i32>(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m128d,
) -> __m512d {
    static_assert_uimm_bits!(IMM8, 2);
    let c = _mm512_insertf64x2::<IMM8>(a, b);
    transmute(simd_select_bitmask(k, c.as_f64x8(), src.as_f64x8()))
}

/// Copy a to tmp, then insert 128 bits (composed of 2 packed double-precision (64-bit) floating-point
/// elements) from b into tmp at the location specified by IMM8, and copy tmp to dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_insertf64x2&ig_expand=3858)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vinsertf64x2, IMM8 = 3))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_insertf64x2<const IMM8: i32>(
    k: __mmask8,
    a: __m512d,
    b: __m128d,
) -> __m512d {
    static_assert_uimm_bits!(IMM8, 2);
    let c = _mm512_insertf64x2::<IMM8>(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, c, zero))
}

/// Copy a to dst, then insert 256 bits (composed of 8 packed 32-bit integers) from b into dst at the
/// location specified by IMM8.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_inserti32x8&ig_expand=3869)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_inserti32x8<const IMM8: i32>(a: __m512i, b: __m256i) -> __m512i {
    static_assert_uimm_bits!(IMM8, 1);
    let a = a.as_i32x16();
    let b = _mm512_castsi256_si512(b).as_i32x16();
    let r: i32x16 = match IMM8 & 1 {
        0 => simd_shuffle!(
            a,
            b,
            [16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15]
        ),
        _ => simd_shuffle!(
            a,
            b,
            [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]
        ),
    };
    transmute(r)
}

/// Copy a to tmp, then insert 256 bits (composed of 8 packed 32-bit integers) from b into tmp at the
/// location specified by IMM8, and copy tmp to dst using writemask k (elements are copied from src if
/// the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_inserti32x8&ig_expand=3870)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vinserti32x8, IMM8 = 1))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_inserti32x8<const IMM8: i32>(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    b: __m256i,
) -> __m512i {
    static_assert_uimm_bits!(IMM8, 1);
    let c = _mm512_inserti32x8::<IMM8>(a, b);
    transmute(simd_select_bitmask(k, c.as_i32x16(), src.as_i32x16()))
}

/// Copy a to tmp, then insert 256 bits (composed of 8 packed 32-bit integers) from b into tmp at the
/// location specified by IMM8, and copy tmp to dst using zeromask k (elements are zeroed out if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_inserti32x8&ig_expand=3871)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vinserti32x8, IMM8 = 1))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_inserti32x8<const IMM8: i32>(
    k: __mmask16,
    a: __m512i,
    b: __m256i,
) -> __m512i {
    static_assert_uimm_bits!(IMM8, 1);
    let c = _mm512_inserti32x8::<IMM8>(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, c, zero))
}

/// Copy a to dst, then insert 128 bits (composed of 2 packed 64-bit integers) from b into dst at the
/// location specified by IMM8.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_inserti64x2&ig_expand=3872)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_inserti64x2<const IMM8: i32>(a: __m256i, b: __m128i) -> __m256i {
    static_assert_uimm_bits!(IMM8, 1);
    let b = _mm256_castsi128_si256(b);
    match IMM8 & 1 {
        0 => simd_shuffle!(a, b, [4, 5, 2, 3]),
        _ => simd_shuffle!(a, b, [0, 1, 4, 5]),
    }
}

/// Copy a to tmp, then insert 128 bits (composed of 2 packed 64-bit integers) from b into tmp at the
/// location specified by IMM8, and copy tmp to dst using writemask k (elements are copied from src if
/// the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_inserti64x2&ig_expand=3873)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vinserti64x2, IMM8 = 1))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_inserti64x2<const IMM8: i32>(
    src: __m256i,
    k: __mmask8,
    a: __m256i,
    b: __m128i,
) -> __m256i {
    static_assert_uimm_bits!(IMM8, 1);
    let c = _mm256_inserti64x2::<IMM8>(a, b);
    transmute(simd_select_bitmask(k, c.as_i64x4(), src.as_i64x4()))
}

/// Copy a to tmp, then insert 128 bits (composed of 2 packed 64-bit integers) from b into tmp at the
/// location specified by IMM8, and copy tmp to dst using zeromask k (elements are zeroed out if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_inserti64x2&ig_expand=3874)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vinserti64x2, IMM8 = 1))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_inserti64x2<const IMM8: i32>(
    k: __mmask8,
    a: __m256i,
    b: __m128i,
) -> __m256i {
    static_assert_uimm_bits!(IMM8, 1);
    let c = _mm256_inserti64x2::<IMM8>(a, b).as_i64x4();
    let zero = _mm256_setzero_si256().as_i64x4();
    transmute(simd_select_bitmask(k, c, zero))
}

/// Copy a to dst, then insert 128 bits (composed of 2 packed 64-bit integers) from b into dst at the
/// location specified by IMM8.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_inserti64x2&ig_expand=3875)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_inserti64x2<const IMM8: i32>(a: __m512i, b: __m128i) -> __m512i {
    static_assert_uimm_bits!(IMM8, 2);
    let b = _mm512_castsi128_si512(b);
    match IMM8 & 3 {
        0 => simd_shuffle!(a, b, [8, 9, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle!(a, b, [0, 1, 8, 9, 4, 5, 6, 7]),
        2 => simd_shuffle!(a, b, [0, 1, 2, 3, 8, 9, 6, 7]),
        _ => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 8, 9]),
    }
}

/// Copy a to tmp, then insert 128 bits (composed of 2 packed 64-bit integers) from b into tmp at the
/// location specified by IMM8, and copy tmp to dst using writemask k (elements are copied from src if
/// the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_inserti64x2&ig_expand=3876)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vinserti64x2, IMM8 = 3))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_inserti64x2<const IMM8: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m512i,
    b: __m128i,
) -> __m512i {
    static_assert_uimm_bits!(IMM8, 2);
    let c = _mm512_inserti64x2::<IMM8>(a, b);
    transmute(simd_select_bitmask(k, c.as_i64x8(), src.as_i64x8()))
}

/// Copy a to tmp, then insert 128 bits (composed of 2 packed 64-bit integers) from b into tmp at the
/// location specified by IMM8, and copy tmp to dst using zeromask k (elements are zeroed out if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_inserti64x2&ig_expand=3877)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vinserti64x2, IMM8 = 3))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_inserti64x2<const IMM8: i32>(
    k: __mmask8,
    a: __m512i,
    b: __m128i,
) -> __m512i {
    static_assert_uimm_bits!(IMM8, 2);
    let c = _mm512_inserti64x2::<IMM8>(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, c, zero))
}

// Convert

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundepi64_pd&ig_expand=1437)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2pd, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvt_roundepi64_pd<const ROUNDING: i32>(a: __m512i) -> __m512d {
    static_assert_rounding!(ROUNDING);
    transmute(vcvtqq2pd_512(a.as_i64x8(), ROUNDING))
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundepi64_pd&ig_expand=1438)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2pd, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvt_roundepi64_pd<const ROUNDING: i32>(
    src: __m512d,
    k: __mmask8,
    a: __m512i,
) -> __m512d {
    static_assert_rounding!(ROUNDING);
    let b = _mm512_cvt_roundepi64_pd::<ROUNDING>(a).as_f64x8();
    transmute(simd_select_bitmask(k, b, src.as_f64x8()))
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundepi64_pd&ig_expand=1439)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2pd, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvt_roundepi64_pd<const ROUNDING: i32>(
    k: __mmask8,
    a: __m512i,
) -> __m512d {
    static_assert_rounding!(ROUNDING);
    let b = _mm512_cvt_roundepi64_pd::<ROUNDING>(a).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtepi64_pd&ig_expand=1705)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_cvtepi64_pd(a: __m128i) -> __m128d {
    transmute(vcvtqq2pd_128(a.as_i64x2(), _MM_FROUND_CUR_DIRECTION))
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepi64_pd&ig_expand=1706)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_cvtepi64_pd(src: __m128d, k: __mmask8, a: __m128i) -> __m128d {
    let b = _mm_cvtepi64_pd(a).as_f64x2();
    transmute(simd_select_bitmask(k, b, src.as_f64x2()))
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtepi64_pd&ig_expand=1707)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_cvtepi64_pd(k: __mmask8, a: __m128i) -> __m128d {
    let b = _mm_cvtepi64_pd(a).as_f64x2();
    let zero = _mm_setzero_pd().as_f64x2();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtepi64_pd&ig_expand=1708)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_cvtepi64_pd(a: __m256i) -> __m256d {
    transmute(vcvtqq2pd_256(a.as_i64x4(), _MM_FROUND_CUR_DIRECTION))
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepi64_pd&ig_expand=1709)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_cvtepi64_pd(src: __m256d, k: __mmask8, a: __m256i) -> __m256d {
    let b = _mm256_cvtepi64_pd(a).as_f64x4();
    transmute(simd_select_bitmask(k, b, src.as_f64x4()))
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtepi64_pd&ig_expand=1710)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_cvtepi64_pd(k: __mmask8, a: __m256i) -> __m256d {
    let b = _mm256_cvtepi64_pd(a).as_f64x4();
    let zero = _mm256_setzero_pd().as_f64x4();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtepi64_pd&ig_expand=1711)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvtepi64_pd(a: __m512i) -> __m512d {
    transmute(vcvtqq2pd_512(a.as_i64x8(), _MM_FROUND_CUR_DIRECTION))
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepi64_pd&ig_expand=1712)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvtepi64_pd(src: __m512d, k: __mmask8, a: __m512i) -> __m512d {
    let b = _mm512_cvtepi64_pd(a).as_f64x8();
    transmute(simd_select_bitmask(k, b, src.as_f64x8()))
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtepi64_pd&ig_expand=1713)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvtepi64_pd(k: __mmask8, a: __m512i) -> __m512d {
    let b = _mm512_cvtepi64_pd(a).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundepi64_ps&ig_expand=1443)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2ps, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvt_roundepi64_ps<const ROUNDING: i32>(a: __m512i) -> __m256 {
    static_assert_rounding!(ROUNDING);
    transmute(vcvtqq2ps_512(a.as_i64x8(), ROUNDING))
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundepi64_ps&ig_expand=1444)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2ps, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvt_roundepi64_ps<const ROUNDING: i32>(
    src: __m256,
    k: __mmask8,
    a: __m512i,
) -> __m256 {
    static_assert_rounding!(ROUNDING);
    let b = _mm512_cvt_roundepi64_ps::<ROUNDING>(a).as_f32x8();
    transmute(simd_select_bitmask(k, b, src.as_f32x8()))
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundepi64_ps&ig_expand=1445)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2ps, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvt_roundepi64_ps<const ROUNDING: i32>(
    k: __mmask8,
    a: __m512i,
) -> __m256 {
    static_assert_rounding!(ROUNDING);
    let b = _mm512_cvt_roundepi64_ps::<ROUNDING>(a).as_f32x8();
    let zero = _mm256_setzero_ps().as_f32x8();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtepi64_ps&ig_expand=1723)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_cvtepi64_ps(a: __m128i) -> __m128 {
    _mm_mask_cvtepi64_ps(_mm_undefined_ps(), 0b11, a)
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepi64_ps&ig_expand=1724)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_cvtepi64_ps(src: __m128, k: __mmask8, a: __m128i) -> __m128 {
    transmute(vcvtqq2ps_128(a.as_i64x2(), src.as_f32x4(), k))
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtepi64_ps&ig_expand=1725)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_cvtepi64_ps(k: __mmask8, a: __m128i) -> __m128 {
    _mm_mask_cvtepi64_ps(_mm_setzero_ps(), k, a)
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtepi64_ps&ig_expand=1726)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_cvtepi64_ps(a: __m256i) -> __m128 {
    transmute(vcvtqq2ps_256(a.as_i64x4(), _MM_FROUND_CUR_DIRECTION))
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepi64_ps&ig_expand=1727)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_cvtepi64_ps(src: __m128, k: __mmask8, a: __m256i) -> __m128 {
    let b = _mm256_cvtepi64_ps(a).as_f32x4();
    transmute(simd_select_bitmask(k, b, src.as_f32x4()))
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtepi64_ps&ig_expand=1728)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_cvtepi64_ps(k: __mmask8, a: __m256i) -> __m128 {
    let b = _mm256_cvtepi64_ps(a).as_f32x4();
    let zero = _mm_setzero_ps().as_f32x4();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtepi64_ps&ig_expand=1729)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvtepi64_ps(a: __m512i) -> __m256 {
    transmute(vcvtqq2ps_512(a.as_i64x8(), _MM_FROUND_CUR_DIRECTION))
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepi64_ps&ig_expand=1730)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvtepi64_ps(src: __m256, k: __mmask8, a: __m512i) -> __m256 {
    let b = _mm512_cvtepi64_ps(a).as_f32x8();
    transmute(simd_select_bitmask(k, b, src.as_f32x8()))
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtepi64_ps&ig_expand=1731)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvtepi64_ps(k: __mmask8, a: __m512i) -> __m256 {
    let b = _mm512_cvtepi64_ps(a).as_f32x8();
    let zero = _mm256_setzero_ps().as_f32x8();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundepu64_pd&ig_expand=1455)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvt_roundepu64_pd<const ROUNDING: i32>(a: __m512i) -> __m512d {
    static_assert_rounding!(ROUNDING);
    transmute(vcvtuqq2pd_512(a.as_u64x8(), ROUNDING))
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundepu64_pd&ig_expand=1456)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvt_roundepu64_pd<const ROUNDING: i32>(
    src: __m512d,
    k: __mmask8,
    a: __m512i,
) -> __m512d {
    static_assert_rounding!(ROUNDING);
    let b = _mm512_cvt_roundepu64_pd::<ROUNDING>(a).as_f64x8();
    transmute(simd_select_bitmask(k, b, src.as_f64x8()))
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundepu64_pd&ig_expand=1457)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvt_roundepu64_pd<const ROUNDING: i32>(
    k: __mmask8,
    a: __m512i,
) -> __m512d {
    static_assert_rounding!(ROUNDING);
    let b = _mm512_cvt_roundepu64_pd::<ROUNDING>(a).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtepu64_pd&ig_expand=1827)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_cvtepu64_pd(a: __m128i) -> __m128d {
    transmute(vcvtuqq2pd_128(a.as_u64x2(), _MM_FROUND_CUR_DIRECTION))
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepu64_pd&ig_expand=1828)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_cvtepu64_pd(src: __m128d, k: __mmask8, a: __m128i) -> __m128d {
    let b = _mm_cvtepu64_pd(a).as_f64x2();
    transmute(simd_select_bitmask(k, b, src.as_f64x2()))
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtepu64_pd&ig_expand=1829)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_cvtepu64_pd(k: __mmask8, a: __m128i) -> __m128d {
    let b = _mm_cvtepu64_pd(a).as_f64x2();
    let zero = _mm_setzero_pd().as_f64x2();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtepu64_pd&ig_expand=1830)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_cvtepu64_pd(a: __m256i) -> __m256d {
    transmute(vcvtuqq2pd_256(a.as_u64x4(), _MM_FROUND_CUR_DIRECTION))
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepu64_pd&ig_expand=1831)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_cvtepu64_pd(src: __m256d, k: __mmask8, a: __m256i) -> __m256d {
    let b = _mm256_cvtepu64_pd(a).as_f64x4();
    transmute(simd_select_bitmask(k, b, src.as_f64x4()))
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtepu64_pd&ig_expand=1832)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_cvtepu64_pd(k: __mmask8, a: __m256i) -> __m256d {
    let b = _mm256_cvtepu64_pd(a).as_f64x4();
    let zero = _mm256_setzero_pd().as_f64x4();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtepu64_pd&ig_expand=1833)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvtepu64_pd(a: __m512i) -> __m512d {
    transmute(vcvtuqq2pd_512(a.as_u64x8(), _MM_FROUND_CUR_DIRECTION))
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepu64_pd&ig_expand=1834)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvtepu64_pd(src: __m512d, k: __mmask8, a: __m512i) -> __m512d {
    let b = _mm512_cvtepu64_pd(a).as_f64x8();
    transmute(simd_select_bitmask(k, b, src.as_f64x8()))
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtepu64_pd&ig_expand=1835)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvtepu64_pd(k: __mmask8, a: __m512i) -> __m512d {
    let b = _mm512_cvtepu64_pd(a).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundepu64_ps&ig_expand=1461)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvt_roundepu64_ps<const ROUNDING: i32>(a: __m512i) -> __m256 {
    static_assert_rounding!(ROUNDING);
    transmute(vcvtuqq2ps_512(a.as_u64x8(), ROUNDING))
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundepu64_ps&ig_expand=1462)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvt_roundepu64_ps<const ROUNDING: i32>(
    src: __m256,
    k: __mmask8,
    a: __m512i,
) -> __m256 {
    static_assert_rounding!(ROUNDING);
    let b = _mm512_cvt_roundepu64_ps::<ROUNDING>(a).as_f32x8();
    transmute(simd_select_bitmask(k, b, src.as_f32x8()))
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundepu64_ps&ig_expand=1463)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvt_roundepu64_ps<const ROUNDING: i32>(
    k: __mmask8,
    a: __m512i,
) -> __m256 {
    static_assert_rounding!(ROUNDING);
    let b = _mm512_cvt_roundepu64_ps::<ROUNDING>(a).as_f32x8();
    let zero = _mm256_setzero_ps().as_f32x8();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtepu64_ps&ig_expand=1845)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_cvtepu64_ps(a: __m128i) -> __m128 {
    _mm_mask_cvtepu64_ps(_mm_undefined_ps(), 0b11, a)
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepu64_ps&ig_expand=1846)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_cvtepu64_ps(src: __m128, k: __mmask8, a: __m128i) -> __m128 {
    transmute(vcvtuqq2ps_128(a.as_u64x2(), src.as_f32x4(), k))
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtepu64_ps&ig_expand=1847)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_cvtepu64_ps(k: __mmask8, a: __m128i) -> __m128 {
    _mm_mask_cvtepu64_ps(_mm_setzero_ps(), k, a)
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtepu64_ps&ig_expand=1848)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_cvtepu64_ps(a: __m256i) -> __m128 {
    transmute(vcvtuqq2ps_256(a.as_u64x4(), _MM_FROUND_CUR_DIRECTION))
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepu64_ps&ig_expand=1849)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_cvtepu64_ps(src: __m128, k: __mmask8, a: __m256i) -> __m128 {
    let b = _mm256_cvtepu64_ps(a).as_f32x4();
    transmute(simd_select_bitmask(k, b, src.as_f32x4()))
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtepu64_ps&ig_expand=1850)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_cvtepu64_ps(k: __mmask8, a: __m256i) -> __m128 {
    let b = _mm256_cvtepu64_ps(a).as_f32x4();
    let zero = _mm_setzero_ps().as_f32x4();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtepu64_ps&ig_expand=1851)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvtepu64_ps(a: __m512i) -> __m256 {
    transmute(vcvtuqq2ps_512(a.as_u64x8(), _MM_FROUND_CUR_DIRECTION))
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepu64_ps&ig_expand=1852)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvtepu64_ps(src: __m256, k: __mmask8, a: __m512i) -> __m256 {
    let b = _mm512_cvtepu64_ps(a).as_f32x8();
    transmute(simd_select_bitmask(k, b, src.as_f32x8()))
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtepu64_ps&ig_expand=1853)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvtepu64_ps(k: __mmask8, a: __m512i) -> __m256 {
    let b = _mm512_cvtepu64_ps(a).as_f32x8();
    let zero = _mm256_setzero_ps().as_f32x8();
    transmute(simd_select_bitmask(k, b, zero))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundpd_epi64&ig_expand=1472)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2qq, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvt_roundpd_epi64<const ROUNDING: i32>(a: __m512d) -> __m512i {
    static_assert_rounding!(ROUNDING);
    _mm512_mask_cvt_roundpd_epi64::<ROUNDING>(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundpd_epi64&ig_expand=1473)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2qq, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvt_roundpd_epi64<const ROUNDING: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m512d,
) -> __m512i {
    static_assert_rounding!(ROUNDING);
    transmute(vcvtpd2qq_512(a.as_f64x8(), src.as_i64x8(), k, ROUNDING))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundpd_epi64&ig_expand=1474)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2qq, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvt_roundpd_epi64<const ROUNDING: i32>(
    k: __mmask8,
    a: __m512d,
) -> __m512i {
    static_assert_rounding!(ROUNDING);
    _mm512_mask_cvt_roundpd_epi64::<ROUNDING>(_mm512_setzero_si512(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtpd_epi64&ig_expand=1941)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_cvtpd_epi64(a: __m128d) -> __m128i {
    _mm_mask_cvtpd_epi64(_mm_undefined_si128(), 0b11, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtpd_epi64&ig_expand=1942)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_cvtpd_epi64(src: __m128i, k: __mmask8, a: __m128d) -> __m128i {
    transmute(vcvtpd2qq_128(a.as_f64x2(), src.as_i64x2(), k))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtpd_epi64&ig_expand=1943)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_cvtpd_epi64(k: __mmask8, a: __m128d) -> __m128i {
    _mm_mask_cvtpd_epi64(_mm_setzero_si128(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtpd_epi64&ig_expand=1944)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_cvtpd_epi64(a: __m256d) -> __m256i {
    _mm256_mask_cvtpd_epi64(_mm256_undefined_si256(), 0xf, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtpd_epi64&ig_expand=1945)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_cvtpd_epi64(src: __m256i, k: __mmask8, a: __m256d) -> __m256i {
    transmute(vcvtpd2qq_256(a.as_f64x4(), src.as_i64x4(), k))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtpd_epi64&ig_expand=1946)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_cvtpd_epi64(k: __mmask8, a: __m256d) -> __m256i {
    _mm256_mask_cvtpd_epi64(_mm256_setzero_si256(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtpd_epi64&ig_expand=1947)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvtpd_epi64(a: __m512d) -> __m512i {
    _mm512_mask_cvtpd_epi64(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtpd_epi64&ig_expand=1948)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvtpd_epi64(src: __m512i, k: __mmask8, a: __m512d) -> __m512i {
    transmute(vcvtpd2qq_512(
        a.as_f64x8(),
        src.as_i64x8(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtpd_epi64&ig_expand=1949)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvtpd_epi64(k: __mmask8, a: __m512d) -> __m512i {
    _mm512_mask_cvtpd_epi64(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundps_epi64&ig_expand=1514)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2qq, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvt_roundps_epi64<const ROUNDING: i32>(a: __m256) -> __m512i {
    static_assert_rounding!(ROUNDING);
    _mm512_mask_cvt_roundps_epi64::<ROUNDING>(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundps_epi64&ig_expand=1515)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2qq, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvt_roundps_epi64<const ROUNDING: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m256,
) -> __m512i {
    static_assert_rounding!(ROUNDING);
    transmute(vcvtps2qq_512(a.as_f32x8(), src.as_i64x8(), k, ROUNDING))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundps_epi64&ig_expand=1516)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2qq, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvt_roundps_epi64<const ROUNDING: i32>(
    k: __mmask8,
    a: __m256,
) -> __m512i {
    static_assert_rounding!(ROUNDING);
    _mm512_mask_cvt_roundps_epi64::<ROUNDING>(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtps_epi64&ig_expand=2075)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_cvtps_epi64(a: __m128) -> __m128i {
    _mm_mask_cvtps_epi64(_mm_undefined_si128(), 0b11, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtps_epi64&ig_expand=2076)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_cvtps_epi64(src: __m128i, k: __mmask8, a: __m128) -> __m128i {
    transmute(vcvtps2qq_128(a.as_f32x4(), src.as_i64x2(), k))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtps_epi64&ig_expand=2077)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_cvtps_epi64(k: __mmask8, a: __m128) -> __m128i {
    _mm_mask_cvtps_epi64(_mm_setzero_si128(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtps_epi64&ig_expand=2078)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_cvtps_epi64(a: __m128) -> __m256i {
    _mm256_mask_cvtps_epi64(_mm256_undefined_si256(), 0xf, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtps_epi64&ig_expand=2079)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_cvtps_epi64(src: __m256i, k: __mmask8, a: __m128) -> __m256i {
    transmute(vcvtps2qq_256(a.as_f32x4(), src.as_i64x4(), k))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtps_epi64&ig_expand=2080)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_cvtps_epi64(k: __mmask8, a: __m128) -> __m256i {
    _mm256_mask_cvtps_epi64(_mm256_setzero_si256(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtps_epi64&ig_expand=2081)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvtps_epi64(a: __m256) -> __m512i {
    _mm512_mask_cvtps_epi64(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtps_epi64&ig_expand=2082)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvtps_epi64(src: __m512i, k: __mmask8, a: __m256) -> __m512i {
    transmute(vcvtps2qq_512(
        a.as_f32x8(),
        src.as_i64x8(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtps_epi64&ig_expand=2083)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvtps_epi64(k: __mmask8, a: __m256) -> __m512i {
    _mm512_mask_cvtps_epi64(_mm512_setzero_si512(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundpd_epu64&ig_expand=1478)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvt_roundpd_epu64<const ROUNDING: i32>(a: __m512d) -> __m512i {
    static_assert_rounding!(ROUNDING);
    _mm512_mask_cvt_roundpd_epu64::<ROUNDING>(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundpd_epu64&ig_expand=1479)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvt_roundpd_epu64<const ROUNDING: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m512d,
) -> __m512i {
    static_assert_rounding!(ROUNDING);
    transmute(vcvtpd2uqq_512(a.as_f64x8(), src.as_u64x8(), k, ROUNDING))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundpd_epu64&ig_expand=1480)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvt_roundpd_epu64<const ROUNDING: i32>(
    k: __mmask8,
    a: __m512d,
) -> __m512i {
    static_assert_rounding!(ROUNDING);
    _mm512_mask_cvt_roundpd_epu64::<ROUNDING>(_mm512_setzero_si512(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtpd_epu64&ig_expand=1959)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_cvtpd_epu64(a: __m128d) -> __m128i {
    _mm_mask_cvtpd_epu64(_mm_undefined_si128(), 0b11, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtpd_epu64&ig_expand=1960)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_cvtpd_epu64(src: __m128i, k: __mmask8, a: __m128d) -> __m128i {
    transmute(vcvtpd2uqq_128(a.as_f64x2(), src.as_u64x2(), k))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtpd_epu64&ig_expand=1961)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_cvtpd_epu64(k: __mmask8, a: __m128d) -> __m128i {
    _mm_mask_cvtpd_epu64(_mm_setzero_si128(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtpd_epu64&ig_expand=1962)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_cvtpd_epu64(a: __m256d) -> __m256i {
    _mm256_mask_cvtpd_epu64(_mm256_undefined_si256(), 0xf, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtpd_epu64&ig_expand=1963)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_cvtpd_epu64(src: __m256i, k: __mmask8, a: __m256d) -> __m256i {
    transmute(vcvtpd2uqq_256(a.as_f64x4(), src.as_u64x4(), k))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtpd_epu64&ig_expand=1964)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_cvtpd_epu64(k: __mmask8, a: __m256d) -> __m256i {
    _mm256_mask_cvtpd_epu64(_mm256_setzero_si256(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtpd_epu64&ig_expand=1965)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvtpd_epu64(a: __m512d) -> __m512i {
    _mm512_mask_cvtpd_epu64(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtpd_epu64&ig_expand=1966)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvtpd_epu64(src: __m512i, k: __mmask8, a: __m512d) -> __m512i {
    transmute(vcvtpd2uqq_512(
        a.as_f64x8(),
        src.as_u64x8(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtpd_epu64&ig_expand=1967)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvtpd_epu64(k: __mmask8, a: __m512d) -> __m512i {
    _mm512_mask_cvtpd_epu64(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundps_epu64&ig_expand=1520)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2uqq, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvt_roundps_epu64<const ROUNDING: i32>(a: __m256) -> __m512i {
    static_assert_rounding!(ROUNDING);
    _mm512_mask_cvt_roundps_epu64::<ROUNDING>(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundps_epu64&ig_expand=1521)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2uqq, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvt_roundps_epu64<const ROUNDING: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m256,
) -> __m512i {
    static_assert_rounding!(ROUNDING);
    transmute(vcvtps2uqq_512(a.as_f32x8(), src.as_u64x8(), k, ROUNDING))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// - (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
/// - (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
/// - (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
/// - (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
/// - _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundps_epu64&ig_expand=1522)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2uqq, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvt_roundps_epu64<const ROUNDING: i32>(
    k: __mmask8,
    a: __m256,
) -> __m512i {
    static_assert_rounding!(ROUNDING);
    _mm512_mask_cvt_roundps_epu64::<ROUNDING>(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtps_epu64&ig_expand=2093)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_cvtps_epu64(a: __m128) -> __m128i {
    _mm_mask_cvtps_epu64(_mm_undefined_si128(), 0b11, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtps_epu64&ig_expand=2094)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_cvtps_epu64(src: __m128i, k: __mmask8, a: __m128) -> __m128i {
    transmute(vcvtps2uqq_128(a.as_f32x4(), src.as_u64x2(), k))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtps_epu64&ig_expand=2095)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_cvtps_epu64(k: __mmask8, a: __m128) -> __m128i {
    _mm_mask_cvtps_epu64(_mm_setzero_si128(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtps_epu64&ig_expand=2096)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_cvtps_epu64(a: __m128) -> __m256i {
    _mm256_mask_cvtps_epu64(_mm256_undefined_si256(), 0xf, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtps_epu64&ig_expand=2097)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_cvtps_epu64(src: __m256i, k: __mmask8, a: __m128) -> __m256i {
    transmute(vcvtps2uqq_256(a.as_f32x4(), src.as_u64x4(), k))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtps_epu64&ig_expand=2098)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_cvtps_epu64(k: __mmask8, a: __m128) -> __m256i {
    _mm256_mask_cvtps_epu64(_mm256_setzero_si256(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtps_epu64&ig_expand=2099)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvtps_epu64(a: __m256) -> __m512i {
    _mm512_mask_cvtps_epu64(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtps_epu64&ig_expand=2100)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvtps_epu64(src: __m512i, k: __mmask8, a: __m256) -> __m512i {
    transmute(vcvtps2uqq_512(
        a.as_f32x8(),
        src.as_u64x8(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtps_epu64&ig_expand=2101)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvtps_epu64(k: __mmask8, a: __m256) -> __m512i {
    _mm512_mask_cvtps_epu64(_mm512_setzero_si512(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst. Exceptions can be suppressed by passing _MM_FROUND_NO_EXC
/// to the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtt_roundpd_epi64&ig_expand=2264)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttpd2qq, SAE = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvtt_roundpd_epi64<const SAE: i32>(a: __m512d) -> __m512i {
    static_assert_sae!(SAE);
    _mm512_mask_cvtt_roundpd_epi64::<SAE>(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set). Exceptions can be suppressed by passing _MM_FROUND_NO_EXC to the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtt_roundpd_epi64&ig_expand=2265)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttpd2qq, SAE = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvtt_roundpd_epi64<const SAE: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m512d,
) -> __m512i {
    static_assert_sae!(SAE);
    transmute(vcvttpd2qq_512(a.as_f64x8(), src.as_i64x8(), k, SAE))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set). Exceptions can be suppressed by passing _MM_FROUND_NO_EXC to the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtt_roundpd_epi64&ig_expand=2266)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttpd2qq, SAE = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvtt_roundpd_epi64<const SAE: i32>(k: __mmask8, a: __m512d) -> __m512i {
    static_assert_sae!(SAE);
    _mm512_mask_cvtt_roundpd_epi64::<SAE>(_mm512_setzero_si512(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvttpd_epi64&ig_expand=2329)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_cvttpd_epi64(a: __m128d) -> __m128i {
    _mm_mask_cvttpd_epi64(_mm_undefined_si128(), 0b11, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvttpd_epi64&ig_expand=2330)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_cvttpd_epi64(src: __m128i, k: __mmask8, a: __m128d) -> __m128i {
    transmute(vcvttpd2qq_128(a.as_f64x2(), src.as_i64x2(), k))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvttpd_epi64&ig_expand=2331)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_cvttpd_epi64(k: __mmask8, a: __m128d) -> __m128i {
    _mm_mask_cvttpd_epi64(_mm_setzero_si128(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvttpd_epi64&ig_expand=2332)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_cvttpd_epi64(a: __m256d) -> __m256i {
    _mm256_mask_cvttpd_epi64(_mm256_undefined_si256(), 0xf, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvttpd_epi64&ig_expand=2333)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_cvttpd_epi64(src: __m256i, k: __mmask8, a: __m256d) -> __m256i {
    transmute(vcvttpd2qq_256(a.as_f64x4(), src.as_i64x4(), k))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvttpd_epi64&ig_expand=2334)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_cvttpd_epi64(k: __mmask8, a: __m256d) -> __m256i {
    _mm256_mask_cvttpd_epi64(_mm256_setzero_si256(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvttpd_epi64&ig_expand=2335)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvttpd_epi64(a: __m512d) -> __m512i {
    _mm512_mask_cvttpd_epi64(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvttpd_epi64&ig_expand=2336)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvttpd_epi64(src: __m512i, k: __mmask8, a: __m512d) -> __m512i {
    transmute(vcvttpd2qq_512(
        a.as_f64x8(),
        src.as_i64x8(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvttpd_epi64&ig_expand=2337)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvttpd_epi64(k: __mmask8, a: __m512d) -> __m512i {
    _mm512_mask_cvttpd_epi64(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst. Exceptions can be suppressed by passing _MM_FROUND_NO_EXC
/// to the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtt_roundps_epi64&ig_expand=2294)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttps2qq, SAE = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvtt_roundps_epi64<const SAE: i32>(a: __m256) -> __m512i {
    static_assert_sae!(SAE);
    _mm512_mask_cvtt_roundps_epi64::<SAE>(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set). Exceptions can be suppressed by passing _MM_FROUND_NO_EXC to the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtt_roundps_epi64&ig_expand=2295)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttps2qq, SAE = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvtt_roundps_epi64<const SAE: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m256,
) -> __m512i {
    static_assert_sae!(SAE);
    transmute(vcvttps2qq_512(a.as_f32x8(), src.as_i64x8(), k, SAE))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set). Exceptions can be suppressed by passing _MM_FROUND_NO_EXC to the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtt_roundps_epi64&ig_expand=2296)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttps2qq, SAE = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvtt_roundps_epi64<const SAE: i32>(k: __mmask8, a: __m256) -> __m512i {
    static_assert_sae!(SAE);
    _mm512_mask_cvtt_roundps_epi64::<SAE>(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvttps_epi64&ig_expand=2420)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_cvttps_epi64(a: __m128) -> __m128i {
    _mm_mask_cvttps_epi64(_mm_undefined_si128(), 0b11, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvttps_epi64&ig_expand=2421)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_cvttps_epi64(src: __m128i, k: __mmask8, a: __m128) -> __m128i {
    transmute(vcvttps2qq_128(a.as_f32x4(), src.as_i64x2(), k))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvttps_epi64&ig_expand=2422)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_cvttps_epi64(k: __mmask8, a: __m128) -> __m128i {
    _mm_mask_cvttps_epi64(_mm_setzero_si128(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvttps_epi64&ig_expand=2423)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_cvttps_epi64(a: __m128) -> __m256i {
    _mm256_mask_cvttps_epi64(_mm256_undefined_si256(), 0xf, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvttps_epi64&ig_expand=2424)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_cvttps_epi64(src: __m256i, k: __mmask8, a: __m128) -> __m256i {
    transmute(vcvttps2qq_256(a.as_f32x4(), src.as_i64x4(), k))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvttps_epi64&ig_expand=2425)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_cvttps_epi64(k: __mmask8, a: __m128) -> __m256i {
    _mm256_mask_cvttps_epi64(_mm256_setzero_si256(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvttps_epi64&ig_expand=2426)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvttps_epi64(a: __m256) -> __m512i {
    _mm512_mask_cvttps_epi64(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvttps_epi64&ig_expand=2427)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvttps_epi64(src: __m512i, k: __mmask8, a: __m256) -> __m512i {
    transmute(vcvttps2qq_512(
        a.as_f32x8(),
        src.as_i64x8(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvttps_epi64&ig_expand=2428)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvttps_epi64(k: __mmask8, a: __m256) -> __m512i {
    _mm512_mask_cvttps_epi64(_mm512_setzero_si512(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst. Exceptions can be suppressed by passing _MM_FROUND_NO_EXC
/// to the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtpd_epu64&ig_expand=1965)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttpd2uqq, SAE = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvtt_roundpd_epu64<const SAE: i32>(a: __m512d) -> __m512i {
    static_assert_sae!(SAE);
    _mm512_mask_cvtt_roundpd_epu64::<SAE>(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set). Exceptions can be suppressed by passing _MM_FROUND_NO_EXC to the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtt_roundpd_epu64&ig_expand=1966)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttpd2uqq, SAE = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvtt_roundpd_epu64<const SAE: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m512d,
) -> __m512i {
    static_assert_sae!(SAE);
    transmute(vcvttpd2uqq_512(a.as_f64x8(), src.as_u64x8(), k, SAE))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set). Exceptions can be suppressed by passing _MM_FROUND_NO_EXC to the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtt_roundpd_epu64&ig_expand=1967)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttpd2uqq, SAE = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvtt_roundpd_epu64<const SAE: i32>(k: __mmask8, a: __m512d) -> __m512i {
    static_assert_sae!(SAE);
    _mm512_mask_cvtt_roundpd_epu64::<SAE>(_mm512_setzero_si512(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvttpd_epu64&ig_expand=2347)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_cvttpd_epu64(a: __m128d) -> __m128i {
    _mm_mask_cvttpd_epu64(_mm_undefined_si128(), 0b11, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvttpd_epu64&ig_expand=2348)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_cvttpd_epu64(src: __m128i, k: __mmask8, a: __m128d) -> __m128i {
    transmute(vcvttpd2uqq_128(a.as_f64x2(), src.as_u64x2(), k))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvttpd_epu64&ig_expand=2349)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_cvttpd_epu64(k: __mmask8, a: __m128d) -> __m128i {
    _mm_mask_cvttpd_epu64(_mm_setzero_si128(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvttpd_epu64&ig_expand=2350)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_cvttpd_epu64(a: __m256d) -> __m256i {
    _mm256_mask_cvttpd_epu64(_mm256_undefined_si256(), 0xf, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the results in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvttpd_epu64&ig_expand=2351)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_cvttpd_epu64(src: __m256i, k: __mmask8, a: __m256d) -> __m256i {
    transmute(vcvttpd2uqq_256(a.as_f64x4(), src.as_u64x4(), k))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the results in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvttpd_epu64&ig_expand=2352)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_cvttpd_epu64(k: __mmask8, a: __m256d) -> __m256i {
    _mm256_mask_cvttpd_epu64(_mm256_setzero_si256(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvttpd_epu64&ig_expand=2353)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvttpd_epu64(a: __m512d) -> __m512i {
    _mm512_mask_cvttpd_epu64(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvttpd_epu64&ig_expand=2354)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvttpd_epu64(src: __m512i, k: __mmask8, a: __m512d) -> __m512i {
    transmute(vcvttpd2uqq_512(
        a.as_f64x8(),
        src.as_u64x8(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
///
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvttpd_epu64&ig_expand=2355)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvttpd_epu64(k: __mmask8, a: __m512d) -> __m512i {
    _mm512_mask_cvttpd_epu64(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst. Exceptions can be suppressed by passing _MM_FROUND_NO_EXC
/// to the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtt_roundps_epu64&ig_expand=2300)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttps2uqq, SAE = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvtt_roundps_epu64<const SAE: i32>(a: __m256) -> __m512i {
    static_assert_sae!(SAE);
    _mm512_mask_cvtt_roundps_epu64::<SAE>(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set). Exceptions can be suppressed by passing _MM_FROUND_NO_EXC to the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtt_roundps_epu64&ig_expand=2301)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttps2uqq, SAE = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvtt_roundps_epu64<const SAE: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m256,
) -> __m512i {
    static_assert_sae!(SAE);
    transmute(vcvttps2uqq_512(a.as_f32x8(), src.as_u64x8(), k, SAE))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set). Exceptions can be suppressed by passing _MM_FROUND_NO_EXC to the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtt_roundps_epu64&ig_expand=2302)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttps2uqq, SAE = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvtt_roundps_epu64<const SAE: i32>(k: __mmask8, a: __m256) -> __m512i {
    static_assert_sae!(SAE);
    _mm512_mask_cvtt_roundps_epu64::<SAE>(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvttps_epu64&ig_expand=2438)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_cvttps_epu64(a: __m128) -> __m128i {
    _mm_mask_cvttps_epu64(_mm_undefined_si128(), 0b11, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvttps_epu64&ig_expand=2439)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_cvttps_epu64(src: __m128i, k: __mmask8, a: __m128) -> __m128i {
    transmute(vcvttps2uqq_128(a.as_f32x4(), src.as_u64x2(), k))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvttps_epu64&ig_expand=2440)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_cvttps_epu64(k: __mmask8, a: __m128) -> __m128i {
    _mm_mask_cvttps_epu64(_mm_setzero_si128(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvttps_epu64&ig_expand=2441)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_cvttps_epu64(a: __m128) -> __m256i {
    _mm256_mask_cvttps_epu64(_mm256_undefined_si256(), 0xf, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvttps_epu64&ig_expand=2442)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_cvttps_epu64(src: __m256i, k: __mmask8, a: __m128) -> __m256i {
    transmute(vcvttps2uqq_256(a.as_f32x4(), src.as_u64x4(), k))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvttps_epu64&ig_expand=2443)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvttps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_cvttps_epu64(k: __mmask8, a: __m128) -> __m256i {
    _mm256_mask_cvttps_epu64(_mm256_setzero_si256(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvttps_epu64&ig_expand=2444)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_cvttps_epu64(a: __m256) -> __m512i {
    _mm512_mask_cvttps_epu64(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst using writemask k (elements are copied from src if the
/// corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvttps_epu64&ig_expand=2445)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_cvttps_epu64(src: __m512i, k: __mmask8, a: __m256) -> __m512i {
    transmute(vcvttps2uqq_512(
        a.as_f32x8(),
        src.as_u64x8(),
        k,
        _MM_FROUND_CUR_DIRECTION,
    ))
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst using zeromask k (elements are zeroed out if the corresponding
/// bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvttps_epu64&ig_expand=2446)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_cvttps_epu64(k: __mmask8, a: __m256) -> __m512i {
    _mm512_mask_cvttps_epu64(_mm512_setzero_si512(), k, a)
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.sitofp.round.v2f64.v2i64"]
    fn vcvtqq2pd_128(a: i64x2, rounding: i32) -> f64x2;
    #[link_name = "llvm.x86.avx512.sitofp.round.v4f64.v4i64"]
    fn vcvtqq2pd_256(a: i64x4, rounding: i32) -> f64x4;
    #[link_name = "llvm.x86.avx512.sitofp.round.v8f64.v8i64"]
    fn vcvtqq2pd_512(a: i64x8, rounding: i32) -> f64x8;

    #[link_name = "llvm.x86.avx512.mask.cvtqq2ps.128"]
    fn vcvtqq2ps_128(a: i64x2, src: f32x4, k: __mmask8) -> f32x4;
    #[link_name = "llvm.x86.avx512.sitofp.round.v4f32.v4i64"]
    fn vcvtqq2ps_256(a: i64x4, rounding: i32) -> f32x4;
    #[link_name = "llvm.x86.avx512.sitofp.round.v8f32.v8i64"]
    fn vcvtqq2ps_512(a: i64x8, rounding: i32) -> f32x8;

    #[link_name = "llvm.x86.avx512.uitofp.round.v2f64.v2u64"]
    fn vcvtuqq2pd_128(a: u64x2, rounding: i32) -> f64x2;
    #[link_name = "llvm.x86.avx512.uitofp.round.v4f64.v4u64"]
    fn vcvtuqq2pd_256(a: u64x4, rounding: i32) -> f64x4;
    #[link_name = "llvm.x86.avx512.uitofp.round.v8f64.v8u64"]
    fn vcvtuqq2pd_512(a: u64x8, rounding: i32) -> f64x8;

    #[link_name = "llvm.x86.avx512.mask.cvtuqq2ps.128"]
    fn vcvtuqq2ps_128(a: u64x2, src: f32x4, k: __mmask8) -> f32x4;
    #[link_name = "llvm.x86.avx512.uitofp.round.v4f32.v4u64"]
    fn vcvtuqq2ps_256(a: u64x4, rounding: i32) -> f32x4;
    #[link_name = "llvm.x86.avx512.uitofp.round.v8f32.v8u64"]
    fn vcvtuqq2ps_512(a: u64x8, rounding: i32) -> f32x8;

    #[link_name = "llvm.x86.avx512.mask.cvtpd2qq.128"]
    fn vcvtpd2qq_128(a: f64x2, src: i64x2, k: __mmask8) -> i64x2;
    #[link_name = "llvm.x86.avx512.mask.cvtpd2qq.256"]
    fn vcvtpd2qq_256(a: f64x4, src: i64x4, k: __mmask8) -> i64x4;
    #[link_name = "llvm.x86.avx512.mask.cvtpd2qq.512"]
    fn vcvtpd2qq_512(a: f64x8, src: i64x8, k: __mmask8, rounding: i32) -> i64x8;

    #[link_name = "llvm.x86.avx512.mask.cvtps2qq.128"]
    fn vcvtps2qq_128(a: f32x4, src: i64x2, k: __mmask8) -> i64x2;
    #[link_name = "llvm.x86.avx512.mask.cvtps2qq.256"]
    fn vcvtps2qq_256(a: f32x4, src: i64x4, k: __mmask8) -> i64x4;
    #[link_name = "llvm.x86.avx512.mask.cvtps2qq.512"]
    fn vcvtps2qq_512(a: f32x8, src: i64x8, k: __mmask8, rounding: i32) -> i64x8;

    #[link_name = "llvm.x86.avx512.mask.cvtpd2uqq.128"]
    fn vcvtpd2uqq_128(a: f64x2, src: u64x2, k: __mmask8) -> u64x2;
    #[link_name = "llvm.x86.avx512.mask.cvtpd2uqq.256"]
    fn vcvtpd2uqq_256(a: f64x4, src: u64x4, k: __mmask8) -> u64x4;
    #[link_name = "llvm.x86.avx512.mask.cvtpd2uqq.512"]
    fn vcvtpd2uqq_512(a: f64x8, src: u64x8, k: __mmask8, rounding: i32) -> u64x8;

    #[link_name = "llvm.x86.avx512.mask.cvtps2uqq.128"]
    fn vcvtps2uqq_128(a: f32x4, src: u64x2, k: __mmask8) -> u64x2;
    #[link_name = "llvm.x86.avx512.mask.cvtps2uqq.256"]
    fn vcvtps2uqq_256(a: f32x4, src: u64x4, k: __mmask8) -> u64x4;
    #[link_name = "llvm.x86.avx512.mask.cvtps2uqq.512"]
    fn vcvtps2uqq_512(a: f32x8, src: u64x8, k: __mmask8, rounding: i32) -> u64x8;

    #[link_name = "llvm.x86.avx512.mask.cvttpd2qq.128"]
    fn vcvttpd2qq_128(a: f64x2, src: i64x2, k: __mmask8) -> i64x2;
    #[link_name = "llvm.x86.avx512.mask.cvttpd2qq.256"]
    fn vcvttpd2qq_256(a: f64x4, src: i64x4, k: __mmask8) -> i64x4;
    #[link_name = "llvm.x86.avx512.mask.cvttpd2qq.512"]
    fn vcvttpd2qq_512(a: f64x8, src: i64x8, k: __mmask8, sae: i32) -> i64x8;

    #[link_name = "llvm.x86.avx512.mask.cvttps2qq.128"]
    fn vcvttps2qq_128(a: f32x4, src: i64x2, k: __mmask8) -> i64x2;
    #[link_name = "llvm.x86.avx512.mask.cvttps2qq.256"]
    fn vcvttps2qq_256(a: f32x4, src: i64x4, k: __mmask8) -> i64x4;
    #[link_name = "llvm.x86.avx512.mask.cvttps2qq.512"]
    fn vcvttps2qq_512(a: f32x8, src: i64x8, k: __mmask8, sae: i32) -> i64x8;

    #[link_name = "llvm.x86.avx512.mask.cvttpd2uqq.128"]
    fn vcvttpd2uqq_128(a: f64x2, src: u64x2, k: __mmask8) -> u64x2;
    #[link_name = "llvm.x86.avx512.mask.cvttpd2uqq.256"]
    fn vcvttpd2uqq_256(a: f64x4, src: u64x4, k: __mmask8) -> u64x4;
    #[link_name = "llvm.x86.avx512.mask.cvttpd2uqq.512"]
    fn vcvttpd2uqq_512(a: f64x8, src: u64x8, k: __mmask8, sae: i32) -> u64x8;

    #[link_name = "llvm.x86.avx512.mask.cvttps2uqq.128"]
    fn vcvttps2uqq_128(a: f32x4, src: u64x2, k: __mmask8) -> u64x2;
    #[link_name = "llvm.x86.avx512.mask.cvttps2uqq.256"]
    fn vcvttps2uqq_256(a: f32x4, src: u64x4, k: __mmask8) -> u64x4;
    #[link_name = "llvm.x86.avx512.mask.cvttps2uqq.512"]
    fn vcvttps2uqq_512(a: f32x8, src: u64x8, k: __mmask8, sae: i32) -> u64x8;
}

#[cfg(test)]
mod tests {
    use super::*;

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;
    use crate::mem::transmute;

    const OPRND1_64: f64 = unsafe { transmute(0x3333333333333333_u64) };
    const OPRND2_64: f64 = unsafe { transmute(0x5555555555555555_u64) };

    const AND_64: f64 = unsafe { transmute(0x1111111111111111_u64) };
    const ANDN_64: f64 = unsafe { transmute(0x4444444444444444_u64) };
    const OR_64: f64 = unsafe { transmute(0x7777777777777777_u64) };
    const XOR_64: f64 = unsafe { transmute(0x6666666666666666_u64) };

    const OPRND1_32: f32 = unsafe { transmute(0x33333333_u32) };
    const OPRND2_32: f32 = unsafe { transmute(0x55555555_u32) };

    const AND_32: f32 = unsafe { transmute(0x11111111_u32) };
    const ANDN_32: f32 = unsafe { transmute(0x44444444_u32) };
    const OR_32: f32 = unsafe { transmute(0x77777777_u32) };
    const XOR_32: f32 = unsafe { transmute(0x66666666_u32) };

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_and_pd() {
        let a = _mm_set1_pd(OPRND1_64);
        let b = _mm_set1_pd(OPRND2_64);
        let src = _mm_set_pd(1., 2.);
        let r = _mm_mask_and_pd(src, 0b01, a, b);
        let e = _mm_set_pd(1., AND_64);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_and_pd() {
        let a = _mm_set1_pd(OPRND1_64);
        let b = _mm_set1_pd(OPRND2_64);
        let r = _mm_maskz_and_pd(0b01, a, b);
        let e = _mm_set_pd(0.0, AND_64);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_and_pd() {
        let a = _mm256_set1_pd(OPRND1_64);
        let b = _mm256_set1_pd(OPRND2_64);
        let src = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_mask_and_pd(src, 0b0101, a, b);
        let e = _mm256_set_pd(1., AND_64, 3., AND_64);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_and_pd() {
        let a = _mm256_set1_pd(OPRND1_64);
        let b = _mm256_set1_pd(OPRND2_64);
        let r = _mm256_maskz_and_pd(0b0101, a, b);
        let e = _mm256_set_pd(0.0, AND_64, 0.0, AND_64);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_and_pd() {
        let a = _mm512_set1_pd(OPRND1_64);
        let b = _mm512_set1_pd(OPRND2_64);
        let r = _mm512_and_pd(a, b);
        let e = _mm512_set1_pd(AND_64);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_and_pd() {
        let a = _mm512_set1_pd(OPRND1_64);
        let b = _mm512_set1_pd(OPRND2_64);
        let src = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_mask_and_pd(src, 0b01010101, a, b);
        let e = _mm512_set_pd(1., AND_64, 3., AND_64, 5., AND_64, 7., AND_64);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_and_pd() {
        let a = _mm512_set1_pd(OPRND1_64);
        let b = _mm512_set1_pd(OPRND2_64);
        let r = _mm512_maskz_and_pd(0b01010101, a, b);
        let e = _mm512_set_pd(0.0, AND_64, 0.0, AND_64, 0.0, AND_64, 0.0, AND_64);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_and_ps() {
        let a = _mm_set1_ps(OPRND1_32);
        let b = _mm_set1_ps(OPRND2_32);
        let src = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm_mask_and_ps(src, 0b0101, a, b);
        let e = _mm_set_ps(1., AND_32, 3., AND_32);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_and_ps() {
        let a = _mm_set1_ps(OPRND1_32);
        let b = _mm_set1_ps(OPRND2_32);
        let r = _mm_maskz_and_ps(0b0101, a, b);
        let e = _mm_set_ps(0.0, AND_32, 0.0, AND_32);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_and_ps() {
        let a = _mm256_set1_ps(OPRND1_32);
        let b = _mm256_set1_ps(OPRND2_32);
        let src = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_mask_and_ps(src, 0b01010101, a, b);
        let e = _mm256_set_ps(1., AND_32, 3., AND_32, 5., AND_32, 7., AND_32);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_and_ps() {
        let a = _mm256_set1_ps(OPRND1_32);
        let b = _mm256_set1_ps(OPRND2_32);
        let r = _mm256_maskz_and_ps(0b01010101, a, b);
        let e = _mm256_set_ps(0.0, AND_32, 0.0, AND_32, 0.0, AND_32, 0.0, AND_32);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_and_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let r = _mm512_and_ps(a, b);
        let e = _mm512_set1_ps(AND_32);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_and_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let src = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let r = _mm512_mask_and_ps(src, 0b0101010101010101, a, b);
        let e = _mm512_set_ps(
            1., AND_32, 3., AND_32, 5., AND_32, 7., AND_32, 9., AND_32, 11., AND_32, 13., AND_32,
            15., AND_32,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_and_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let r = _mm512_maskz_and_ps(0b0101010101010101, a, b);
        let e = _mm512_set_ps(
            0., AND_32, 0., AND_32, 0., AND_32, 0., AND_32, 0., AND_32, 0., AND_32, 0., AND_32, 0.,
            AND_32,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_andnot_pd() {
        let a = _mm_set1_pd(OPRND1_64);
        let b = _mm_set1_pd(OPRND2_64);
        let src = _mm_set_pd(1., 2.);
        let r = _mm_mask_andnot_pd(src, 0b01, a, b);
        let e = _mm_set_pd(1., ANDN_64);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_andnot_pd() {
        let a = _mm_set1_pd(OPRND1_64);
        let b = _mm_set1_pd(OPRND2_64);
        let r = _mm_maskz_andnot_pd(0b01, a, b);
        let e = _mm_set_pd(0.0, ANDN_64);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_andnot_pd() {
        let a = _mm256_set1_pd(OPRND1_64);
        let b = _mm256_set1_pd(OPRND2_64);
        let src = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_mask_andnot_pd(src, 0b0101, a, b);
        let e = _mm256_set_pd(1., ANDN_64, 3., ANDN_64);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_andnot_pd() {
        let a = _mm256_set1_pd(OPRND1_64);
        let b = _mm256_set1_pd(OPRND2_64);
        let r = _mm256_maskz_andnot_pd(0b0101, a, b);
        let e = _mm256_set_pd(0.0, ANDN_64, 0.0, ANDN_64);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_andnot_pd() {
        let a = _mm512_set1_pd(OPRND1_64);
        let b = _mm512_set1_pd(OPRND2_64);
        let r = _mm512_andnot_pd(a, b);
        let e = _mm512_set1_pd(ANDN_64);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_andnot_pd() {
        let a = _mm512_set1_pd(OPRND1_64);
        let b = _mm512_set1_pd(OPRND2_64);
        let src = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_mask_andnot_pd(src, 0b01010101, a, b);
        let e = _mm512_set_pd(1., ANDN_64, 3., ANDN_64, 5., ANDN_64, 7., ANDN_64);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_andnot_pd() {
        let a = _mm512_set1_pd(OPRND1_64);
        let b = _mm512_set1_pd(OPRND2_64);
        let r = _mm512_maskz_andnot_pd(0b01010101, a, b);
        let e = _mm512_set_pd(0.0, ANDN_64, 0.0, ANDN_64, 0.0, ANDN_64, 0.0, ANDN_64);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_andnot_ps() {
        let a = _mm_set1_ps(OPRND1_32);
        let b = _mm_set1_ps(OPRND2_32);
        let src = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm_mask_andnot_ps(src, 0b0101, a, b);
        let e = _mm_set_ps(1., ANDN_32, 3., ANDN_32);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_andnot_ps() {
        let a = _mm_set1_ps(OPRND1_32);
        let b = _mm_set1_ps(OPRND2_32);
        let r = _mm_maskz_andnot_ps(0b0101, a, b);
        let e = _mm_set_ps(0.0, ANDN_32, 0.0, ANDN_32);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_andnot_ps() {
        let a = _mm256_set1_ps(OPRND1_32);
        let b = _mm256_set1_ps(OPRND2_32);
        let src = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_mask_andnot_ps(src, 0b01010101, a, b);
        let e = _mm256_set_ps(1., ANDN_32, 3., ANDN_32, 5., ANDN_32, 7., ANDN_32);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_andnot_ps() {
        let a = _mm256_set1_ps(OPRND1_32);
        let b = _mm256_set1_ps(OPRND2_32);
        let r = _mm256_maskz_andnot_ps(0b01010101, a, b);
        let e = _mm256_set_ps(0.0, ANDN_32, 0.0, ANDN_32, 0.0, ANDN_32, 0.0, ANDN_32);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_andnot_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let r = _mm512_andnot_ps(a, b);
        let e = _mm512_set1_ps(ANDN_32);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_andnot_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let src = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let r = _mm512_mask_andnot_ps(src, 0b0101010101010101, a, b);
        let e = _mm512_set_ps(
            1., ANDN_32, 3., ANDN_32, 5., ANDN_32, 7., ANDN_32, 9., ANDN_32, 11., ANDN_32, 13.,
            ANDN_32, 15., ANDN_32,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_andnot_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let r = _mm512_maskz_andnot_ps(0b0101010101010101, a, b);
        let e = _mm512_set_ps(
            0., ANDN_32, 0., ANDN_32, 0., ANDN_32, 0., ANDN_32, 0., ANDN_32, 0., ANDN_32, 0.,
            ANDN_32, 0., ANDN_32,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_or_pd() {
        let a = _mm_set1_pd(OPRND1_64);
        let b = _mm_set1_pd(OPRND2_64);
        let src = _mm_set_pd(1., 2.);
        let r = _mm_mask_or_pd(src, 0b01, a, b);
        let e = _mm_set_pd(1., OR_64);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_or_pd() {
        let a = _mm_set1_pd(OPRND1_64);
        let b = _mm_set1_pd(OPRND2_64);
        let r = _mm_maskz_or_pd(0b01, a, b);
        let e = _mm_set_pd(0.0, OR_64);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_or_pd() {
        let a = _mm256_set1_pd(OPRND1_64);
        let b = _mm256_set1_pd(OPRND2_64);
        let src = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_mask_or_pd(src, 0b0101, a, b);
        let e = _mm256_set_pd(1., OR_64, 3., OR_64);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_or_pd() {
        let a = _mm256_set1_pd(OPRND1_64);
        let b = _mm256_set1_pd(OPRND2_64);
        let r = _mm256_maskz_or_pd(0b0101, a, b);
        let e = _mm256_set_pd(0.0, OR_64, 0.0, OR_64);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_or_pd() {
        let a = _mm512_set1_pd(OPRND1_64);
        let b = _mm512_set1_pd(OPRND2_64);
        let r = _mm512_or_pd(a, b);
        let e = _mm512_set1_pd(OR_64);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_or_pd() {
        let a = _mm512_set1_pd(OPRND1_64);
        let b = _mm512_set1_pd(OPRND2_64);
        let src = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_mask_or_pd(src, 0b01010101, a, b);
        let e = _mm512_set_pd(1., OR_64, 3., OR_64, 5., OR_64, 7., OR_64);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_or_pd() {
        let a = _mm512_set1_pd(OPRND1_64);
        let b = _mm512_set1_pd(OPRND2_64);
        let r = _mm512_maskz_or_pd(0b01010101, a, b);
        let e = _mm512_set_pd(0.0, OR_64, 0.0, OR_64, 0.0, OR_64, 0.0, OR_64);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_or_ps() {
        let a = _mm_set1_ps(OPRND1_32);
        let b = _mm_set1_ps(OPRND2_32);
        let src = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm_mask_or_ps(src, 0b0101, a, b);
        let e = _mm_set_ps(1., OR_32, 3., OR_32);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_or_ps() {
        let a = _mm_set1_ps(OPRND1_32);
        let b = _mm_set1_ps(OPRND2_32);
        let r = _mm_maskz_or_ps(0b0101, a, b);
        let e = _mm_set_ps(0.0, OR_32, 0.0, OR_32);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_or_ps() {
        let a = _mm256_set1_ps(OPRND1_32);
        let b = _mm256_set1_ps(OPRND2_32);
        let src = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_mask_or_ps(src, 0b01010101, a, b);
        let e = _mm256_set_ps(1., OR_32, 3., OR_32, 5., OR_32, 7., OR_32);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_or_ps() {
        let a = _mm256_set1_ps(OPRND1_32);
        let b = _mm256_set1_ps(OPRND2_32);
        let r = _mm256_maskz_or_ps(0b01010101, a, b);
        let e = _mm256_set_ps(0.0, OR_32, 0.0, OR_32, 0.0, OR_32, 0.0, OR_32);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_or_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let r = _mm512_or_ps(a, b);
        let e = _mm512_set1_ps(OR_32);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_or_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let src = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let r = _mm512_mask_or_ps(src, 0b0101010101010101, a, b);
        let e = _mm512_set_ps(
            1., OR_32, 3., OR_32, 5., OR_32, 7., OR_32, 9., OR_32, 11., OR_32, 13., OR_32, 15.,
            OR_32,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_or_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let r = _mm512_maskz_or_ps(0b0101010101010101, a, b);
        let e = _mm512_set_ps(
            0., OR_32, 0., OR_32, 0., OR_32, 0., OR_32, 0., OR_32, 0., OR_32, 0., OR_32, 0., OR_32,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_xor_pd() {
        let a = _mm_set1_pd(OPRND1_64);
        let b = _mm_set1_pd(OPRND2_64);
        let src = _mm_set_pd(1., 2.);
        let r = _mm_mask_xor_pd(src, 0b01, a, b);
        let e = _mm_set_pd(1., XOR_64);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_xor_pd() {
        let a = _mm_set1_pd(OPRND1_64);
        let b = _mm_set1_pd(OPRND2_64);
        let r = _mm_maskz_xor_pd(0b01, a, b);
        let e = _mm_set_pd(0.0, XOR_64);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_xor_pd() {
        let a = _mm256_set1_pd(OPRND1_64);
        let b = _mm256_set1_pd(OPRND2_64);
        let src = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_mask_xor_pd(src, 0b0101, a, b);
        let e = _mm256_set_pd(1., XOR_64, 3., XOR_64);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_xor_pd() {
        let a = _mm256_set1_pd(OPRND1_64);
        let b = _mm256_set1_pd(OPRND2_64);
        let r = _mm256_maskz_xor_pd(0b0101, a, b);
        let e = _mm256_set_pd(0.0, XOR_64, 0.0, XOR_64);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_xor_pd() {
        let a = _mm512_set1_pd(OPRND1_64);
        let b = _mm512_set1_pd(OPRND2_64);
        let r = _mm512_xor_pd(a, b);
        let e = _mm512_set1_pd(XOR_64);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_xor_pd() {
        let a = _mm512_set1_pd(OPRND1_64);
        let b = _mm512_set1_pd(OPRND2_64);
        let src = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_mask_xor_pd(src, 0b01010101, a, b);
        let e = _mm512_set_pd(1., XOR_64, 3., XOR_64, 5., XOR_64, 7., XOR_64);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_xor_pd() {
        let a = _mm512_set1_pd(OPRND1_64);
        let b = _mm512_set1_pd(OPRND2_64);
        let r = _mm512_maskz_xor_pd(0b01010101, a, b);
        let e = _mm512_set_pd(0.0, XOR_64, 0.0, XOR_64, 0.0, XOR_64, 0.0, XOR_64);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_xor_ps() {
        let a = _mm_set1_ps(OPRND1_32);
        let b = _mm_set1_ps(OPRND2_32);
        let src = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm_mask_xor_ps(src, 0b0101, a, b);
        let e = _mm_set_ps(1., XOR_32, 3., XOR_32);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_xor_ps() {
        let a = _mm_set1_ps(OPRND1_32);
        let b = _mm_set1_ps(OPRND2_32);
        let r = _mm_maskz_xor_ps(0b0101, a, b);
        let e = _mm_set_ps(0.0, XOR_32, 0.0, XOR_32);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_xor_ps() {
        let a = _mm256_set1_ps(OPRND1_32);
        let b = _mm256_set1_ps(OPRND2_32);
        let src = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_mask_xor_ps(src, 0b01010101, a, b);
        let e = _mm256_set_ps(1., XOR_32, 3., XOR_32, 5., XOR_32, 7., XOR_32);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_xor_ps() {
        let a = _mm256_set1_ps(OPRND1_32);
        let b = _mm256_set1_ps(OPRND2_32);
        let r = _mm256_maskz_xor_ps(0b01010101, a, b);
        let e = _mm256_set_ps(0.0, XOR_32, 0.0, XOR_32, 0.0, XOR_32, 0.0, XOR_32);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_xor_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let r = _mm512_xor_ps(a, b);
        let e = _mm512_set1_ps(XOR_32);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_xor_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let src = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let r = _mm512_mask_xor_ps(src, 0b0101010101010101, a, b);
        let e = _mm512_set_ps(
            1., XOR_32, 3., XOR_32, 5., XOR_32, 7., XOR_32, 9., XOR_32, 11., XOR_32, 13., XOR_32,
            15., XOR_32,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_xor_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let r = _mm512_maskz_xor_ps(0b0101010101010101, a, b);
        let e = _mm512_set_ps(
            0., XOR_32, 0., XOR_32, 0., XOR_32, 0., XOR_32, 0., XOR_32, 0., XOR_32, 0., XOR_32, 0.,
            XOR_32,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_broadcast_f32x2() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm256_broadcast_f32x2(a);
        let e = _mm256_set_ps(1., 2., 1., 2., 1., 2., 1., 2.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_broadcast_f32x2() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm256_set_ps(5., 6., 7., 8., 9., 10., 11., 12.);
        let r = _mm256_mask_broadcast_f32x2(b, 0b01101001, a);
        let e = _mm256_set_ps(5., 2., 1., 8., 1., 10., 11., 2.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_broadcast_f32x2() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm256_maskz_broadcast_f32x2(0b01101001, a);
        let e = _mm256_set_ps(0., 2., 1., 0., 1., 0., 0., 2.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_broadcast_f32x2() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm512_broadcast_f32x2(a);
        let e = _mm512_set_ps(
            1., 2., 1., 2., 1., 2., 1., 2., 1., 2., 1., 2., 1., 2., 1., 2.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_broadcast_f32x2() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm512_set_ps(
            5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
        );
        let r = _mm512_mask_broadcast_f32x2(b, 0b0110100100111100, a);
        let e = _mm512_set_ps(
            5., 2., 1., 8., 1., 10., 11., 2., 13., 14., 1., 2., 1., 2., 19., 20.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_broadcast_f32x2() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm512_maskz_broadcast_f32x2(0b0110100100111100, a);
        let e = _mm512_set_ps(
            0., 2., 1., 0., 1., 0., 0., 2., 0., 0., 1., 2., 1., 2., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_broadcast_f32x8() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_broadcast_f32x8(a);
        let e = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 1., 2., 3., 4., 5., 6., 7., 8.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_broadcast_f32x8() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_ps(
            9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.,
        );
        let r = _mm512_mask_broadcast_f32x8(b, 0b0110100100111100, a);
        let e = _mm512_set_ps(
            9., 2., 3., 12., 5., 14., 15., 8., 17., 18., 3., 4., 5., 6., 23., 24.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_broadcast_f32x8() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_broadcast_f32x8(0b0110100100111100, a);
        let e = _mm512_set_ps(
            0., 2., 3., 0., 5., 0., 0., 8., 0., 0., 3., 4., 5., 6., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_broadcast_f64x2() {
        let a = _mm_set_pd(1., 2.);
        let r = _mm256_broadcast_f64x2(a);
        let e = _mm256_set_pd(1., 2., 1., 2.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_broadcast_f64x2() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm256_set_pd(3., 4., 5., 6.);
        let r = _mm256_mask_broadcast_f64x2(b, 0b0110, a);
        let e = _mm256_set_pd(3., 2., 1., 6.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_broadcast_f64x2() {
        let a = _mm_set_pd(1., 2.);
        let r = _mm256_maskz_broadcast_f64x2(0b0110, a);
        let e = _mm256_set_pd(0., 2., 1., 0.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_broadcast_f64x2() {
        let a = _mm_set_pd(1., 2.);
        let r = _mm512_broadcast_f64x2(a);
        let e = _mm512_set_pd(1., 2., 1., 2., 1., 2., 1., 2.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_broadcast_f64x2() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm512_set_pd(3., 4., 5., 6., 7., 8., 9., 10.);
        let r = _mm512_mask_broadcast_f64x2(b, 0b01101001, a);
        let e = _mm512_set_pd(3., 2., 1., 6., 1., 8., 9., 2.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_broadcast_f64x2() {
        let a = _mm_set_pd(1., 2.);
        let r = _mm512_maskz_broadcast_f64x2(0b01101001, a);
        let e = _mm512_set_pd(0., 2., 1., 0., 1., 0., 0., 2.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let r = _mm_broadcast_i32x2(a);
        let e = _mm_set_epi32(1, 2, 1, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let b = _mm_set_epi32(5, 6, 7, 8);
        let r = _mm_mask_broadcast_i32x2(b, 0b0110, a);
        let e = _mm_set_epi32(5, 2, 1, 6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let r = _mm_maskz_broadcast_i32x2(0b0110, a);
        let e = _mm_set_epi32(0, 2, 1, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let r = _mm256_broadcast_i32x2(a);
        let e = _mm256_set_epi32(1, 2, 1, 2, 1, 2, 1, 2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let b = _mm256_set_epi32(5, 6, 7, 8, 9, 10, 11, 12);
        let r = _mm256_mask_broadcast_i32x2(b, 0b01101001, a);
        let e = _mm256_set_epi32(5, 2, 1, 6, 1, 10, 11, 2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let r = _mm256_maskz_broadcast_i32x2(0b01101001, a);
        let e = _mm256_set_epi32(0, 2, 1, 0, 1, 0, 0, 2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let r = _mm512_broadcast_i32x2(a);
        let e = _mm512_set_epi32(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let b = _mm512_set_epi32(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
        let r = _mm512_mask_broadcast_i32x2(b, 0b0110100100111100, a);
        let e = _mm512_set_epi32(5, 2, 1, 8, 1, 10, 11, 2, 13, 14, 1, 2, 1, 2, 19, 20);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let r = _mm512_maskz_broadcast_i32x2(0b0110100100111100, a);
        let e = _mm512_set_epi32(0, 2, 1, 0, 1, 0, 0, 2, 0, 0, 1, 2, 1, 2, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_broadcast_i32x8() {
        let a = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_broadcast_i32x8(a);
        let e = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_broadcast_i32x8() {
        let a = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm512_set_epi32(
            9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        );
        let r = _mm512_mask_broadcast_i32x8(b, 0b0110100100111100, a);
        let e = _mm512_set_epi32(9, 2, 3, 12, 5, 14, 15, 8, 17, 18, 3, 4, 5, 6, 23, 24);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_broadcast_i32x8() {
        let a = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_maskz_broadcast_i32x8(0b0110100100111100, a);
        let e = _mm512_set_epi32(0, 2, 3, 0, 5, 0, 0, 8, 0, 0, 3, 4, 5, 6, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_broadcast_i64x2() {
        let a = _mm_set_epi64x(1, 2);
        let r = _mm256_broadcast_i64x2(a);
        let e = _mm256_set_epi64x(1, 2, 1, 2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_broadcast_i64x2() {
        let a = _mm_set_epi64x(1, 2);
        let b = _mm256_set_epi64x(3, 4, 5, 6);
        let r = _mm256_mask_broadcast_i64x2(b, 0b0110, a);
        let e = _mm256_set_epi64x(3, 2, 1, 6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_broadcast_i64x2() {
        let a = _mm_set_epi64x(1, 2);
        let r = _mm256_maskz_broadcast_i64x2(0b0110, a);
        let e = _mm256_set_epi64x(0, 2, 1, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_broadcast_i64x2() {
        let a = _mm_set_epi64x(1, 2);
        let r = _mm512_broadcast_i64x2(a);
        let e = _mm512_set_epi64(1, 2, 1, 2, 1, 2, 1, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_broadcast_i64x2() {
        let a = _mm_set_epi64x(1, 2);
        let b = _mm512_set_epi64(3, 4, 5, 6, 7, 8, 9, 10);
        let r = _mm512_mask_broadcast_i64x2(b, 0b01101001, a);
        let e = _mm512_set_epi64(3, 2, 1, 6, 1, 8, 9, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_broadcast_i64x2() {
        let a = _mm_set_epi64x(1, 2);
        let r = _mm512_maskz_broadcast_i64x2(0b01101001, a);
        let e = _mm512_set_epi64(0, 2, 1, 0, 1, 0, 0, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_extractf32x8_ps() {
        let a = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let r = _mm512_extractf32x8_ps::<1>(a);
        let e = _mm256_set_ps(9., 10., 11., 12., 13., 14., 15., 16.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_extractf32x8_ps() {
        let a = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let b = _mm256_set_ps(17., 18., 19., 20., 21., 22., 23., 24.);
        let r = _mm512_mask_extractf32x8_ps::<1>(b, 0b01101001, a);
        let e = _mm256_set_ps(17., 10., 11., 20., 13., 22., 23., 16.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_extractf32x8_ps() {
        let a = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let r = _mm512_maskz_extractf32x8_ps::<1>(0b01101001, a);
        let e = _mm256_set_ps(0., 10., 11., 0., 13., 0., 0., 16.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_extractf64x2_pd() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_extractf64x2_pd::<1>(a);
        let e = _mm_set_pd(3., 4.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_extractf64x2_pd() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm_set_pd(5., 6.);
        let r = _mm256_mask_extractf64x2_pd::<1>(b, 0b01, a);
        let e = _mm_set_pd(5., 4.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_extractf64x2_pd() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_maskz_extractf64x2_pd::<1>(0b01, a);
        let e = _mm_set_pd(0., 4.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_extractf64x2_pd() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_extractf64x2_pd::<2>(a);
        let e = _mm_set_pd(5., 6.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_extractf64x2_pd() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm_set_pd(9., 10.);
        let r = _mm512_mask_extractf64x2_pd::<2>(b, 0b01, a);
        let e = _mm_set_pd(9., 6.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_extractf64x2_pd() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_extractf64x2_pd::<2>(0b01, a);
        let e = _mm_set_pd(0., 6.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_extracti32x8_epi32() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_extracti32x8_epi32::<1>(a);
        let e = _mm256_set_epi32(9, 10, 11, 12, 13, 14, 15, 16);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_extracti32x8_epi32() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm256_set_epi32(17, 18, 19, 20, 21, 22, 23, 24);
        let r = _mm512_mask_extracti32x8_epi32::<1>(b, 0b01101001, a);
        let e = _mm256_set_epi32(17, 10, 11, 20, 13, 22, 23, 16);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_extracti32x8_epi32() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_maskz_extracti32x8_epi32::<1>(0b01101001, a);
        let e = _mm256_set_epi32(0, 10, 11, 0, 13, 0, 0, 16);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_extracti64x2_epi64() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let r = _mm256_extracti64x2_epi64::<1>(a);
        let e = _mm_set_epi64x(3, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_extracti64x2_epi64() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm_set_epi64x(5, 6);
        let r = _mm256_mask_extracti64x2_epi64::<1>(b, 0b01, a);
        let e = _mm_set_epi64x(5, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_extracti64x2_epi64() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let r = _mm256_maskz_extracti64x2_epi64::<1>(0b01, a);
        let e = _mm_set_epi64x(0, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_extracti64x2_epi64() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_extracti64x2_epi64::<2>(a);
        let e = _mm_set_epi64x(5, 6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_extracti64x2_epi64() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_set_epi64x(9, 10);
        let r = _mm512_mask_extracti64x2_epi64::<2>(b, 0b01, a);
        let e = _mm_set_epi64x(9, 6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_extracti64x2_epi64() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_maskz_extracti64x2_epi64::<2>(0b01, a);
        let e = _mm_set_epi64x(0, 6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_insertf32x8() {
        let a = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let b = _mm256_set_ps(17., 18., 19., 20., 21., 22., 23., 24.);
        let r = _mm512_insertf32x8::<1>(a, b);
        let e = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 17., 18., 19., 20., 21., 22., 23., 24.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_insertf32x8() {
        let a = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let b = _mm256_set_ps(17., 18., 19., 20., 21., 22., 23., 24.);
        let src = _mm512_set_ps(
            25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40.,
        );
        let r = _mm512_mask_insertf32x8::<1>(src, 0b0110100100111100, a, b);
        let e = _mm512_set_ps(
            25., 2., 3., 28., 5., 30., 31., 8., 33., 34., 19., 20., 21., 22., 39., 40.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_insertf32x8() {
        let a = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let b = _mm256_set_ps(17., 18., 19., 20., 21., 22., 23., 24.);
        let r = _mm512_maskz_insertf32x8::<1>(0b0110100100111100, a, b);
        let e = _mm512_set_ps(
            0., 2., 3., 0., 5., 0., 0., 8., 0., 0., 19., 20., 21., 22., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_insertf64x2() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm_set_pd(5., 6.);
        let r = _mm256_insertf64x2::<1>(a, b);
        let e = _mm256_set_pd(1., 2., 5., 6.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_insertf64x2() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm_set_pd(5., 6.);
        let src = _mm256_set_pd(7., 8., 9., 10.);
        let r = _mm256_mask_insertf64x2::<1>(src, 0b0110, a, b);
        let e = _mm256_set_pd(7., 2., 5., 10.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_insertf64x2() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm_set_pd(5., 6.);
        let r = _mm256_maskz_insertf64x2::<1>(0b0110, a, b);
        let e = _mm256_set_pd(0., 2., 5., 0.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_insertf64x2() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm_set_pd(9., 10.);
        let r = _mm512_insertf64x2::<2>(a, b);
        let e = _mm512_set_pd(1., 2., 3., 4., 9., 10., 7., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_insertf64x2() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm_set_pd(9., 10.);
        let src = _mm512_set_pd(11., 12., 13., 14., 15., 16., 17., 18.);
        let r = _mm512_mask_insertf64x2::<2>(src, 0b01101001, a, b);
        let e = _mm512_set_pd(11., 2., 3., 14., 9., 16., 17., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_insertf64x2() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm_set_pd(9., 10.);
        let r = _mm512_maskz_insertf64x2::<2>(0b01101001, a, b);
        let e = _mm512_set_pd(0., 2., 3., 0., 9., 0., 0., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_inserti32x8() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm256_set_epi32(17, 18, 19, 20, 21, 22, 23, 24);
        let r = _mm512_inserti32x8::<1>(a, b);
        let e = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_inserti32x8() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm256_set_epi32(17, 18, 19, 20, 21, 22, 23, 24);
        let src = _mm512_set_epi32(
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        );
        let r = _mm512_mask_inserti32x8::<1>(src, 0b0110100100111100, a, b);
        let e = _mm512_set_epi32(25, 2, 3, 28, 5, 30, 31, 8, 33, 34, 19, 20, 21, 22, 39, 40);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_inserti32x8() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm256_set_epi32(17, 18, 19, 20, 21, 22, 23, 24);
        let r = _mm512_maskz_inserti32x8::<1>(0b0110100100111100, a, b);
        let e = _mm512_set_epi32(0, 2, 3, 0, 5, 0, 0, 8, 0, 0, 19, 20, 21, 22, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_inserti64x2() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm_set_epi64x(5, 6);
        let r = _mm256_inserti64x2::<1>(a, b);
        let e = _mm256_set_epi64x(1, 2, 5, 6);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_inserti64x2() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm_set_epi64x(5, 6);
        let src = _mm256_set_epi64x(7, 8, 9, 10);
        let r = _mm256_mask_inserti64x2::<1>(src, 0b0110, a, b);
        let e = _mm256_set_epi64x(7, 2, 5, 10);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_inserti64x2() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm_set_epi64x(5, 6);
        let r = _mm256_maskz_inserti64x2::<1>(0b0110, a, b);
        let e = _mm256_set_epi64x(0, 2, 5, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_inserti64x2() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_set_epi64x(9, 10);
        let r = _mm512_inserti64x2::<2>(a, b);
        let e = _mm512_set_epi64(1, 2, 3, 4, 9, 10, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_inserti64x2() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_set_epi64x(9, 10);
        let src = _mm512_set_epi64(11, 12, 13, 14, 15, 16, 17, 18);
        let r = _mm512_mask_inserti64x2::<2>(src, 0b01101001, a, b);
        let e = _mm512_set_epi64(11, 2, 3, 14, 9, 16, 17, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_inserti64x2() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_set_epi64x(9, 10);
        let r = _mm512_maskz_inserti64x2::<2>(0b01101001, a, b);
        let e = _mm512_set_epi64(0, 2, 3, 0, 9, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvt_roundepi64_pd() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_cvt_roundepi64_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a);
        let e = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvt_roundepi64_pd() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm512_set_pd(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm512_mask_cvt_roundepi64_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            b, 0b01101001, a,
        );
        let e = _mm512_set_pd(9., 2., 3., 12., 5., 14., 15., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvt_roundepi64_pd() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_maskz_cvt_roundepi64_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            0b01101001, a,
        );
        let e = _mm512_set_pd(0., 2., 3., 0., 5., 0., 0., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_cvtepi64_pd() {
        let a = _mm_set_epi64x(1, 2);
        let r = _mm_cvtepi64_pd(a);
        let e = _mm_set_pd(1., 2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_cvtepi64_pd() {
        let a = _mm_set_epi64x(1, 2);
        let b = _mm_set_pd(3., 4.);
        let r = _mm_mask_cvtepi64_pd(b, 0b01, a);
        let e = _mm_set_pd(3., 2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_cvtepi64_pd() {
        let a = _mm_set_epi64x(1, 2);
        let r = _mm_maskz_cvtepi64_pd(0b01, a);
        let e = _mm_set_pd(0., 2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_cvtepi64_pd() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let r = _mm256_cvtepi64_pd(a);
        let e = _mm256_set_pd(1., 2., 3., 4.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_cvtepi64_pd() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm256_set_pd(5., 6., 7., 8.);
        let r = _mm256_mask_cvtepi64_pd(b, 0b0110, a);
        let e = _mm256_set_pd(5., 2., 3., 8.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_cvtepi64_pd() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let r = _mm256_maskz_cvtepi64_pd(0b0110, a);
        let e = _mm256_set_pd(0., 2., 3., 0.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvtepi64_pd() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_cvtepi64_pd(a);
        let e = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvtepi64_pd() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm512_set_pd(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm512_mask_cvtepi64_pd(b, 0b01101001, a);
        let e = _mm512_set_pd(9., 2., 3., 12., 5., 14., 15., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvtepi64_pd() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_maskz_cvtepi64_pd(0b01101001, a);
        let e = _mm512_set_pd(0., 2., 3., 0., 5., 0., 0., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvt_roundepi64_ps() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_cvt_roundepi64_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a);
        let e = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvt_roundepi64_ps() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm256_set_ps(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm512_mask_cvt_roundepi64_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            b, 0b01101001, a,
        );
        let e = _mm256_set_ps(9., 2., 3., 12., 5., 14., 15., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvt_roundepi64_ps() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_maskz_cvt_roundepi64_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            0b01101001, a,
        );
        let e = _mm256_set_ps(0., 2., 3., 0., 5., 0., 0., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_cvtepi64_ps() {
        let a = _mm_set_epi64x(1, 2);
        let r = _mm_cvtepi64_ps(a);
        let e = _mm_set_ps(0., 0., 1., 2.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_cvtepi64_ps() {
        let a = _mm_set_epi64x(1, 2);
        let b = _mm_set_ps(3., 4., 5., 6.);
        let r = _mm_mask_cvtepi64_ps(b, 0b01, a);
        let e = _mm_set_ps(0., 0., 5., 2.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_cvtepi64_ps() {
        let a = _mm_set_epi64x(1, 2);
        let r = _mm_maskz_cvtepi64_ps(0b01, a);
        let e = _mm_set_ps(0., 0., 0., 2.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_cvtepi64_ps() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let r = _mm256_cvtepi64_ps(a);
        let e = _mm_set_ps(1., 2., 3., 4.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_cvtepi64_ps() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm_set_ps(5., 6., 7., 8.);
        let r = _mm256_mask_cvtepi64_ps(b, 0b0110, a);
        let e = _mm_set_ps(5., 2., 3., 8.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_cvtepi64_ps() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let r = _mm256_maskz_cvtepi64_ps(0b0110, a);
        let e = _mm_set_ps(0., 2., 3., 0.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvtepi64_ps() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_cvtepi64_ps(a);
        let e = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvtepi64_ps() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm256_set_ps(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm512_mask_cvtepi64_ps(b, 0b01101001, a);
        let e = _mm256_set_ps(9., 2., 3., 12., 5., 14., 15., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvtepi64_ps() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_maskz_cvtepi64_ps(0b01101001, a);
        let e = _mm256_set_ps(0., 2., 3., 0., 5., 0., 0., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvt_roundepu64_pd() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_cvt_roundepu64_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a);
        let e = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvt_roundepu64_pd() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm512_set_pd(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm512_mask_cvt_roundepu64_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            b, 0b01101001, a,
        );
        let e = _mm512_set_pd(9., 2., 3., 12., 5., 14., 15., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvt_roundepu64_pd() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_maskz_cvt_roundepu64_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            0b01101001, a,
        );
        let e = _mm512_set_pd(0., 2., 3., 0., 5., 0., 0., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_cvtepu64_pd() {
        let a = _mm_set_epi64x(1, 2);
        let r = _mm_cvtepu64_pd(a);
        let e = _mm_set_pd(1., 2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_cvtepu64_pd() {
        let a = _mm_set_epi64x(1, 2);
        let b = _mm_set_pd(3., 4.);
        let r = _mm_mask_cvtepu64_pd(b, 0b01, a);
        let e = _mm_set_pd(3., 2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_cvtepu64_pd() {
        let a = _mm_set_epi64x(1, 2);
        let r = _mm_maskz_cvtepu64_pd(0b01, a);
        let e = _mm_set_pd(0., 2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_cvtepu64_pd() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let r = _mm256_cvtepu64_pd(a);
        let e = _mm256_set_pd(1., 2., 3., 4.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_cvtepu64_pd() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm256_set_pd(5., 6., 7., 8.);
        let r = _mm256_mask_cvtepu64_pd(b, 0b0110, a);
        let e = _mm256_set_pd(5., 2., 3., 8.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_cvtepu64_pd() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let r = _mm256_maskz_cvtepu64_pd(0b0110, a);
        let e = _mm256_set_pd(0., 2., 3., 0.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvtepu64_pd() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_cvtepu64_pd(a);
        let e = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvtepu64_pd() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm512_set_pd(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm512_mask_cvtepu64_pd(b, 0b01101001, a);
        let e = _mm512_set_pd(9., 2., 3., 12., 5., 14., 15., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvtepu64_pd() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_maskz_cvtepu64_pd(0b01101001, a);
        let e = _mm512_set_pd(0., 2., 3., 0., 5., 0., 0., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvt_roundepu64_ps() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_cvt_roundepu64_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a);
        let e = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvt_roundepu64_ps() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm256_set_ps(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm512_mask_cvt_roundepu64_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            b, 0b01101001, a,
        );
        let e = _mm256_set_ps(9., 2., 3., 12., 5., 14., 15., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvt_roundepu64_ps() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_maskz_cvt_roundepu64_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            0b01101001, a,
        );
        let e = _mm256_set_ps(0., 2., 3., 0., 5., 0., 0., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_cvtepu64_ps() {
        let a = _mm_set_epi64x(1, 2);
        let r = _mm_cvtepu64_ps(a);
        let e = _mm_set_ps(0., 0., 1., 2.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_cvtepu64_ps() {
        let a = _mm_set_epi64x(1, 2);
        let b = _mm_set_ps(3., 4., 5., 6.);
        let r = _mm_mask_cvtepu64_ps(b, 0b01, a);
        let e = _mm_set_ps(0., 0., 5., 2.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_cvtepu64_ps() {
        let a = _mm_set_epi64x(1, 2);
        let r = _mm_maskz_cvtepu64_ps(0b01, a);
        let e = _mm_set_ps(0., 0., 0., 2.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_cvtepu64_ps() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let r = _mm256_cvtepu64_ps(a);
        let e = _mm_set_ps(1., 2., 3., 4.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_cvtepu64_ps() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm_set_ps(5., 6., 7., 8.);
        let r = _mm256_mask_cvtepu64_ps(b, 0b0110, a);
        let e = _mm_set_ps(5., 2., 3., 8.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_cvtepu64_ps() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let r = _mm256_maskz_cvtepu64_ps(0b0110, a);
        let e = _mm_set_ps(0., 2., 3., 0.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvtepu64_ps() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_cvtepu64_ps(a);
        let e = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvtepu64_ps() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm256_set_ps(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm512_mask_cvtepu64_ps(b, 0b01101001, a);
        let e = _mm256_set_ps(9., 2., 3., 12., 5., 14., 15., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvtepu64_ps() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_maskz_cvtepu64_ps(0b01101001, a);
        let e = _mm256_set_ps(0., 2., 3., 0., 5., 0., 0., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvt_roundpd_epi64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvt_roundpd_epi64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvt_roundpd_epi64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvt_roundpd_epi64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            b, 0b01101001, a,
        );
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvt_roundpd_epi64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvt_roundpd_epi64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            0b01101001, a,
        );
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_cvtpd_epi64() {
        let a = _mm_set_pd(1., 2.);
        let r = _mm_cvtpd_epi64(a);
        let e = _mm_set_epi64x(1, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_cvtpd_epi64() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm_set_epi64x(3, 4);
        let r = _mm_mask_cvtpd_epi64(b, 0b01, a);
        let e = _mm_set_epi64x(3, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_cvtpd_epi64() {
        let a = _mm_set_pd(1., 2.);
        let r = _mm_maskz_cvtpd_epi64(0b01, a);
        let e = _mm_set_epi64x(0, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_cvtpd_epi64() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_cvtpd_epi64(a);
        let e = _mm256_set_epi64x(1, 2, 3, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_cvtpd_epi64() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm256_set_epi64x(5, 6, 7, 8);
        let r = _mm256_mask_cvtpd_epi64(b, 0b0110, a);
        let e = _mm256_set_epi64x(5, 2, 3, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_cvtpd_epi64() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_maskz_cvtpd_epi64(0b0110, a);
        let e = _mm256_set_epi64x(0, 2, 3, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvtpd_epi64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvtpd_epi64(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvtpd_epi64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvtpd_epi64(b, 0b01101001, a);
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvtpd_epi64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvtpd_epi64(0b01101001, a);
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvt_roundps_epi64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvt_roundps_epi64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvt_roundps_epi64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvt_roundps_epi64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            b, 0b01101001, a,
        );
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvt_roundps_epi64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvt_roundps_epi64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            0b01101001, a,
        );
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_cvtps_epi64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm_cvtps_epi64(a);
        let e = _mm_set_epi64x(3, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_cvtps_epi64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm_set_epi64x(5, 6);
        let r = _mm_mask_cvtps_epi64(b, 0b01, a);
        let e = _mm_set_epi64x(5, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_cvtps_epi64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm_maskz_cvtps_epi64(0b01, a);
        let e = _mm_set_epi64x(0, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_cvtps_epi64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm256_cvtps_epi64(a);
        let e = _mm256_set_epi64x(1, 2, 3, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_cvtps_epi64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm256_set_epi64x(5, 6, 7, 8);
        let r = _mm256_mask_cvtps_epi64(b, 0b0110, a);
        let e = _mm256_set_epi64x(5, 2, 3, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_cvtps_epi64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm256_maskz_cvtps_epi64(0b0110, a);
        let e = _mm256_set_epi64x(0, 2, 3, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvtps_epi64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvtps_epi64(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvtps_epi64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvtps_epi64(b, 0b01101001, a);
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvtps_epi64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvtps_epi64(0b01101001, a);
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvt_roundpd_epu64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvt_roundpd_epu64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvt_roundpd_epu64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvt_roundpd_epu64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            b, 0b01101001, a,
        );
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvt_roundpd_epu64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvt_roundpd_epu64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            0b01101001, a,
        );
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_cvtpd_epu64() {
        let a = _mm_set_pd(1., 2.);
        let r = _mm_cvtpd_epu64(a);
        let e = _mm_set_epi64x(1, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_cvtpd_epu64() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm_set_epi64x(3, 4);
        let r = _mm_mask_cvtpd_epu64(b, 0b01, a);
        let e = _mm_set_epi64x(3, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_cvtpd_epu64() {
        let a = _mm_set_pd(1., 2.);
        let r = _mm_maskz_cvtpd_epu64(0b01, a);
        let e = _mm_set_epi64x(0, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_cvtpd_epu64() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_cvtpd_epu64(a);
        let e = _mm256_set_epi64x(1, 2, 3, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_cvtpd_epu64() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm256_set_epi64x(5, 6, 7, 8);
        let r = _mm256_mask_cvtpd_epu64(b, 0b0110, a);
        let e = _mm256_set_epi64x(5, 2, 3, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_cvtpd_epu64() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_maskz_cvtpd_epu64(0b0110, a);
        let e = _mm256_set_epi64x(0, 2, 3, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvtpd_epu64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvtpd_epu64(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvtpd_epu64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvtpd_epu64(b, 0b01101001, a);
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvtpd_epu64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvtpd_epu64(0b01101001, a);
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvt_roundps_epu64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvt_roundps_epu64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvt_roundps_epu64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvt_roundps_epu64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            b, 0b01101001, a,
        );
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvt_roundps_epu64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvt_roundps_epu64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            0b01101001, a,
        );
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_cvtps_epu64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm_cvtps_epu64(a);
        let e = _mm_set_epi64x(3, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_cvtps_epu64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm_set_epi64x(5, 6);
        let r = _mm_mask_cvtps_epu64(b, 0b01, a);
        let e = _mm_set_epi64x(5, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_cvtps_epu64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm_maskz_cvtps_epu64(0b01, a);
        let e = _mm_set_epi64x(0, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_cvtps_epu64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm256_cvtps_epu64(a);
        let e = _mm256_set_epi64x(1, 2, 3, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_cvtps_epu64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm256_set_epi64x(5, 6, 7, 8);
        let r = _mm256_mask_cvtps_epu64(b, 0b0110, a);
        let e = _mm256_set_epi64x(5, 2, 3, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_cvtps_epu64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm256_maskz_cvtps_epu64(0b0110, a);
        let e = _mm256_set_epi64x(0, 2, 3, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvtps_epu64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvtps_epu64(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvtps_epu64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvtps_epu64(b, 0b01101001, a);
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvtps_epu64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvtps_epu64(0b01101001, a);
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvtt_roundpd_epi64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvtt_roundpd_epi64::<_MM_FROUND_NO_EXC>(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvtt_roundpd_epi64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvtt_roundpd_epi64::<_MM_FROUND_NO_EXC>(b, 0b01101001, a);
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvtt_roundpd_epi64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvtt_roundpd_epi64::<_MM_FROUND_NO_EXC>(0b01101001, a);
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_cvttpd_epi64() {
        let a = _mm_set_pd(1., 2.);
        let r = _mm_cvttpd_epi64(a);
        let e = _mm_set_epi64x(1, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_cvttpd_epi64() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm_set_epi64x(3, 4);
        let r = _mm_mask_cvttpd_epi64(b, 0b01, a);
        let e = _mm_set_epi64x(3, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_cvttpd_epi64() {
        let a = _mm_set_pd(1., 2.);
        let r = _mm_maskz_cvttpd_epi64(0b01, a);
        let e = _mm_set_epi64x(0, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_cvttpd_epi64() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_cvttpd_epi64(a);
        let e = _mm256_set_epi64x(1, 2, 3, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_cvttpd_epi64() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm256_set_epi64x(5, 6, 7, 8);
        let r = _mm256_mask_cvttpd_epi64(b, 0b0110, a);
        let e = _mm256_set_epi64x(5, 2, 3, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_cvttpd_epi64() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_maskz_cvttpd_epi64(0b0110, a);
        let e = _mm256_set_epi64x(0, 2, 3, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvttpd_epi64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvttpd_epi64(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvttpd_epi64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvttpd_epi64(b, 0b01101001, a);
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvttpd_epi64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvttpd_epi64(0b01101001, a);
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvtt_roundps_epi64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvtt_roundps_epi64::<_MM_FROUND_NO_EXC>(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvtt_roundps_epi64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvtt_roundps_epi64::<_MM_FROUND_NO_EXC>(b, 0b01101001, a);
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvtt_roundps_epi64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvtt_roundps_epi64::<_MM_FROUND_NO_EXC>(0b01101001, a);
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_cvttps_epi64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm_cvttps_epi64(a);
        let e = _mm_set_epi64x(3, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_cvttps_epi64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm_set_epi64x(5, 6);
        let r = _mm_mask_cvttps_epi64(b, 0b01, a);
        let e = _mm_set_epi64x(5, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_cvttps_epi64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm_maskz_cvttps_epi64(0b01, a);
        let e = _mm_set_epi64x(0, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_cvttps_epi64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm256_cvttps_epi64(a);
        let e = _mm256_set_epi64x(1, 2, 3, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_cvttps_epi64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm256_set_epi64x(5, 6, 7, 8);
        let r = _mm256_mask_cvttps_epi64(b, 0b0110, a);
        let e = _mm256_set_epi64x(5, 2, 3, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_cvttps_epi64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm256_maskz_cvttps_epi64(0b0110, a);
        let e = _mm256_set_epi64x(0, 2, 3, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvttps_epi64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvttps_epi64(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvttps_epi64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvttps_epi64(b, 0b01101001, a);
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvttps_epi64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvttps_epi64(0b01101001, a);
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvtt_roundpd_epu64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvtt_roundpd_epu64::<_MM_FROUND_NO_EXC>(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvtt_roundpd_epu64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvtt_roundpd_epu64::<_MM_FROUND_NO_EXC>(b, 0b01101001, a);
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvtt_roundpd_epu64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvtt_roundpd_epu64::<_MM_FROUND_NO_EXC>(0b01101001, a);
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_cvttpd_epu64() {
        let a = _mm_set_pd(1., 2.);
        let r = _mm_cvttpd_epu64(a);
        let e = _mm_set_epi64x(1, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_cvttpd_epu64() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm_set_epi64x(3, 4);
        let r = _mm_mask_cvttpd_epu64(b, 0b01, a);
        let e = _mm_set_epi64x(3, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_cvttpd_epu64() {
        let a = _mm_set_pd(1., 2.);
        let r = _mm_maskz_cvttpd_epu64(0b01, a);
        let e = _mm_set_epi64x(0, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_cvttpd_epu64() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_cvttpd_epu64(a);
        let e = _mm256_set_epi64x(1, 2, 3, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_cvttpd_epu64() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm256_set_epi64x(5, 6, 7, 8);
        let r = _mm256_mask_cvttpd_epu64(b, 0b0110, a);
        let e = _mm256_set_epi64x(5, 2, 3, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_cvttpd_epu64() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_maskz_cvttpd_epu64(0b0110, a);
        let e = _mm256_set_epi64x(0, 2, 3, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvttpd_epu64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvttpd_epu64(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvttpd_epu64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvttpd_epu64(b, 0b01101001, a);
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvttpd_epu64() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvttpd_epu64(0b01101001, a);
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvtt_roundps_epu64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvtt_roundps_epu64::<_MM_FROUND_NO_EXC>(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvtt_roundps_epu64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvtt_roundps_epu64::<_MM_FROUND_NO_EXC>(b, 0b01101001, a);
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvtt_roundps_epu64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvtt_roundps_epu64::<_MM_FROUND_NO_EXC>(0b01101001, a);
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_cvttps_epu64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm_cvttps_epu64(a);
        let e = _mm_set_epi64x(3, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_cvttps_epu64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm_set_epi64x(5, 6);
        let r = _mm_mask_cvttps_epu64(b, 0b01, a);
        let e = _mm_set_epi64x(5, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_cvttps_epu64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm_maskz_cvttps_epu64(0b01, a);
        let e = _mm_set_epi64x(0, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_cvttps_epu64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm256_cvttps_epu64(a);
        let e = _mm256_set_epi64x(1, 2, 3, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_cvttps_epu64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm256_set_epi64x(5, 6, 7, 8);
        let r = _mm256_mask_cvttps_epu64(b, 0b0110, a);
        let e = _mm256_set_epi64x(5, 2, 3, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_cvttps_epu64() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm256_maskz_cvttps_epu64(0b0110, a);
        let e = _mm256_set_epi64x(0, 2, 3, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_cvttps_epu64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_cvttps_epu64(a);
        let e = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_cvttps_epu64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mask_cvttps_epu64(b, 0b01101001, a);
        let e = _mm512_set_epi64(9, 2, 3, 12, 5, 14, 15, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_cvttps_epu64() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_cvttps_epu64(0b01101001, a);
        let e = _mm512_set_epi64(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m512i(r, e);
    }
}
