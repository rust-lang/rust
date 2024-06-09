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
#[cfg_attr(test, assert_instr(vandps))] // FIXME: should be `vandpd` instruction.
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
#[cfg_attr(test, assert_instr(vandps))] // FIXME: should be `vandpd` instruction.
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
#[cfg_attr(test, assert_instr(vandps))] // FIXME: should be `vandpd` instruction.
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
#[cfg_attr(test, assert_instr(vandps))] // FIXME: should be `vandpd` instruction.
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
#[cfg_attr(test, assert_instr(vandps))] // FIXME: should be `vandpd` instruction.
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
#[cfg_attr(test, assert_instr(vandps))] // FIXME: should be `vandpd` instruction.
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
#[cfg_attr(test, assert_instr(vandnps))] // FIXME: should be `vandnpd` instruction.
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
#[cfg_attr(test, assert_instr(vandnps))] // FIXME: should be `vandnpd` instruction.
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
#[cfg_attr(test, assert_instr(vandnps))] // FIXME: should be `vandnpd` instruction.
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
#[cfg_attr(test, assert_instr(vandnps))] // FIXME: should be `vandnpd` instruction.
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
#[cfg_attr(test, assert_instr(vandnps))] // FIXME: should be `vandnpd` instruction.
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
#[cfg_attr(test, assert_instr(vandnps))] // FIXME: should be `vandnpd` instruction.
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
#[cfg_attr(test, assert_instr(vorps))] // FIXME: should be `vorpd` instruction.
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
#[cfg_attr(test, assert_instr(vorps))] // FIXME: should be `vorpd` instruction.
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
#[cfg_attr(test, assert_instr(vorps))] // FIXME: should be `vorpd` instruction.
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
#[cfg_attr(test, assert_instr(vorps))] // FIXME: should be `vorpd` instruction.
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
#[cfg_attr(test, assert_instr(vorps))] // FIXME: should be `vorpd` instruction.
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
#[cfg_attr(test, assert_instr(vorps))] // FIXME: should be `vorpd` instruction.
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
#[cfg_attr(test, assert_instr(vxorps))] // FIXME: should be `vxorpd` instruction.
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
#[cfg_attr(test, assert_instr(vxorps))] // FIXME: should be `vxorpd` instruction.
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
#[cfg_attr(test, assert_instr(vxorps))] // FIXME: should be `vxorpd` instruction.
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
#[cfg_attr(test, assert_instr(vxorps))] // FIXME: should be `vxorpd` instruction.
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
#[cfg_attr(test, assert_instr(vxorps))] // FIXME: should be `vxorpd` instruction.
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
#[cfg_attr(test, assert_instr(vxorps))] // FIXME: should be `vxorpd` instruction.
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
pub unsafe fn _mm512_maskz_xor_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    let xor = _mm512_xor_ps(a, b).as_f32x16();
    let zero = _mm512_setzero_ps().as_f32x16();
    transmute(simd_select_bitmask(k, xor, zero))
}

#[cfg(test)]
mod tests {
    use super::*;

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;
    use crate::core_arch::x86_64::*;

    const OPRND1_64: f64 = f64::from_bits(0x3333333333333333);
    const OPRND2_64: f64 = f64::from_bits(0x5555555555555555);

    const AND_64: f64 = f64::from_bits(0x1111111111111111);
    const ANDN_64: f64 = f64::from_bits(0x4444444444444444);
    const OR_64: f64 = f64::from_bits(0x7777777777777777);
    const XOR_64: f64 = f64::from_bits(0x6666666666666666);

    const OPRND1_32: f32 = f32::from_bits(0x33333333);
    const OPRND2_32: f32 = f32::from_bits(0x55555555);

    const AND_32: f32 = f32::from_bits(0x11111111);
    const ANDN_32: f32 = f32::from_bits(0x44444444);
    const OR_32: f32 = f32::from_bits(0x77777777);
    const XOR_32: f32 = f32::from_bits(0x66666666);

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
        let src = _mm512_set_ps(1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm512_mask_and_ps(src, 0b0101010101010101, a, b);
        let e = _mm512_set_ps(1., AND_32, 3., AND_32, 5., AND_32, 7., AND_32, 9., AND_32, 11., AND_32, 13., AND_32, 15., AND_32);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_and_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let r = _mm512_maskz_and_ps(0b0101010101010101, a, b);
        let e = _mm512_set_ps(0., AND_32, 0., AND_32, 0., AND_32, 0., AND_32, 0., AND_32, 0., AND_32, 0., AND_32, 0., AND_32);
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
        let src = _mm512_set_ps(1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm512_mask_andnot_ps(src, 0b0101010101010101, a, b);
        let e = _mm512_set_ps(1., ANDN_32, 3., ANDN_32, 5., ANDN_32, 7., ANDN_32, 9., ANDN_32, 11., ANDN_32, 13., ANDN_32, 15., ANDN_32);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_andnot_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let r = _mm512_maskz_andnot_ps(0b0101010101010101, a, b);
        let e = _mm512_set_ps(0., ANDN_32, 0., ANDN_32, 0., ANDN_32, 0., ANDN_32, 0., ANDN_32, 0., ANDN_32, 0., ANDN_32, 0., ANDN_32);
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
        let src = _mm512_set_ps(1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm512_mask_or_ps(src, 0b0101010101010101, a, b);
        let e = _mm512_set_ps(1., OR_32, 3., OR_32, 5., OR_32, 7., OR_32, 9., OR_32, 11., OR_32, 13., OR_32, 15., OR_32);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_or_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let r = _mm512_maskz_or_ps(0b0101010101010101, a, b);
        let e = _mm512_set_ps(0., OR_32, 0., OR_32, 0., OR_32, 0., OR_32, 0., OR_32, 0., OR_32, 0., OR_32, 0., OR_32);
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
        let src = _mm512_set_ps(1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm512_mask_xor_ps(src, 0b0101010101010101, a, b);
        let e = _mm512_set_ps(1., XOR_32, 3., XOR_32, 5., XOR_32, 7., XOR_32, 9., XOR_32, 11., XOR_32, 13., XOR_32, 15., XOR_32);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_xor_ps() {
        let a = _mm512_set1_ps(OPRND1_32);
        let b = _mm512_set1_ps(OPRND2_32);
        let r = _mm512_maskz_xor_ps(0b0101010101010101, a, b);
        let e = _mm512_set_ps(0., XOR_32, 0., XOR_32, 0., XOR_32, 0., XOR_32, 0., XOR_32, 0., XOR_32, 0., XOR_32, 0., XOR_32);
        assert_eq_m512(r, e);
    }


}
