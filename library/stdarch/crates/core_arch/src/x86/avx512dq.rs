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
pub fn _mm_mask_and_pd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let and = _mm_and_pd(a, b).as_f64x2();
        transmute(simd_select_bitmask(k, and, src.as_f64x2()))
    }
}

/// Compute the bitwise AND of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_and_pd&ig_expand=289)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_and_pd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let and = _mm_and_pd(a, b).as_f64x2();
        transmute(simd_select_bitmask(k, and, f64x2::ZERO))
    }
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
pub fn _mm256_mask_and_pd(src: __m256d, k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    unsafe {
        let and = _mm256_and_pd(a, b).as_f64x4();
        transmute(simd_select_bitmask(k, and, src.as_f64x4()))
    }
}

/// Compute the bitwise AND of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_and_pd&ig_expand=292)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_and_pd(k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    unsafe {
        let and = _mm256_and_pd(a, b).as_f64x4();
        transmute(simd_select_bitmask(k, and, f64x4::ZERO))
    }
}

/// Compute the bitwise AND of packed double-precision (64-bit) floating point numbers in a and b
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_and_pd&ig_expand=293)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandp))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_and_pd(a: __m512d, b: __m512d) -> __m512d {
    unsafe { transmute(simd_and(transmute::<_, u64x8>(a), transmute::<_, u64x8>(b))) }
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
pub fn _mm512_mask_and_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    unsafe {
        let and = _mm512_and_pd(a, b).as_f64x8();
        transmute(simd_select_bitmask(k, and, src.as_f64x8()))
    }
}

/// Compute the bitwise AND of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_and_pd&ig_expand=295)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_and_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    unsafe {
        let and = _mm512_and_pd(a, b).as_f64x8();
        transmute(simd_select_bitmask(k, and, f64x8::ZERO))
    }
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
pub fn _mm_mask_and_ps(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    unsafe {
        let and = _mm_and_ps(a, b).as_f32x4();
        transmute(simd_select_bitmask(k, and, src.as_f32x4()))
    }
}

/// Compute the bitwise AND of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_and_ps&ig_expand=298)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_and_ps(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    unsafe {
        let and = _mm_and_ps(a, b).as_f32x4();
        transmute(simd_select_bitmask(k, and, f32x4::ZERO))
    }
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
pub fn _mm256_mask_and_ps(src: __m256, k: __mmask8, a: __m256, b: __m256) -> __m256 {
    unsafe {
        let and = _mm256_and_ps(a, b).as_f32x8();
        transmute(simd_select_bitmask(k, and, src.as_f32x8()))
    }
}

/// Compute the bitwise AND of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_and_ps&ig_expand=301)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vandps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_and_ps(k: __mmask8, a: __m256, b: __m256) -> __m256 {
    unsafe {
        let and = _mm256_and_ps(a, b).as_f32x8();
        transmute(simd_select_bitmask(k, and, f32x8::ZERO))
    }
}

/// Compute the bitwise AND of packed single-precision (32-bit) floating point numbers in a and b
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_and_ps&ig_expand=303)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_and_ps(a: __m512, b: __m512) -> __m512 {
    unsafe {
        transmute(simd_and(
            transmute::<_, u32x16>(a),
            transmute::<_, u32x16>(b),
        ))
    }
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
pub fn _mm512_mask_and_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    unsafe {
        let and = _mm512_and_ps(a, b).as_f32x16();
        transmute(simd_select_bitmask(k, and, src.as_f32x16()))
    }
}

/// Compute the bitwise AND of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_and_ps&ig_expand=305)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_and_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    unsafe {
        let and = _mm512_and_ps(a, b).as_f32x16();
        transmute(simd_select_bitmask(k, and, f32x16::ZERO))
    }
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
pub fn _mm_mask_andnot_pd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let andnot = _mm_andnot_pd(a, b).as_f64x2();
        transmute(simd_select_bitmask(k, andnot, src.as_f64x2()))
    }
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
pub fn _mm_maskz_andnot_pd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let andnot = _mm_andnot_pd(a, b).as_f64x2();
        transmute(simd_select_bitmask(k, andnot, f64x2::ZERO))
    }
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
pub fn _mm256_mask_andnot_pd(src: __m256d, k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    unsafe {
        let andnot = _mm256_andnot_pd(a, b).as_f64x4();
        transmute(simd_select_bitmask(k, andnot, src.as_f64x4()))
    }
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
pub fn _mm256_maskz_andnot_pd(k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    unsafe {
        let andnot = _mm256_andnot_pd(a, b).as_f64x4();
        transmute(simd_select_bitmask(k, andnot, f64x4::ZERO))
    }
}

/// Compute the bitwise NOT of packed double-precision (64-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_andnot_pd&ig_expand=331)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnp))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_andnot_pd(a: __m512d, b: __m512d) -> __m512d {
    unsafe { _mm512_and_pd(_mm512_xor_pd(a, transmute(_mm512_set1_epi64(-1))), b) }
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
pub fn _mm512_mask_andnot_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    unsafe {
        let andnot = _mm512_andnot_pd(a, b).as_f64x8();
        transmute(simd_select_bitmask(k, andnot, src.as_f64x8()))
    }
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
pub fn _mm512_maskz_andnot_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    unsafe {
        let andnot = _mm512_andnot_pd(a, b).as_f64x8();
        transmute(simd_select_bitmask(k, andnot, f64x8::ZERO))
    }
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
pub fn _mm_mask_andnot_ps(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    unsafe {
        let andnot = _mm_andnot_ps(a, b).as_f32x4();
        transmute(simd_select_bitmask(k, andnot, src.as_f32x4()))
    }
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
pub fn _mm_maskz_andnot_ps(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    unsafe {
        let andnot = _mm_andnot_ps(a, b).as_f32x4();
        transmute(simd_select_bitmask(k, andnot, f32x4::ZERO))
    }
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
pub fn _mm256_mask_andnot_ps(src: __m256, k: __mmask8, a: __m256, b: __m256) -> __m256 {
    unsafe {
        let andnot = _mm256_andnot_ps(a, b).as_f32x8();
        transmute(simd_select_bitmask(k, andnot, src.as_f32x8()))
    }
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
pub fn _mm256_maskz_andnot_ps(k: __mmask8, a: __m256, b: __m256) -> __m256 {
    unsafe {
        let andnot = _mm256_andnot_ps(a, b).as_f32x8();
        transmute(simd_select_bitmask(k, andnot, f32x8::ZERO))
    }
}

/// Compute the bitwise NOT of packed single-precision (32-bit) floating point numbers in a and then
/// bitwise AND with b and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_andnot_ps&ig_expand=340)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_andnot_ps(a: __m512, b: __m512) -> __m512 {
    unsafe { _mm512_and_ps(_mm512_xor_ps(a, transmute(_mm512_set1_epi32(-1))), b) }
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
pub fn _mm512_mask_andnot_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    unsafe {
        let andnot = _mm512_andnot_ps(a, b).as_f32x16();
        transmute(simd_select_bitmask(k, andnot, src.as_f32x16()))
    }
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
pub fn _mm512_maskz_andnot_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    unsafe {
        let andnot = _mm512_andnot_ps(a, b).as_f32x16();
        transmute(simd_select_bitmask(k, andnot, f32x16::ZERO))
    }
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
pub fn _mm_mask_or_pd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let or = _mm_or_pd(a, b).as_f64x2();
        transmute(simd_select_bitmask(k, or, src.as_f64x2()))
    }
}

/// Compute the bitwise OR of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_or_pd&ig_expand=4825)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_or_pd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let or = _mm_or_pd(a, b).as_f64x2();
        transmute(simd_select_bitmask(k, or, f64x2::ZERO))
    }
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
pub fn _mm256_mask_or_pd(src: __m256d, k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    unsafe {
        let or = _mm256_or_pd(a, b).as_f64x4();
        transmute(simd_select_bitmask(k, or, src.as_f64x4()))
    }
}

/// Compute the bitwise OR of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_or_pd&ig_expand=4828)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_or_pd(k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    unsafe {
        let or = _mm256_or_pd(a, b).as_f64x4();
        transmute(simd_select_bitmask(k, or, f64x4::ZERO))
    }
}

/// Compute the bitwise OR of packed double-precision (64-bit) floating point numbers in a and b
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_or_pd&ig_expand=4829)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorp))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_or_pd(a: __m512d, b: __m512d) -> __m512d {
    unsafe { transmute(simd_or(transmute::<_, u64x8>(a), transmute::<_, u64x8>(b))) }
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
pub fn _mm512_mask_or_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    unsafe {
        let or = _mm512_or_pd(a, b).as_f64x8();
        transmute(simd_select_bitmask(k, or, src.as_f64x8()))
    }
}

/// Compute the bitwise OR of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_or_pd&ig_expand=4831)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_or_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    unsafe {
        let or = _mm512_or_pd(a, b).as_f64x8();
        transmute(simd_select_bitmask(k, or, f64x8::ZERO))
    }
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
pub fn _mm_mask_or_ps(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    unsafe {
        let or = _mm_or_ps(a, b).as_f32x4();
        transmute(simd_select_bitmask(k, or, src.as_f32x4()))
    }
}

/// Compute the bitwise OR of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_or_ps&ig_expand=4834)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_or_ps(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    unsafe {
        let or = _mm_or_ps(a, b).as_f32x4();
        transmute(simd_select_bitmask(k, or, f32x4::ZERO))
    }
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
pub fn _mm256_mask_or_ps(src: __m256, k: __mmask8, a: __m256, b: __m256) -> __m256 {
    unsafe {
        let or = _mm256_or_ps(a, b).as_f32x8();
        transmute(simd_select_bitmask(k, or, src.as_f32x8()))
    }
}

/// Compute the bitwise OR of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_or_ps&ig_expand=4837)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_or_ps(k: __mmask8, a: __m256, b: __m256) -> __m256 {
    unsafe {
        let or = _mm256_or_ps(a, b).as_f32x8();
        transmute(simd_select_bitmask(k, or, f32x8::ZERO))
    }
}

/// Compute the bitwise OR of packed single-precision (32-bit) floating point numbers in a and b
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_or_ps&ig_expand=4838)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_or_ps(a: __m512, b: __m512) -> __m512 {
    unsafe {
        transmute(simd_or(
            transmute::<_, u32x16>(a),
            transmute::<_, u32x16>(b),
        ))
    }
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
pub fn _mm512_mask_or_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    unsafe {
        let or = _mm512_or_ps(a, b).as_f32x16();
        transmute(simd_select_bitmask(k, or, src.as_f32x16()))
    }
}

/// Compute the bitwise OR of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_or_ps&ig_expand=4840)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_or_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    unsafe {
        let or = _mm512_or_ps(a, b).as_f32x16();
        transmute(simd_select_bitmask(k, or, f32x16::ZERO))
    }
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
pub fn _mm_mask_xor_pd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let xor = _mm_xor_pd(a, b).as_f64x2();
        transmute(simd_select_bitmask(k, xor, src.as_f64x2()))
    }
}

/// Compute the bitwise XOR of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_xor_pd&ig_expand=7095)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_xor_pd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let xor = _mm_xor_pd(a, b).as_f64x2();
        transmute(simd_select_bitmask(k, xor, f64x2::ZERO))
    }
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
pub fn _mm256_mask_xor_pd(src: __m256d, k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    unsafe {
        let xor = _mm256_xor_pd(a, b).as_f64x4();
        transmute(simd_select_bitmask(k, xor, src.as_f64x4()))
    }
}

/// Compute the bitwise XOR of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_xor_pd&ig_expand=7098)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_xor_pd(k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    unsafe {
        let xor = _mm256_xor_pd(a, b).as_f64x4();
        transmute(simd_select_bitmask(k, xor, f64x4::ZERO))
    }
}

/// Compute the bitwise XOR of packed double-precision (64-bit) floating point numbers in a and b
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_xor_pd&ig_expand=7102)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorp))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_xor_pd(a: __m512d, b: __m512d) -> __m512d {
    unsafe { transmute(simd_xor(transmute::<_, u64x8>(a), transmute::<_, u64x8>(b))) }
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
pub fn _mm512_mask_xor_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    unsafe {
        let xor = _mm512_xor_pd(a, b).as_f64x8();
        transmute(simd_select_bitmask(k, xor, src.as_f64x8()))
    }
}

/// Compute the bitwise XOR of packed double-precision (64-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_xor_pd&ig_expand=7101)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorpd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_xor_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    unsafe {
        let xor = _mm512_xor_pd(a, b).as_f64x8();
        transmute(simd_select_bitmask(k, xor, f64x8::ZERO))
    }
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
pub fn _mm_mask_xor_ps(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    unsafe {
        let xor = _mm_xor_ps(a, b).as_f32x4();
        transmute(simd_select_bitmask(k, xor, src.as_f32x4()))
    }
}

/// Compute the bitwise XOR of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_xor_ps&ig_expand=7104)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_xor_ps(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    unsafe {
        let xor = _mm_xor_ps(a, b).as_f32x4();
        transmute(simd_select_bitmask(k, xor, f32x4::ZERO))
    }
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
pub fn _mm256_mask_xor_ps(src: __m256, k: __mmask8, a: __m256, b: __m256) -> __m256 {
    unsafe {
        let xor = _mm256_xor_ps(a, b).as_f32x8();
        transmute(simd_select_bitmask(k, xor, src.as_f32x8()))
    }
}

/// Compute the bitwise XOR of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_xor_ps&ig_expand=7107)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vxorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_xor_ps(k: __mmask8, a: __m256, b: __m256) -> __m256 {
    unsafe {
        let xor = _mm256_xor_ps(a, b).as_f32x8();
        transmute(simd_select_bitmask(k, xor, f32x8::ZERO))
    }
}

/// Compute the bitwise XOR of packed single-precision (32-bit) floating point numbers in a and b
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_xor_ps&ig_expand=7111)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_xor_ps(a: __m512, b: __m512) -> __m512 {
    unsafe {
        transmute(simd_xor(
            transmute::<_, u32x16>(a),
            transmute::<_, u32x16>(b),
        ))
    }
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
pub fn _mm512_mask_xor_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    unsafe {
        let xor = _mm512_xor_ps(a, b).as_f32x16();
        transmute(simd_select_bitmask(k, xor, src.as_f32x16()))
    }
}

/// Compute the bitwise XOR of packed single-precision (32-bit) floating point numbers in a and b and
/// store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_xor_ps&ig_expand=7110)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_xor_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    unsafe {
        let xor = _mm512_xor_ps(a, b).as_f32x16();
        transmute(simd_select_bitmask(k, xor, f32x16::ZERO))
    }
}

// Broadcast

/// Broadcasts the lower 2 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_f32x2&ig_expand=509)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_broadcast_f32x2(a: __m128) -> __m256 {
    unsafe {
        let b: f32x8 = simd_shuffle!(a, a, [0, 1, 0, 1, 0, 1, 0, 1]);
        transmute(b)
    }
}

/// Broadcasts the lower 2 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst using writemask k (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_broadcast_f32x2&ig_expand=510)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vbroadcastf32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_broadcast_f32x2(src: __m256, k: __mmask8, a: __m128) -> __m256 {
    unsafe {
        let b = _mm256_broadcast_f32x2(a).as_f32x8();
        transmute(simd_select_bitmask(k, b, src.as_f32x8()))
    }
}

/// Broadcasts the lower 2 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_broadcast_f32x2&ig_expand=511)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vbroadcastf32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_broadcast_f32x2(k: __mmask8, a: __m128) -> __m256 {
    unsafe {
        let b = _mm256_broadcast_f32x2(a).as_f32x8();
        transmute(simd_select_bitmask(k, b, f32x8::ZERO))
    }
}

/// Broadcasts the lower 2 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcast_f32x2&ig_expand=512)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_broadcast_f32x2(a: __m128) -> __m512 {
    unsafe {
        let b: f32x16 = simd_shuffle!(a, a, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
        transmute(b)
    }
}

/// Broadcasts the lower 2 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst using writemask k (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_broadcast_f32x2&ig_expand=513)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vbroadcastf32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_broadcast_f32x2(src: __m512, k: __mmask16, a: __m128) -> __m512 {
    unsafe {
        let b = _mm512_broadcast_f32x2(a).as_f32x16();
        transmute(simd_select_bitmask(k, b, src.as_f32x16()))
    }
}

/// Broadcasts the lower 2 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_broadcast_f32x2&ig_expand=514)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vbroadcastf32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_broadcast_f32x2(k: __mmask16, a: __m128) -> __m512 {
    unsafe {
        let b = _mm512_broadcast_f32x2(a).as_f32x16();
        transmute(simd_select_bitmask(k, b, f32x16::ZERO))
    }
}

/// Broadcasts the 8 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcast_f32x8&ig_expand=521)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_broadcast_f32x8(a: __m256) -> __m512 {
    unsafe {
        let b: f32x16 = simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
        transmute(b)
    }
}

/// Broadcasts the 8 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst using writemask k (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_broadcast_f32x8&ig_expand=522)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_broadcast_f32x8(src: __m512, k: __mmask16, a: __m256) -> __m512 {
    unsafe {
        let b = _mm512_broadcast_f32x8(a).as_f32x16();
        transmute(simd_select_bitmask(k, b, src.as_f32x16()))
    }
}

/// Broadcasts the 8 packed single-precision (32-bit) floating-point elements from a to all
/// elements of dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_broadcast_f32x8&ig_expand=523)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_broadcast_f32x8(k: __mmask16, a: __m256) -> __m512 {
    unsafe {
        let b = _mm512_broadcast_f32x8(a).as_f32x16();
        transmute(simd_select_bitmask(k, b, f32x16::ZERO))
    }
}

/// Broadcasts the 2 packed double-precision (64-bit) floating-point elements from a to all
/// elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_f64x2&ig_expand=524)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_broadcast_f64x2(a: __m128d) -> __m256d {
    unsafe {
        let b: f64x4 = simd_shuffle!(a, a, [0, 1, 0, 1]);
        transmute(b)
    }
}

/// Broadcasts the 2 packed double-precision (64-bit) floating-point elements from a to all
/// elements of dst using writemask k (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_broadcast_f64x2&ig_expand=525)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_broadcast_f64x2(src: __m256d, k: __mmask8, a: __m128d) -> __m256d {
    unsafe {
        let b = _mm256_broadcast_f64x2(a).as_f64x4();
        transmute(simd_select_bitmask(k, b, src.as_f64x4()))
    }
}

/// Broadcasts the 2 packed double-precision (64-bit) floating-point elements from a to all
/// elements of dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_broadcast_f64x2&ig_expand=526)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_broadcast_f64x2(k: __mmask8, a: __m128d) -> __m256d {
    unsafe {
        let b = _mm256_broadcast_f64x2(a).as_f64x4();
        transmute(simd_select_bitmask(k, b, f64x4::ZERO))
    }
}

/// Broadcasts the 2 packed double-precision (64-bit) floating-point elements from a to all
/// elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcast_f64x2&ig_expand=527)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_broadcast_f64x2(a: __m128d) -> __m512d {
    unsafe {
        let b: f64x8 = simd_shuffle!(a, a, [0, 1, 0, 1, 0, 1, 0, 1]);
        transmute(b)
    }
}

/// Broadcasts the 2 packed double-precision (64-bit) floating-point elements from a to all
/// elements of dst using writemask k (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_broadcast_f64x2&ig_expand=528)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_broadcast_f64x2(src: __m512d, k: __mmask8, a: __m128d) -> __m512d {
    unsafe {
        let b = _mm512_broadcast_f64x2(a).as_f64x8();
        transmute(simd_select_bitmask(k, b, src.as_f64x8()))
    }
}

/// Broadcasts the 2 packed double-precision (64-bit) floating-point elements from a to all
/// elements of dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_broadcast_f64x2&ig_expand=529)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_broadcast_f64x2(k: __mmask8, a: __m128d) -> __m512d {
    unsafe {
        let b = _mm512_broadcast_f64x2(a).as_f64x8();
        transmute(simd_select_bitmask(k, b, f64x8::ZERO))
    }
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_broadcast_i32x2&ig_expand=533)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_broadcast_i32x2(a: __m128i) -> __m128i {
    unsafe {
        let a = a.as_i32x4();
        let b: i32x4 = simd_shuffle!(a, a, [0, 1, 0, 1]);
        transmute(b)
    }
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_broadcast_i32x2&ig_expand=534)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vbroadcasti32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_broadcast_i32x2(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let b = _mm_broadcast_i32x2(a).as_i32x4();
        transmute(simd_select_bitmask(k, b, src.as_i32x4()))
    }
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_broadcast_i32x2&ig_expand=535)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vbroadcasti32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_broadcast_i32x2(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let b = _mm_broadcast_i32x2(a).as_i32x4();
        transmute(simd_select_bitmask(k, b, i32x4::ZERO))
    }
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_i32x2&ig_expand=536)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_broadcast_i32x2(a: __m128i) -> __m256i {
    unsafe {
        let a = a.as_i32x4();
        let b: i32x8 = simd_shuffle!(a, a, [0, 1, 0, 1, 0, 1, 0, 1]);
        transmute(b)
    }
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_broadcast_i32x2&ig_expand=537)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vbroadcasti32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_broadcast_i32x2(src: __m256i, k: __mmask8, a: __m128i) -> __m256i {
    unsafe {
        let b = _mm256_broadcast_i32x2(a).as_i32x8();
        transmute(simd_select_bitmask(k, b, src.as_i32x8()))
    }
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_broadcast_i32x2&ig_expand=538)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vbroadcasti32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_broadcast_i32x2(k: __mmask8, a: __m128i) -> __m256i {
    unsafe {
        let b = _mm256_broadcast_i32x2(a).as_i32x8();
        transmute(simd_select_bitmask(k, b, i32x8::ZERO))
    }
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcast_i32x2&ig_expand=539)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_broadcast_i32x2(a: __m128i) -> __m512i {
    unsafe {
        let a = a.as_i32x4();
        let b: i32x16 = simd_shuffle!(a, a, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
        transmute(b)
    }
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_broadcast_i32x2&ig_expand=540)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vbroadcasti32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_broadcast_i32x2(src: __m512i, k: __mmask16, a: __m128i) -> __m512i {
    unsafe {
        let b = _mm512_broadcast_i32x2(a).as_i32x16();
        transmute(simd_select_bitmask(k, b, src.as_i32x16()))
    }
}

/// Broadcasts the lower 2 packed 32-bit integers from a to all elements of dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_broadcast_i32x2&ig_expand=541)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vbroadcasti32x2))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_broadcast_i32x2(k: __mmask16, a: __m128i) -> __m512i {
    unsafe {
        let b = _mm512_broadcast_i32x2(a).as_i32x16();
        transmute(simd_select_bitmask(k, b, i32x16::ZERO))
    }
}

/// Broadcasts the 8 packed 32-bit integers from a to all elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcast_i32x8&ig_expand=548)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_broadcast_i32x8(a: __m256i) -> __m512i {
    unsafe {
        let a = a.as_i32x8();
        let b: i32x16 = simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
        transmute(b)
    }
}

/// Broadcasts the 8 packed 32-bit integers from a to all elements of dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_broadcast_i32x8&ig_expand=549)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_broadcast_i32x8(src: __m512i, k: __mmask16, a: __m256i) -> __m512i {
    unsafe {
        let b = _mm512_broadcast_i32x8(a).as_i32x16();
        transmute(simd_select_bitmask(k, b, src.as_i32x16()))
    }
}

/// Broadcasts the 8 packed 32-bit integers from a to all elements of dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_broadcast_i32x8&ig_expand=550)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_broadcast_i32x8(k: __mmask16, a: __m256i) -> __m512i {
    unsafe {
        let b = _mm512_broadcast_i32x8(a).as_i32x16();
        transmute(simd_select_bitmask(k, b, i32x16::ZERO))
    }
}

/// Broadcasts the 2 packed 64-bit integers from a to all elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_i64x2&ig_expand=551)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_broadcast_i64x2(a: __m128i) -> __m256i {
    unsafe {
        let a = a.as_i64x2();
        let b: i64x4 = simd_shuffle!(a, a, [0, 1, 0, 1]);
        transmute(b)
    }
}

/// Broadcasts the 2 packed 64-bit integers from a to all elements of dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_broadcast_i64x2&ig_expand=552)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_broadcast_i64x2(src: __m256i, k: __mmask8, a: __m128i) -> __m256i {
    unsafe {
        let b = _mm256_broadcast_i64x2(a).as_i64x4();
        transmute(simd_select_bitmask(k, b, src.as_i64x4()))
    }
}

/// Broadcasts the 2 packed 64-bit integers from a to all elements of dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_broadcast_i64x2&ig_expand=553)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_broadcast_i64x2(k: __mmask8, a: __m128i) -> __m256i {
    unsafe {
        let b = _mm256_broadcast_i64x2(a).as_i64x4();
        transmute(simd_select_bitmask(k, b, i64x4::ZERO))
    }
}

/// Broadcasts the 2 packed 64-bit integers from a to all elements of dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcast_i64x2&ig_expand=554)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_broadcast_i64x2(a: __m128i) -> __m512i {
    unsafe {
        let a = a.as_i64x2();
        let b: i64x8 = simd_shuffle!(a, a, [0, 1, 0, 1, 0, 1, 0, 1]);
        transmute(b)
    }
}

/// Broadcasts the 2 packed 64-bit integers from a to all elements of dst using writemask k
/// (elements are copied from src if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_broadcast_i64x2&ig_expand=555)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_broadcast_i64x2(src: __m512i, k: __mmask8, a: __m128i) -> __m512i {
    unsafe {
        let b = _mm512_broadcast_i64x2(a).as_i64x8();
        transmute(simd_select_bitmask(k, b, src.as_i64x8()))
    }
}

/// Broadcasts the 2 packed 64-bit integers from a to all elements of dst using zeromask k
/// (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_broadcast_i64x2&ig_expand=556)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_broadcast_i64x2(k: __mmask8, a: __m128i) -> __m512i {
    unsafe {
        let b = _mm512_broadcast_i64x2(a).as_i64x8();
        transmute(simd_select_bitmask(k, b, i64x8::ZERO))
    }
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
pub fn _mm512_extractf32x8_ps<const IMM8: i32>(a: __m512) -> __m256 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        match IMM8 & 1 {
            0 => simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7]),
            _ => simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]),
        }
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
pub fn _mm512_mask_extractf32x8_ps<const IMM8: i32>(src: __m256, k: __mmask8, a: __m512) -> __m256 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let b = _mm512_extractf32x8_ps::<IMM8>(a);
        transmute(simd_select_bitmask(k, b.as_f32x8(), src.as_f32x8()))
    }
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
pub fn _mm512_maskz_extractf32x8_ps<const IMM8: i32>(k: __mmask8, a: __m512) -> __m256 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let b = _mm512_extractf32x8_ps::<IMM8>(a);
        transmute(simd_select_bitmask(k, b.as_f32x8(), f32x8::ZERO))
    }
}

/// Extracts 128 bits (composed of 2 packed double-precision (64-bit) floating-point elements) from a,
/// selected with IMM8, and stores the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_extractf64x2_pd&ig_expand=2949)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_extractf64x2_pd<const IMM8: i32>(a: __m256d) -> __m128d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        match IMM8 & 1 {
            0 => simd_shuffle!(a, a, [0, 1]),
            _ => simd_shuffle!(a, a, [2, 3]),
        }
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
pub fn _mm256_mask_extractf64x2_pd<const IMM8: i32>(
    src: __m128d,
    k: __mmask8,
    a: __m256d,
) -> __m128d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let b = _mm256_extractf64x2_pd::<IMM8>(a);
        transmute(simd_select_bitmask(k, b.as_f64x2(), src.as_f64x2()))
    }
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
pub fn _mm256_maskz_extractf64x2_pd<const IMM8: i32>(k: __mmask8, a: __m256d) -> __m128d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let b = _mm256_extractf64x2_pd::<IMM8>(a);
        transmute(simd_select_bitmask(k, b.as_f64x2(), f64x2::ZERO))
    }
}

/// Extracts 128 bits (composed of 2 packed double-precision (64-bit) floating-point elements) from a,
/// selected with IMM8, and stores the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_extractf64x2_pd&ig_expand=2952)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_extractf64x2_pd<const IMM8: i32>(a: __m512d) -> __m128d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 2);
        match IMM8 & 3 {
            0 => simd_shuffle!(a, a, [0, 1]),
            1 => simd_shuffle!(a, a, [2, 3]),
            2 => simd_shuffle!(a, a, [4, 5]),
            _ => simd_shuffle!(a, a, [6, 7]),
        }
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
pub fn _mm512_mask_extractf64x2_pd<const IMM8: i32>(
    src: __m128d,
    k: __mmask8,
    a: __m512d,
) -> __m128d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 2);
        let b = _mm512_extractf64x2_pd::<IMM8>(a).as_f64x2();
        transmute(simd_select_bitmask(k, b, src.as_f64x2()))
    }
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
pub fn _mm512_maskz_extractf64x2_pd<const IMM8: i32>(k: __mmask8, a: __m512d) -> __m128d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 2);
        let b = _mm512_extractf64x2_pd::<IMM8>(a).as_f64x2();
        transmute(simd_select_bitmask(k, b, f64x2::ZERO))
    }
}

/// Extracts 256 bits (composed of 8 packed 32-bit integers) from a, selected with IMM8, and stores
/// the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_extracti32x8_epi32&ig_expand=2965)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_extracti32x8_epi32<const IMM8: i32>(a: __m512i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let a = a.as_i32x16();
        let b: i32x8 = match IMM8 & 1 {
            0 => simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7]),
            _ => simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]),
        };
        transmute(b)
    }
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
pub fn _mm512_mask_extracti32x8_epi32<const IMM8: i32>(
    src: __m256i,
    k: __mmask8,
    a: __m512i,
) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let b = _mm512_extracti32x8_epi32::<IMM8>(a).as_i32x8();
        transmute(simd_select_bitmask(k, b, src.as_i32x8()))
    }
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
pub fn _mm512_maskz_extracti32x8_epi32<const IMM8: i32>(k: __mmask8, a: __m512i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let b = _mm512_extracti32x8_epi32::<IMM8>(a).as_i32x8();
        transmute(simd_select_bitmask(k, b, i32x8::ZERO))
    }
}

/// Extracts 128 bits (composed of 2 packed 64-bit integers) from a, selected with IMM8, and stores
/// the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_extracti64x2_epi64&ig_expand=2968)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_extracti64x2_epi64<const IMM8: i32>(a: __m256i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let a = a.as_i64x4();
        match IMM8 & 1 {
            0 => simd_shuffle!(a, a, [0, 1]),
            _ => simd_shuffle!(a, a, [2, 3]),
        }
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
pub fn _mm256_mask_extracti64x2_epi64<const IMM8: i32>(
    src: __m128i,
    k: __mmask8,
    a: __m256i,
) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let b = _mm256_extracti64x2_epi64::<IMM8>(a).as_i64x2();
        transmute(simd_select_bitmask(k, b, src.as_i64x2()))
    }
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
pub fn _mm256_maskz_extracti64x2_epi64<const IMM8: i32>(k: __mmask8, a: __m256i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let b = _mm256_extracti64x2_epi64::<IMM8>(a).as_i64x2();
        transmute(simd_select_bitmask(k, b, i64x2::ZERO))
    }
}

/// Extracts 128 bits (composed of 2 packed 64-bit integers) from a, selected with IMM8, and stores
/// the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_extracti64x2_epi64&ig_expand=2971)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_extracti64x2_epi64<const IMM8: i32>(a: __m512i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 2);
        let a = a.as_i64x8();
        match IMM8 & 3 {
            0 => simd_shuffle!(a, a, [0, 1]),
            1 => simd_shuffle!(a, a, [2, 3]),
            2 => simd_shuffle!(a, a, [4, 5]),
            _ => simd_shuffle!(a, a, [6, 7]),
        }
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
pub fn _mm512_mask_extracti64x2_epi64<const IMM8: i32>(
    src: __m128i,
    k: __mmask8,
    a: __m512i,
) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 2);
        let b = _mm512_extracti64x2_epi64::<IMM8>(a).as_i64x2();
        transmute(simd_select_bitmask(k, b, src.as_i64x2()))
    }
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
pub fn _mm512_maskz_extracti64x2_epi64<const IMM8: i32>(k: __mmask8, a: __m512i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 2);
        let b = _mm512_extracti64x2_epi64::<IMM8>(a).as_i64x2();
        transmute(simd_select_bitmask(k, b, i64x2::ZERO))
    }
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
pub fn _mm512_insertf32x8<const IMM8: i32>(a: __m512, b: __m256) -> __m512 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let b = _mm512_castps256_ps512(b);
        match IMM8 & 1 {
            0 => {
                simd_shuffle!(
                    a,
                    b,
                    [16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15]
                )
            }
            _ => {
                simd_shuffle!(
                    a,
                    b,
                    [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]
                )
            }
        }
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
pub fn _mm512_mask_insertf32x8<const IMM8: i32>(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m256,
) -> __m512 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let c = _mm512_insertf32x8::<IMM8>(a, b);
        transmute(simd_select_bitmask(k, c.as_f32x16(), src.as_f32x16()))
    }
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
pub fn _mm512_maskz_insertf32x8<const IMM8: i32>(k: __mmask16, a: __m512, b: __m256) -> __m512 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let c = _mm512_insertf32x8::<IMM8>(a, b).as_f32x16();
        transmute(simd_select_bitmask(k, c, f32x16::ZERO))
    }
}

/// Copy a to dst, then insert 128 bits (composed of 2 packed double-precision (64-bit) floating-point
/// elements) from b into dst at the location specified by IMM8.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_insertf64x2&ig_expand=3853)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_insertf64x2<const IMM8: i32>(a: __m256d, b: __m128d) -> __m256d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let b = _mm256_castpd128_pd256(b);
        match IMM8 & 1 {
            0 => simd_shuffle!(a, b, [4, 5, 2, 3]),
            _ => simd_shuffle!(a, b, [0, 1, 4, 5]),
        }
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
pub fn _mm256_mask_insertf64x2<const IMM8: i32>(
    src: __m256d,
    k: __mmask8,
    a: __m256d,
    b: __m128d,
) -> __m256d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let c = _mm256_insertf64x2::<IMM8>(a, b);
        transmute(simd_select_bitmask(k, c.as_f64x4(), src.as_f64x4()))
    }
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
pub fn _mm256_maskz_insertf64x2<const IMM8: i32>(k: __mmask8, a: __m256d, b: __m128d) -> __m256d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let c = _mm256_insertf64x2::<IMM8>(a, b).as_f64x4();
        transmute(simd_select_bitmask(k, c, f64x4::ZERO))
    }
}

/// Copy a to dst, then insert 128 bits (composed of 2 packed double-precision (64-bit) floating-point
/// elements) from b into dst at the location specified by IMM8.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_insertf64x2&ig_expand=3856)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_insertf64x2<const IMM8: i32>(a: __m512d, b: __m128d) -> __m512d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 2);
        let b = _mm512_castpd128_pd512(b);
        match IMM8 & 3 {
            0 => simd_shuffle!(a, b, [8, 9, 2, 3, 4, 5, 6, 7]),
            1 => simd_shuffle!(a, b, [0, 1, 8, 9, 4, 5, 6, 7]),
            2 => simd_shuffle!(a, b, [0, 1, 2, 3, 8, 9, 6, 7]),
            _ => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 8, 9]),
        }
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
pub fn _mm512_mask_insertf64x2<const IMM8: i32>(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m128d,
) -> __m512d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 2);
        let c = _mm512_insertf64x2::<IMM8>(a, b);
        transmute(simd_select_bitmask(k, c.as_f64x8(), src.as_f64x8()))
    }
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
pub fn _mm512_maskz_insertf64x2<const IMM8: i32>(k: __mmask8, a: __m512d, b: __m128d) -> __m512d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 2);
        let c = _mm512_insertf64x2::<IMM8>(a, b).as_f64x8();
        transmute(simd_select_bitmask(k, c, f64x8::ZERO))
    }
}

/// Copy a to dst, then insert 256 bits (composed of 8 packed 32-bit integers) from b into dst at the
/// location specified by IMM8.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_inserti32x8&ig_expand=3869)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_inserti32x8<const IMM8: i32>(a: __m512i, b: __m256i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let a = a.as_i32x16();
        let b = _mm512_castsi256_si512(b).as_i32x16();
        let r: i32x16 = match IMM8 & 1 {
            0 => {
                simd_shuffle!(
                    a,
                    b,
                    [16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15]
                )
            }
            _ => {
                simd_shuffle!(
                    a,
                    b,
                    [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]
                )
            }
        };
        transmute(r)
    }
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
pub fn _mm512_mask_inserti32x8<const IMM8: i32>(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    b: __m256i,
) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let c = _mm512_inserti32x8::<IMM8>(a, b);
        transmute(simd_select_bitmask(k, c.as_i32x16(), src.as_i32x16()))
    }
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
pub fn _mm512_maskz_inserti32x8<const IMM8: i32>(k: __mmask16, a: __m512i, b: __m256i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let c = _mm512_inserti32x8::<IMM8>(a, b).as_i32x16();
        transmute(simd_select_bitmask(k, c, i32x16::ZERO))
    }
}

/// Copy a to dst, then insert 128 bits (composed of 2 packed 64-bit integers) from b into dst at the
/// location specified by IMM8.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_inserti64x2&ig_expand=3872)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_inserti64x2<const IMM8: i32>(a: __m256i, b: __m128i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let a = a.as_i64x4();
        let b = _mm256_castsi128_si256(b).as_i64x4();
        match IMM8 & 1 {
            0 => simd_shuffle!(a, b, [4, 5, 2, 3]),
            _ => simd_shuffle!(a, b, [0, 1, 4, 5]),
        }
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
pub fn _mm256_mask_inserti64x2<const IMM8: i32>(
    src: __m256i,
    k: __mmask8,
    a: __m256i,
    b: __m128i,
) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let c = _mm256_inserti64x2::<IMM8>(a, b);
        transmute(simd_select_bitmask(k, c.as_i64x4(), src.as_i64x4()))
    }
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
pub fn _mm256_maskz_inserti64x2<const IMM8: i32>(k: __mmask8, a: __m256i, b: __m128i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 1);
        let c = _mm256_inserti64x2::<IMM8>(a, b).as_i64x4();
        transmute(simd_select_bitmask(k, c, i64x4::ZERO))
    }
}

/// Copy a to dst, then insert 128 bits (composed of 2 packed 64-bit integers) from b into dst at the
/// location specified by IMM8.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_inserti64x2&ig_expand=3875)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_inserti64x2<const IMM8: i32>(a: __m512i, b: __m128i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 2);
        let a = a.as_i64x8();
        let b = _mm512_castsi128_si512(b).as_i64x8();
        match IMM8 & 3 {
            0 => simd_shuffle!(a, b, [8, 9, 2, 3, 4, 5, 6, 7]),
            1 => simd_shuffle!(a, b, [0, 1, 8, 9, 4, 5, 6, 7]),
            2 => simd_shuffle!(a, b, [0, 1, 2, 3, 8, 9, 6, 7]),
            _ => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 8, 9]),
        }
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
pub fn _mm512_mask_inserti64x2<const IMM8: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m512i,
    b: __m128i,
) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 2);
        let c = _mm512_inserti64x2::<IMM8>(a, b);
        transmute(simd_select_bitmask(k, c.as_i64x8(), src.as_i64x8()))
    }
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
pub fn _mm512_maskz_inserti64x2<const IMM8: i32>(k: __mmask8, a: __m512i, b: __m128i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 2);
        let c = _mm512_inserti64x2::<IMM8>(a, b).as_i64x8();
        transmute(simd_select_bitmask(k, c, i64x8::ZERO))
    }
}

// Convert

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundepi64_pd&ig_expand=1437)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2pd, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_cvt_roundepi64_pd<const ROUNDING: i32>(a: __m512i) -> __m512d {
    unsafe {
        static_assert_rounding!(ROUNDING);
        transmute(vcvtqq2pd_512(a.as_i64x8(), ROUNDING))
    }
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundepi64_pd&ig_expand=1438)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2pd, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_cvt_roundepi64_pd<const ROUNDING: i32>(
    src: __m512d,
    k: __mmask8,
    a: __m512i,
) -> __m512d {
    unsafe {
        static_assert_rounding!(ROUNDING);
        let b = _mm512_cvt_roundepi64_pd::<ROUNDING>(a).as_f64x8();
        transmute(simd_select_bitmask(k, b, src.as_f64x8()))
    }
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundepi64_pd&ig_expand=1439)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2pd, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvt_roundepi64_pd<const ROUNDING: i32>(k: __mmask8, a: __m512i) -> __m512d {
    unsafe {
        static_assert_rounding!(ROUNDING);
        let b = _mm512_cvt_roundepi64_pd::<ROUNDING>(a).as_f64x8();
        transmute(simd_select_bitmask(k, b, f64x8::ZERO))
    }
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtepi64_pd&ig_expand=1705)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_cvtepi64_pd(a: __m128i) -> __m128d {
    unsafe { transmute(vcvtqq2pd_128(a.as_i64x2(), _MM_FROUND_CUR_DIRECTION)) }
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
pub fn _mm_mask_cvtepi64_pd(src: __m128d, k: __mmask8, a: __m128i) -> __m128d {
    unsafe {
        let b = _mm_cvtepi64_pd(a).as_f64x2();
        transmute(simd_select_bitmask(k, b, src.as_f64x2()))
    }
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtepi64_pd&ig_expand=1707)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_cvtepi64_pd(k: __mmask8, a: __m128i) -> __m128d {
    unsafe {
        let b = _mm_cvtepi64_pd(a).as_f64x2();
        transmute(simd_select_bitmask(k, b, f64x2::ZERO))
    }
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtepi64_pd&ig_expand=1708)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_cvtepi64_pd(a: __m256i) -> __m256d {
    unsafe { transmute(vcvtqq2pd_256(a.as_i64x4(), _MM_FROUND_CUR_DIRECTION)) }
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
pub fn _mm256_mask_cvtepi64_pd(src: __m256d, k: __mmask8, a: __m256i) -> __m256d {
    unsafe {
        let b = _mm256_cvtepi64_pd(a).as_f64x4();
        transmute(simd_select_bitmask(k, b, src.as_f64x4()))
    }
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtepi64_pd&ig_expand=1710)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_cvtepi64_pd(k: __mmask8, a: __m256i) -> __m256d {
    unsafe {
        let b = _mm256_cvtepi64_pd(a).as_f64x4();
        transmute(simd_select_bitmask(k, b, f64x4::ZERO))
    }
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtepi64_pd&ig_expand=1711)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_cvtepi64_pd(a: __m512i) -> __m512d {
    unsafe { transmute(vcvtqq2pd_512(a.as_i64x8(), _MM_FROUND_CUR_DIRECTION)) }
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
pub fn _mm512_mask_cvtepi64_pd(src: __m512d, k: __mmask8, a: __m512i) -> __m512d {
    unsafe {
        let b = _mm512_cvtepi64_pd(a).as_f64x8();
        transmute(simd_select_bitmask(k, b, src.as_f64x8()))
    }
}

/// Convert packed signed 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtepi64_pd&ig_expand=1713)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvtepi64_pd(k: __mmask8, a: __m512i) -> __m512d {
    unsafe {
        let b = _mm512_cvtepi64_pd(a).as_f64x8();
        transmute(simd_select_bitmask(k, b, f64x8::ZERO))
    }
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundepi64_ps&ig_expand=1443)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2ps, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_cvt_roundepi64_ps<const ROUNDING: i32>(a: __m512i) -> __m256 {
    unsafe {
        static_assert_rounding!(ROUNDING);
        transmute(vcvtqq2ps_512(a.as_i64x8(), ROUNDING))
    }
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundepi64_ps&ig_expand=1444)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2ps, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_cvt_roundepi64_ps<const ROUNDING: i32>(
    src: __m256,
    k: __mmask8,
    a: __m512i,
) -> __m256 {
    unsafe {
        static_assert_rounding!(ROUNDING);
        let b = _mm512_cvt_roundepi64_ps::<ROUNDING>(a).as_f32x8();
        transmute(simd_select_bitmask(k, b, src.as_f32x8()))
    }
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundepi64_ps&ig_expand=1445)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2ps, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvt_roundepi64_ps<const ROUNDING: i32>(k: __mmask8, a: __m512i) -> __m256 {
    unsafe {
        static_assert_rounding!(ROUNDING);
        let b = _mm512_cvt_roundepi64_ps::<ROUNDING>(a).as_f32x8();
        transmute(simd_select_bitmask(k, b, f32x8::ZERO))
    }
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtepi64_ps&ig_expand=1723)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_cvtepi64_ps(a: __m128i) -> __m128 {
    _mm_mask_cvtepi64_ps(_mm_undefined_ps(), 0xff, a)
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
pub fn _mm_mask_cvtepi64_ps(src: __m128, k: __mmask8, a: __m128i) -> __m128 {
    unsafe { transmute(vcvtqq2ps_128(a.as_i64x2(), src.as_f32x4(), k)) }
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtepi64_ps&ig_expand=1725)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_cvtepi64_ps(k: __mmask8, a: __m128i) -> __m128 {
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
pub fn _mm256_cvtepi64_ps(a: __m256i) -> __m128 {
    unsafe { transmute(vcvtqq2ps_256(a.as_i64x4(), _MM_FROUND_CUR_DIRECTION)) }
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
pub fn _mm256_mask_cvtepi64_ps(src: __m128, k: __mmask8, a: __m256i) -> __m128 {
    unsafe {
        let b = _mm256_cvtepi64_ps(a).as_f32x4();
        transmute(simd_select_bitmask(k, b, src.as_f32x4()))
    }
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtepi64_ps&ig_expand=1728)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_cvtepi64_ps(k: __mmask8, a: __m256i) -> __m128 {
    unsafe {
        let b = _mm256_cvtepi64_ps(a).as_f32x4();
        transmute(simd_select_bitmask(k, b, f32x4::ZERO))
    }
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtepi64_ps&ig_expand=1729)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_cvtepi64_ps(a: __m512i) -> __m256 {
    unsafe { transmute(vcvtqq2ps_512(a.as_i64x8(), _MM_FROUND_CUR_DIRECTION)) }
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
pub fn _mm512_mask_cvtepi64_ps(src: __m256, k: __mmask8, a: __m512i) -> __m256 {
    unsafe {
        let b = _mm512_cvtepi64_ps(a).as_f32x8();
        transmute(simd_select_bitmask(k, b, src.as_f32x8()))
    }
}

/// Convert packed signed 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtepi64_ps&ig_expand=1731)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvtepi64_ps(k: __mmask8, a: __m512i) -> __m256 {
    unsafe {
        let b = _mm512_cvtepi64_ps(a).as_f32x8();
        transmute(simd_select_bitmask(k, b, f32x8::ZERO))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundepu64_pd&ig_expand=1455)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_cvt_roundepu64_pd<const ROUNDING: i32>(a: __m512i) -> __m512d {
    unsafe {
        static_assert_rounding!(ROUNDING);
        transmute(vcvtuqq2pd_512(a.as_u64x8(), ROUNDING))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundepu64_pd&ig_expand=1456)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_cvt_roundepu64_pd<const ROUNDING: i32>(
    src: __m512d,
    k: __mmask8,
    a: __m512i,
) -> __m512d {
    unsafe {
        static_assert_rounding!(ROUNDING);
        let b = _mm512_cvt_roundepu64_pd::<ROUNDING>(a).as_f64x8();
        transmute(simd_select_bitmask(k, b, src.as_f64x8()))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundepu64_pd&ig_expand=1457)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvt_roundepu64_pd<const ROUNDING: i32>(k: __mmask8, a: __m512i) -> __m512d {
    unsafe {
        static_assert_rounding!(ROUNDING);
        let b = _mm512_cvt_roundepu64_pd::<ROUNDING>(a).as_f64x8();
        transmute(simd_select_bitmask(k, b, f64x8::ZERO))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtepu64_pd&ig_expand=1827)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_cvtepu64_pd(a: __m128i) -> __m128d {
    unsafe { transmute(vcvtuqq2pd_128(a.as_u64x2(), _MM_FROUND_CUR_DIRECTION)) }
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
pub fn _mm_mask_cvtepu64_pd(src: __m128d, k: __mmask8, a: __m128i) -> __m128d {
    unsafe {
        let b = _mm_cvtepu64_pd(a).as_f64x2();
        transmute(simd_select_bitmask(k, b, src.as_f64x2()))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtepu64_pd&ig_expand=1829)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_cvtepu64_pd(k: __mmask8, a: __m128i) -> __m128d {
    unsafe {
        let b = _mm_cvtepu64_pd(a).as_f64x2();
        transmute(simd_select_bitmask(k, b, f64x2::ZERO))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtepu64_pd&ig_expand=1830)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_cvtepu64_pd(a: __m256i) -> __m256d {
    unsafe { transmute(vcvtuqq2pd_256(a.as_u64x4(), _MM_FROUND_CUR_DIRECTION)) }
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
pub fn _mm256_mask_cvtepu64_pd(src: __m256d, k: __mmask8, a: __m256i) -> __m256d {
    unsafe {
        let b = _mm256_cvtepu64_pd(a).as_f64x4();
        transmute(simd_select_bitmask(k, b, src.as_f64x4()))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtepu64_pd&ig_expand=1832)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_cvtepu64_pd(k: __mmask8, a: __m256i) -> __m256d {
    unsafe {
        let b = _mm256_cvtepu64_pd(a).as_f64x4();
        transmute(simd_select_bitmask(k, b, f64x4::ZERO))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtepu64_pd&ig_expand=1833)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_cvtepu64_pd(a: __m512i) -> __m512d {
    unsafe { transmute(vcvtuqq2pd_512(a.as_u64x8(), _MM_FROUND_CUR_DIRECTION)) }
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
pub fn _mm512_mask_cvtepu64_pd(src: __m512d, k: __mmask8, a: __m512i) -> __m512d {
    unsafe {
        let b = _mm512_cvtepu64_pd(a).as_f64x8();
        transmute(simd_select_bitmask(k, b, src.as_f64x8()))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed double-precision (64-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtepu64_pd&ig_expand=1835)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2pd))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvtepu64_pd(k: __mmask8, a: __m512i) -> __m512d {
    unsafe {
        let b = _mm512_cvtepu64_pd(a).as_f64x8();
        transmute(simd_select_bitmask(k, b, f64x8::ZERO))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundepu64_ps&ig_expand=1461)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_cvt_roundepu64_ps<const ROUNDING: i32>(a: __m512i) -> __m256 {
    unsafe {
        static_assert_rounding!(ROUNDING);
        transmute(vcvtuqq2ps_512(a.as_u64x8(), ROUNDING))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundepu64_ps&ig_expand=1462)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_cvt_roundepu64_ps<const ROUNDING: i32>(
    src: __m256,
    k: __mmask8,
    a: __m512i,
) -> __m256 {
    unsafe {
        static_assert_rounding!(ROUNDING);
        let b = _mm512_cvt_roundepu64_ps::<ROUNDING>(a).as_f32x8();
        transmute(simd_select_bitmask(k, b, src.as_f32x8()))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundepu64_ps&ig_expand=1463)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvt_roundepu64_ps<const ROUNDING: i32>(k: __mmask8, a: __m512i) -> __m256 {
    unsafe {
        static_assert_rounding!(ROUNDING);
        let b = _mm512_cvt_roundepu64_ps::<ROUNDING>(a).as_f32x8();
        transmute(simd_select_bitmask(k, b, f32x8::ZERO))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtepu64_ps&ig_expand=1845)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_cvtepu64_ps(a: __m128i) -> __m128 {
    _mm_mask_cvtepu64_ps(_mm_undefined_ps(), 0xff, a)
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
pub fn _mm_mask_cvtepu64_ps(src: __m128, k: __mmask8, a: __m128i) -> __m128 {
    unsafe { transmute(vcvtuqq2ps_128(a.as_u64x2(), src.as_f32x4(), k)) }
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtepu64_ps&ig_expand=1847)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_cvtepu64_ps(k: __mmask8, a: __m128i) -> __m128 {
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
pub fn _mm256_cvtepu64_ps(a: __m256i) -> __m128 {
    unsafe { transmute(vcvtuqq2ps_256(a.as_u64x4(), _MM_FROUND_CUR_DIRECTION)) }
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
pub fn _mm256_mask_cvtepu64_ps(src: __m128, k: __mmask8, a: __m256i) -> __m128 {
    unsafe {
        let b = _mm256_cvtepu64_ps(a).as_f32x4();
        transmute(simd_select_bitmask(k, b, src.as_f32x4()))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtepu64_ps&ig_expand=1850)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_cvtepu64_ps(k: __mmask8, a: __m256i) -> __m128 {
    unsafe {
        let b = _mm256_cvtepu64_ps(a).as_f32x4();
        transmute(simd_select_bitmask(k, b, f32x4::ZERO))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtepu64_ps&ig_expand=1851)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_cvtepu64_ps(a: __m512i) -> __m256 {
    unsafe { transmute(vcvtuqq2ps_512(a.as_u64x8(), _MM_FROUND_CUR_DIRECTION)) }
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
pub fn _mm512_mask_cvtepu64_ps(src: __m256, k: __mmask8, a: __m512i) -> __m256 {
    unsafe {
        let b = _mm512_cvtepu64_ps(a).as_f32x8();
        transmute(simd_select_bitmask(k, b, src.as_f32x8()))
    }
}

/// Convert packed unsigned 64-bit integers in a to packed single-precision (32-bit) floating-point elements,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtepu64_ps&ig_expand=1853)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtuqq2ps))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvtepu64_ps(k: __mmask8, a: __m512i) -> __m256 {
    unsafe {
        let b = _mm512_cvtepu64_ps(a).as_f32x8();
        transmute(simd_select_bitmask(k, b, f32x8::ZERO))
    }
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundpd_epi64&ig_expand=1472)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2qq, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_cvt_roundpd_epi64<const ROUNDING: i32>(a: __m512d) -> __m512i {
    static_assert_rounding!(ROUNDING);
    _mm512_mask_cvt_roundpd_epi64::<ROUNDING>(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundpd_epi64&ig_expand=1473)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2qq, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_cvt_roundpd_epi64<const ROUNDING: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m512d,
) -> __m512i {
    unsafe {
        static_assert_rounding!(ROUNDING);
        transmute(vcvtpd2qq_512(a.as_f64x8(), src.as_i64x8(), k, ROUNDING))
    }
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundpd_epi64&ig_expand=1474)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2qq, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvt_roundpd_epi64<const ROUNDING: i32>(k: __mmask8, a: __m512d) -> __m512i {
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
pub fn _mm_cvtpd_epi64(a: __m128d) -> __m128i {
    _mm_mask_cvtpd_epi64(_mm_undefined_si128(), 0xff, a)
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
pub fn _mm_mask_cvtpd_epi64(src: __m128i, k: __mmask8, a: __m128d) -> __m128i {
    unsafe { transmute(vcvtpd2qq_128(a.as_f64x2(), src.as_i64x2(), k)) }
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtpd_epi64&ig_expand=1943)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_cvtpd_epi64(k: __mmask8, a: __m128d) -> __m128i {
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
pub fn _mm256_cvtpd_epi64(a: __m256d) -> __m256i {
    _mm256_mask_cvtpd_epi64(_mm256_undefined_si256(), 0xff, a)
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
pub fn _mm256_mask_cvtpd_epi64(src: __m256i, k: __mmask8, a: __m256d) -> __m256i {
    unsafe { transmute(vcvtpd2qq_256(a.as_f64x4(), src.as_i64x4(), k)) }
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtpd_epi64&ig_expand=1946)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_cvtpd_epi64(k: __mmask8, a: __m256d) -> __m256i {
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
pub fn _mm512_cvtpd_epi64(a: __m512d) -> __m512i {
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
pub fn _mm512_mask_cvtpd_epi64(src: __m512i, k: __mmask8, a: __m512d) -> __m512i {
    unsafe {
        transmute(vcvtpd2qq_512(
            a.as_f64x8(),
            src.as_i64x8(),
            k,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtpd_epi64&ig_expand=1949)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvtpd_epi64(k: __mmask8, a: __m512d) -> __m512i {
    _mm512_mask_cvtpd_epi64(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundps_epi64&ig_expand=1514)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2qq, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_cvt_roundps_epi64<const ROUNDING: i32>(a: __m256) -> __m512i {
    static_assert_rounding!(ROUNDING);
    _mm512_mask_cvt_roundps_epi64::<ROUNDING>(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundps_epi64&ig_expand=1515)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2qq, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_cvt_roundps_epi64<const ROUNDING: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m256,
) -> __m512i {
    unsafe {
        static_assert_rounding!(ROUNDING);
        transmute(vcvtps2qq_512(a.as_f32x8(), src.as_i64x8(), k, ROUNDING))
    }
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundps_epi64&ig_expand=1516)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2qq, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvt_roundps_epi64<const ROUNDING: i32>(k: __mmask8, a: __m256) -> __m512i {
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
pub fn _mm_cvtps_epi64(a: __m128) -> __m128i {
    _mm_mask_cvtps_epi64(_mm_undefined_si128(), 0xff, a)
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
pub fn _mm_mask_cvtps_epi64(src: __m128i, k: __mmask8, a: __m128) -> __m128i {
    unsafe { transmute(vcvtps2qq_128(a.as_f32x4(), src.as_i64x2(), k)) }
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtps_epi64&ig_expand=2077)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_cvtps_epi64(k: __mmask8, a: __m128) -> __m128i {
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
pub fn _mm256_cvtps_epi64(a: __m128) -> __m256i {
    _mm256_mask_cvtps_epi64(_mm256_undefined_si256(), 0xff, a)
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
pub fn _mm256_mask_cvtps_epi64(src: __m256i, k: __mmask8, a: __m128) -> __m256i {
    unsafe { transmute(vcvtps2qq_256(a.as_f32x4(), src.as_i64x4(), k)) }
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtps_epi64&ig_expand=2080)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_cvtps_epi64(k: __mmask8, a: __m128) -> __m256i {
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
pub fn _mm512_cvtps_epi64(a: __m256) -> __m512i {
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
pub fn _mm512_mask_cvtps_epi64(src: __m512i, k: __mmask8, a: __m256) -> __m512i {
    unsafe {
        transmute(vcvtps2qq_512(
            a.as_f32x8(),
            src.as_i64x8(),
            k,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed signed 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtps_epi64&ig_expand=2083)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2qq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvtps_epi64(k: __mmask8, a: __m256) -> __m512i {
    _mm512_mask_cvtps_epi64(_mm512_setzero_si512(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundpd_epu64&ig_expand=1478)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_cvt_roundpd_epu64<const ROUNDING: i32>(a: __m512d) -> __m512i {
    static_assert_rounding!(ROUNDING);
    _mm512_mask_cvt_roundpd_epu64::<ROUNDING>(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundpd_epu64&ig_expand=1479)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_cvt_roundpd_epu64<const ROUNDING: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m512d,
) -> __m512i {
    unsafe {
        static_assert_rounding!(ROUNDING);
        transmute(vcvtpd2uqq_512(a.as_f64x8(), src.as_u64x8(), k, ROUNDING))
    }
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundpd_epu64&ig_expand=1480)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvt_roundpd_epu64<const ROUNDING: i32>(k: __mmask8, a: __m512d) -> __m512i {
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
pub fn _mm_cvtpd_epu64(a: __m128d) -> __m128i {
    _mm_mask_cvtpd_epu64(_mm_undefined_si128(), 0xff, a)
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
pub fn _mm_mask_cvtpd_epu64(src: __m128i, k: __mmask8, a: __m128d) -> __m128i {
    unsafe { transmute(vcvtpd2uqq_128(a.as_f64x2(), src.as_u64x2(), k)) }
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtpd_epu64&ig_expand=1961)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_cvtpd_epu64(k: __mmask8, a: __m128d) -> __m128i {
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
pub fn _mm256_cvtpd_epu64(a: __m256d) -> __m256i {
    _mm256_mask_cvtpd_epu64(_mm256_undefined_si256(), 0xff, a)
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
pub fn _mm256_mask_cvtpd_epu64(src: __m256i, k: __mmask8, a: __m256d) -> __m256i {
    unsafe { transmute(vcvtpd2uqq_256(a.as_f64x4(), src.as_u64x4(), k)) }
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtpd_epu64&ig_expand=1964)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_cvtpd_epu64(k: __mmask8, a: __m256d) -> __m256i {
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
pub fn _mm512_cvtpd_epu64(a: __m512d) -> __m512i {
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
pub fn _mm512_mask_cvtpd_epu64(src: __m512i, k: __mmask8, a: __m512d) -> __m512i {
    unsafe {
        transmute(vcvtpd2uqq_512(
            a.as_f64x8(),
            src.as_u64x8(),
            k,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtpd_epu64&ig_expand=1967)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtpd2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvtpd_epu64(k: __mmask8, a: __m512d) -> __m512i {
    _mm512_mask_cvtpd_epu64(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst. Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvt_roundps_epu64&ig_expand=1520)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2uqq, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_cvt_roundps_epu64<const ROUNDING: i32>(a: __m256) -> __m512i {
    static_assert_rounding!(ROUNDING);
    _mm512_mask_cvt_roundps_epu64::<ROUNDING>(_mm512_undefined_epi32(), 0xff, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using writemask k (elements are copied from src if the corresponding bit is
/// not set). Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvt_roundps_epu64&ig_expand=1521)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2uqq, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_cvt_roundps_epu64<const ROUNDING: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m256,
) -> __m512i {
    unsafe {
        static_assert_rounding!(ROUNDING);
        transmute(vcvtps2uqq_512(a.as_f32x8(), src.as_u64x8(), k, ROUNDING))
    }
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
/// Rounding is done according to the ROUNDING parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvt_roundps_epu64&ig_expand=1522)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2uqq, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvt_roundps_epu64<const ROUNDING: i32>(k: __mmask8, a: __m256) -> __m512i {
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
pub fn _mm_cvtps_epu64(a: __m128) -> __m128i {
    _mm_mask_cvtps_epu64(_mm_undefined_si128(), 0xff, a)
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
pub fn _mm_mask_cvtps_epu64(src: __m128i, k: __mmask8, a: __m128) -> __m128i {
    unsafe { transmute(vcvtps2uqq_128(a.as_f32x4(), src.as_u64x2(), k)) }
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtps_epu64&ig_expand=2095)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_cvtps_epu64(k: __mmask8, a: __m128) -> __m128i {
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
pub fn _mm256_cvtps_epu64(a: __m128) -> __m256i {
    _mm256_mask_cvtps_epu64(_mm256_undefined_si256(), 0xff, a)
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
pub fn _mm256_mask_cvtps_epu64(src: __m256i, k: __mmask8, a: __m128) -> __m256i {
    unsafe { transmute(vcvtps2uqq_256(a.as_f32x4(), src.as_u64x4(), k)) }
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtps_epu64&ig_expand=2098)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vcvtps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_cvtps_epu64(k: __mmask8, a: __m128) -> __m256i {
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
pub fn _mm512_cvtps_epu64(a: __m256) -> __m512i {
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
pub fn _mm512_mask_cvtps_epu64(src: __m512i, k: __mmask8, a: __m256) -> __m512i {
    unsafe {
        transmute(vcvtps2uqq_512(
            a.as_f32x8(),
            src.as_u64x8(),
            k,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed unsigned 64-bit integers,
/// and store the results in dst using zeromask k (elements are zeroed out if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtps_epu64&ig_expand=2101)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvtps2uqq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_cvtps_epu64(k: __mmask8, a: __m256) -> __m512i {
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
pub fn _mm512_cvtt_roundpd_epi64<const SAE: i32>(a: __m512d) -> __m512i {
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
pub fn _mm512_mask_cvtt_roundpd_epi64<const SAE: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m512d,
) -> __m512i {
    unsafe {
        static_assert_sae!(SAE);
        transmute(vcvttpd2qq_512(a.as_f64x8(), src.as_i64x8(), k, SAE))
    }
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
pub fn _mm512_maskz_cvtt_roundpd_epi64<const SAE: i32>(k: __mmask8, a: __m512d) -> __m512i {
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
pub fn _mm_cvttpd_epi64(a: __m128d) -> __m128i {
    _mm_mask_cvttpd_epi64(_mm_undefined_si128(), 0xff, a)
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
pub fn _mm_mask_cvttpd_epi64(src: __m128i, k: __mmask8, a: __m128d) -> __m128i {
    unsafe { transmute(vcvttpd2qq_128(a.as_f64x2(), src.as_i64x2(), k)) }
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
pub fn _mm_maskz_cvttpd_epi64(k: __mmask8, a: __m128d) -> __m128i {
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
pub fn _mm256_cvttpd_epi64(a: __m256d) -> __m256i {
    _mm256_mask_cvttpd_epi64(_mm256_undefined_si256(), 0xff, a)
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
pub fn _mm256_mask_cvttpd_epi64(src: __m256i, k: __mmask8, a: __m256d) -> __m256i {
    unsafe { transmute(vcvttpd2qq_256(a.as_f64x4(), src.as_i64x4(), k)) }
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
pub fn _mm256_maskz_cvttpd_epi64(k: __mmask8, a: __m256d) -> __m256i {
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
pub fn _mm512_cvttpd_epi64(a: __m512d) -> __m512i {
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
pub fn _mm512_mask_cvttpd_epi64(src: __m512i, k: __mmask8, a: __m512d) -> __m512i {
    unsafe {
        transmute(vcvttpd2qq_512(
            a.as_f64x8(),
            src.as_i64x8(),
            k,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
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
pub fn _mm512_maskz_cvttpd_epi64(k: __mmask8, a: __m512d) -> __m512i {
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
pub fn _mm512_cvtt_roundps_epi64<const SAE: i32>(a: __m256) -> __m512i {
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
pub fn _mm512_mask_cvtt_roundps_epi64<const SAE: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m256,
) -> __m512i {
    unsafe {
        static_assert_sae!(SAE);
        transmute(vcvttps2qq_512(a.as_f32x8(), src.as_i64x8(), k, SAE))
    }
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
pub fn _mm512_maskz_cvtt_roundps_epi64<const SAE: i32>(k: __mmask8, a: __m256) -> __m512i {
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
pub fn _mm_cvttps_epi64(a: __m128) -> __m128i {
    _mm_mask_cvttps_epi64(_mm_undefined_si128(), 0xff, a)
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
pub fn _mm_mask_cvttps_epi64(src: __m128i, k: __mmask8, a: __m128) -> __m128i {
    unsafe { transmute(vcvttps2qq_128(a.as_f32x4(), src.as_i64x2(), k)) }
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
pub fn _mm_maskz_cvttps_epi64(k: __mmask8, a: __m128) -> __m128i {
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
pub fn _mm256_cvttps_epi64(a: __m128) -> __m256i {
    _mm256_mask_cvttps_epi64(_mm256_undefined_si256(), 0xff, a)
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
pub fn _mm256_mask_cvttps_epi64(src: __m256i, k: __mmask8, a: __m128) -> __m256i {
    unsafe { transmute(vcvttps2qq_256(a.as_f32x4(), src.as_i64x4(), k)) }
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
pub fn _mm256_maskz_cvttps_epi64(k: __mmask8, a: __m128) -> __m256i {
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
pub fn _mm512_cvttps_epi64(a: __m256) -> __m512i {
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
pub fn _mm512_mask_cvttps_epi64(src: __m512i, k: __mmask8, a: __m256) -> __m512i {
    unsafe {
        transmute(vcvttps2qq_512(
            a.as_f32x8(),
            src.as_i64x8(),
            k,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
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
pub fn _mm512_maskz_cvttps_epi64(k: __mmask8, a: __m256) -> __m512i {
    _mm512_mask_cvttps_epi64(_mm512_setzero_si512(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in a to packed unsigned 64-bit integers
/// with truncation, and store the result in dst. Exceptions can be suppressed by passing _MM_FROUND_NO_EXC
/// to the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtt_roundpd_epu64&ig_expand=1965)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vcvttpd2uqq, SAE = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_cvtt_roundpd_epu64<const SAE: i32>(a: __m512d) -> __m512i {
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
pub fn _mm512_mask_cvtt_roundpd_epu64<const SAE: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m512d,
) -> __m512i {
    unsafe {
        static_assert_sae!(SAE);
        transmute(vcvttpd2uqq_512(a.as_f64x8(), src.as_u64x8(), k, SAE))
    }
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
pub fn _mm512_maskz_cvtt_roundpd_epu64<const SAE: i32>(k: __mmask8, a: __m512d) -> __m512i {
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
pub fn _mm_cvttpd_epu64(a: __m128d) -> __m128i {
    _mm_mask_cvttpd_epu64(_mm_undefined_si128(), 0xff, a)
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
pub fn _mm_mask_cvttpd_epu64(src: __m128i, k: __mmask8, a: __m128d) -> __m128i {
    unsafe { transmute(vcvttpd2uqq_128(a.as_f64x2(), src.as_u64x2(), k)) }
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
pub fn _mm_maskz_cvttpd_epu64(k: __mmask8, a: __m128d) -> __m128i {
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
pub fn _mm256_cvttpd_epu64(a: __m256d) -> __m256i {
    _mm256_mask_cvttpd_epu64(_mm256_undefined_si256(), 0xff, a)
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
pub fn _mm256_mask_cvttpd_epu64(src: __m256i, k: __mmask8, a: __m256d) -> __m256i {
    unsafe { transmute(vcvttpd2uqq_256(a.as_f64x4(), src.as_u64x4(), k)) }
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
pub fn _mm256_maskz_cvttpd_epu64(k: __mmask8, a: __m256d) -> __m256i {
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
pub fn _mm512_cvttpd_epu64(a: __m512d) -> __m512i {
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
pub fn _mm512_mask_cvttpd_epu64(src: __m512i, k: __mmask8, a: __m512d) -> __m512i {
    unsafe {
        transmute(vcvttpd2uqq_512(
            a.as_f64x8(),
            src.as_u64x8(),
            k,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
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
pub fn _mm512_maskz_cvttpd_epu64(k: __mmask8, a: __m512d) -> __m512i {
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
pub fn _mm512_cvtt_roundps_epu64<const SAE: i32>(a: __m256) -> __m512i {
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
pub fn _mm512_mask_cvtt_roundps_epu64<const SAE: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m256,
) -> __m512i {
    unsafe {
        static_assert_sae!(SAE);
        transmute(vcvttps2uqq_512(a.as_f32x8(), src.as_u64x8(), k, SAE))
    }
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
pub fn _mm512_maskz_cvtt_roundps_epu64<const SAE: i32>(k: __mmask8, a: __m256) -> __m512i {
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
pub fn _mm_cvttps_epu64(a: __m128) -> __m128i {
    _mm_mask_cvttps_epu64(_mm_undefined_si128(), 0xff, a)
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
pub fn _mm_mask_cvttps_epu64(src: __m128i, k: __mmask8, a: __m128) -> __m128i {
    unsafe { transmute(vcvttps2uqq_128(a.as_f32x4(), src.as_u64x2(), k)) }
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
pub fn _mm_maskz_cvttps_epu64(k: __mmask8, a: __m128) -> __m128i {
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
pub fn _mm256_cvttps_epu64(a: __m128) -> __m256i {
    _mm256_mask_cvttps_epu64(_mm256_undefined_si256(), 0xff, a)
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
pub fn _mm256_mask_cvttps_epu64(src: __m256i, k: __mmask8, a: __m128) -> __m256i {
    unsafe { transmute(vcvttps2uqq_256(a.as_f32x4(), src.as_u64x4(), k)) }
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
pub fn _mm256_maskz_cvttps_epu64(k: __mmask8, a: __m128) -> __m256i {
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
pub fn _mm512_cvttps_epu64(a: __m256) -> __m512i {
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
pub fn _mm512_mask_cvttps_epu64(src: __m512i, k: __mmask8, a: __m256) -> __m512i {
    unsafe {
        transmute(vcvttps2uqq_512(
            a.as_f32x8(),
            src.as_u64x8(),
            k,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
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
pub fn _mm512_maskz_cvttps_epu64(k: __mmask8, a: __m256) -> __m512i {
    _mm512_mask_cvttps_epu64(_mm512_setzero_si512(), k, a)
}

// Multiply-Low

/// Multiply packed 64-bit integers in `a` and `b`, producing intermediate 128-bit integers, and store
/// the low 64 bits of the intermediate integers in `dst`.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mullo_epi64&ig_expand=4778)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vpmullq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mullo_epi64(a: __m128i, b: __m128i) -> __m128i {
    unsafe { transmute(simd_mul(a.as_i64x2(), b.as_i64x2())) }
}

/// Multiply packed 64-bit integers in `a` and `b`, producing intermediate 128-bit integers, and store
/// the low 64 bits of the intermediate integers in `dst` using writemask `k` (elements are copied from
/// `src` if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_mullo_epi64&ig_expand=4776)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vpmullq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_mullo_epi64(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let b = _mm_mullo_epi64(a, b).as_i64x2();
        transmute(simd_select_bitmask(k, b, src.as_i64x2()))
    }
}

/// Multiply packed 64-bit integers in `a` and `b`, producing intermediate 128-bit integers, and store
/// the low 64 bits of the intermediate integers in `dst` using zeromask `k` (elements are zeroed out if
/// the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_mullo_epi64&ig_expand=4777)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vpmullq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_mullo_epi64(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let b = _mm_mullo_epi64(a, b).as_i64x2();
        transmute(simd_select_bitmask(k, b, i64x2::ZERO))
    }
}

/// Multiply packed 64-bit integers in `a` and `b`, producing intermediate 128-bit integers, and store
/// the low 64 bits of the intermediate integers in `dst`.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mullo_epi64&ig_expand=4781)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vpmullq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mullo_epi64(a: __m256i, b: __m256i) -> __m256i {
    unsafe { transmute(simd_mul(a.as_i64x4(), b.as_i64x4())) }
}

/// Multiply packed 64-bit integers in `a` and `b`, producing intermediate 128-bit integers, and store
/// the low 64 bits of the intermediate integers in `dst` using writemask `k` (elements are copied from
/// `src` if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_mullo_epi64&ig_expand=4779)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vpmullq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_mullo_epi64(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let b = _mm256_mullo_epi64(a, b).as_i64x4();
        transmute(simd_select_bitmask(k, b, src.as_i64x4()))
    }
}

/// Multiply packed 64-bit integers in `a` and `b`, producing intermediate 128-bit integers, and store
/// the low 64 bits of the intermediate integers in `dst` using zeromask `k` (elements are zeroed out if
/// the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_mullo_epi64&ig_expand=4780)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vpmullq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_mullo_epi64(k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let b = _mm256_mullo_epi64(a, b).as_i64x4();
        transmute(simd_select_bitmask(k, b, i64x4::ZERO))
    }
}

/// Multiply packed 64-bit integers in `a` and `b`, producing intermediate 128-bit integers, and store
/// the low 64 bits of the intermediate integers in `dst`.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mullo_epi64&ig_expand=4784)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vpmullq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mullo_epi64(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_mul(a.as_i64x8(), b.as_i64x8())) }
}

/// Multiply packed 64-bit integers in `a` and `b`, producing intermediate 128-bit integers, and store
/// the low 64 bits of the intermediate integers in `dst` using writemask `k` (elements are copied from
/// `src` if the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_mullo_epi64&ig_expand=4782)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vpmullq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_mullo_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let b = _mm512_mullo_epi64(a, b).as_i64x8();
        transmute(simd_select_bitmask(k, b, src.as_i64x8()))
    }
}

/// Multiply packed 64-bit integers in `a` and `b`, producing intermediate 128-bit integers, and store
/// the low 64 bits of the intermediate integers in `dst` using zeromask `k` (elements are zeroed out if
/// the corresponding bit is not set).
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_mullo_epi64&ig_expand=4783)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vpmullq))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_mullo_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let b = _mm512_mullo_epi64(a, b).as_i64x8();
        transmute(simd_select_bitmask(k, b, i64x8::ZERO))
    }
}

// Mask Registers

/// Convert 8-bit mask a to a 32-bit integer value and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_cvtmask8_u32&ig_expand=1891)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _cvtmask8_u32(a: __mmask8) -> u32 {
    a as u32
}

/// Convert 32-bit integer value a to an 8-bit mask and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_cvtu32_mask8&ig_expand=2467)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _cvtu32_mask8(a: u32) -> __mmask8 {
    a as __mmask8
}

/// Add 16-bit masks a and b, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kadd_mask16&ig_expand=3903)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kadd_mask16(a: __mmask16, b: __mmask16) -> __mmask16 {
    a + b
}

/// Add 8-bit masks a and b, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kadd_mask8&ig_expand=3906)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kadd_mask8(a: __mmask8, b: __mmask8) -> __mmask8 {
    a + b
}

/// Bitwise AND of 8-bit masks a and b, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kand_mask8&ig_expand=3911)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kand_mask8(a: __mmask8, b: __mmask8) -> __mmask8 {
    a & b
}

/// Bitwise AND NOT of 8-bit masks a and b, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kandn_mask8&ig_expand=3916)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kandn_mask8(a: __mmask8, b: __mmask8) -> __mmask8 {
    _knot_mask8(a) & b
}

/// Bitwise NOT of 8-bit mask a, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_knot_mask8&ig_expand=3922)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _knot_mask8(a: __mmask8) -> __mmask8 {
    a ^ 0b11111111
}

/// Bitwise OR of 8-bit masks a and b, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kor_mask8&ig_expand=3927)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kor_mask8(a: __mmask8, b: __mmask8) -> __mmask8 {
    a | b
}

/// Bitwise XNOR of 8-bit masks a and b, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kxnor_mask8&ig_expand=3969)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kxnor_mask8(a: __mmask8, b: __mmask8) -> __mmask8 {
    _knot_mask8(_kxor_mask8(a, b))
}

/// Bitwise XOR of 8-bit masks a and b, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kxor_mask8&ig_expand=3974)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kxor_mask8(a: __mmask8, b: __mmask8) -> __mmask8 {
    a ^ b
}

/// Compute the bitwise OR of 8-bit masks a and b. If the result is all zeros, store 1 in dst, otherwise
/// store 0 in dst. If the result is all ones, store 1 in all_ones, otherwise store 0 in all_ones.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kortest_mask8_u8&ig_expand=3931)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _kortest_mask8_u8(a: __mmask8, b: __mmask8, all_ones: *mut u8) -> u8 {
    let tmp = _kor_mask8(a, b);
    *all_ones = (tmp == 0xff) as u8;
    (tmp == 0) as u8
}

/// Compute the bitwise OR of 8-bit masks a and b. If the result is all ones, store 1 in dst, otherwise
/// store 0 in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kortestc_mask8_u8&ig_expand=3936)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kortestc_mask8_u8(a: __mmask8, b: __mmask8) -> u8 {
    (_kor_mask8(a, b) == 0xff) as u8
}

/// Compute the bitwise OR of 8-bit masks a and b. If the result is all zeros, store 1 in dst, otherwise
/// store 0 in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kortestz_mask8_u8&ig_expand=3941)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kortestz_mask8_u8(a: __mmask8, b: __mmask8) -> u8 {
    (_kor_mask8(a, b) == 0) as u8
}

/// Shift 8-bit mask a left by count bits while shifting in zeros, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kshiftli_mask8&ig_expand=3945)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kshiftli_mask8<const COUNT: u32>(a: __mmask8) -> __mmask8 {
    a << COUNT
}

/// Shift 8-bit mask a right by count bits while shifting in zeros, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kshiftri_mask8&ig_expand=3949)
#[inline]
#[target_feature(enable = "avx512dq")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kshiftri_mask8<const COUNT: u32>(a: __mmask8) -> __mmask8 {
    a >> COUNT
}

/// Compute the bitwise AND of 16-bit masks a and b, and if the result is all zeros, store 1 in dst,
/// otherwise store 0 in dst. Compute the bitwise NOT of a and then AND with b, if the result is all
/// zeros, store 1 in and_not, otherwise store 0 in and_not.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_ktest_mask16_u8&ig_expand=3950)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _ktest_mask16_u8(a: __mmask16, b: __mmask16, and_not: *mut u8) -> u8 {
    *and_not = (_kandn_mask16(a, b) == 0) as u8;
    (_kand_mask16(a, b) == 0) as u8
}

/// Compute the bitwise AND of 8-bit masks a and b, and if the result is all zeros, store 1 in dst,
/// otherwise store 0 in dst. Compute the bitwise NOT of a and then AND with b, if the result is all
/// zeros, store 1 in and_not, otherwise store 0 in and_not.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_ktest_mask8_u8&ig_expand=3953)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _ktest_mask8_u8(a: __mmask8, b: __mmask8, and_not: *mut u8) -> u8 {
    *and_not = (_kandn_mask8(a, b) == 0) as u8;
    (_kand_mask8(a, b) == 0) as u8
}

/// Compute the bitwise NOT of 16-bit mask a and then AND with 16-bit mask b, if the result is all
/// zeros, store 1 in dst, otherwise store 0 in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_ktestc_mask16_u8&ig_expand=3954)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _ktestc_mask16_u8(a: __mmask16, b: __mmask16) -> u8 {
    (_kandn_mask16(a, b) == 0) as u8
}

/// Compute the bitwise NOT of 8-bit mask a and then AND with 8-bit mask b, if the result is all
/// zeros, store 1 in dst, otherwise store 0 in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_ktestc_mask8_u8&ig_expand=3957)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _ktestc_mask8_u8(a: __mmask8, b: __mmask8) -> u8 {
    (_kandn_mask8(a, b) == 0) as u8
}

/// Compute the bitwise AND of 16-bit masks a and  b, if the result is all zeros, store 1 in dst, otherwise
/// store 0 in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_ktestz_mask16_u8&ig_expand=3958)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _ktestz_mask16_u8(a: __mmask16, b: __mmask16) -> u8 {
    (_kand_mask16(a, b) == 0) as u8
}

/// Compute the bitwise AND of 8-bit masks a and  b, if the result is all zeros, store 1 in dst, otherwise
/// store 0 in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_ktestz_mask8_u8&ig_expand=3961)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _ktestz_mask8_u8(a: __mmask8, b: __mmask8) -> u8 {
    (_kand_mask8(a, b) == 0) as u8
}

/// Load 8-bit mask from memory
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_load_mask8&ig_expand=3999)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _load_mask8(mem_addr: *const __mmask8) -> __mmask8 {
    *mem_addr
}

/// Store 8-bit mask to memory
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_store_mask8&ig_expand=6468)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _store_mask8(mem_addr: *mut __mmask8, a: __mmask8) {
    *mem_addr = a;
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 32-bit
/// integer in a.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movepi32_mask&ig_expand=4612)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_movepi32_mask(a: __m128i) -> __mmask8 {
    let zero = _mm_setzero_si128();
    _mm_cmplt_epi32_mask(a, zero)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 32-bit
/// integer in a.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_movepi32_mask&ig_expand=4613)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_movepi32_mask(a: __m256i) -> __mmask8 {
    let zero = _mm256_setzero_si256();
    _mm256_cmplt_epi32_mask(a, zero)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 32-bit
/// integer in a.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_movepi32_mask&ig_expand=4614)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_movepi32_mask(a: __m512i) -> __mmask16 {
    let zero = _mm512_setzero_si512();
    _mm512_cmplt_epi32_mask(a, zero)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 64-bit
/// integer in a.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movepi64_mask&ig_expand=4615)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_movepi64_mask(a: __m128i) -> __mmask8 {
    let zero = _mm_setzero_si128();
    _mm_cmplt_epi64_mask(a, zero)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 64-bit
/// integer in a.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_movepi64_mask&ig_expand=4616)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_movepi64_mask(a: __m256i) -> __mmask8 {
    let zero = _mm256_setzero_si256();
    _mm256_cmplt_epi64_mask(a, zero)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 64-bit
/// integer in a.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_movepi64_mask&ig_expand=4617)
#[inline]
#[target_feature(enable = "avx512dq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_movepi64_mask(a: __m512i) -> __mmask8 {
    let zero = _mm512_setzero_si512();
    _mm512_cmplt_epi64_mask(a, zero)
}

/// Set each packed 32-bit integer in dst to all ones or all zeros based on the value of the corresponding
/// bit in k.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movm_epi32&ig_expand=4625)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovm2d))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_movm_epi32(k: __mmask8) -> __m128i {
    let ones = _mm_set1_epi32(-1);
    _mm_maskz_mov_epi32(k, ones)
}

/// Set each packed 32-bit integer in dst to all ones or all zeros based on the value of the corresponding
/// bit in k.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_movm_epi32&ig_expand=4626)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovm2d))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_movm_epi32(k: __mmask8) -> __m256i {
    let ones = _mm256_set1_epi32(-1);
    _mm256_maskz_mov_epi32(k, ones)
}

/// Set each packed 32-bit integer in dst to all ones or all zeros based on the value of the corresponding
/// bit in k.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_movm_epi32&ig_expand=4627)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vpmovm2d))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_movm_epi32(k: __mmask16) -> __m512i {
    let ones = _mm512_set1_epi32(-1);
    _mm512_maskz_mov_epi32(k, ones)
}

/// Set each packed 64-bit integer in dst to all ones or all zeros based on the value of the corresponding
/// bit in k.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movm_epi64&ig_expand=4628)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovm2q))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_movm_epi64(k: __mmask8) -> __m128i {
    let ones = _mm_set1_epi64x(-1);
    _mm_maskz_mov_epi64(k, ones)
}

/// Set each packed 64-bit integer in dst to all ones or all zeros based on the value of the corresponding
/// bit in k.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_movm_epi64&ig_expand=4629)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovm2q))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_movm_epi64(k: __mmask8) -> __m256i {
    let ones = _mm256_set1_epi64x(-1);
    _mm256_maskz_mov_epi64(k, ones)
}

/// Set each packed 64-bit integer in dst to all ones or all zeros based on the value of the corresponding
/// bit in k.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_movm_epi64&ig_expand=4630)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vpmovm2q))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_movm_epi64(k: __mmask8) -> __m512i {
    let ones = _mm512_set1_epi64(-1);
    _mm512_maskz_mov_epi64(k, ones)
}

// Range

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_range_round_pd&ig_expand=5210)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangepd, IMM8 = 5, SAE = 8))]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_range_round_pd<const IMM8: i32, const SAE: i32>(a: __m512d, b: __m512d) -> __m512d {
    static_assert_uimm_bits!(IMM8, 4);
    static_assert_sae!(SAE);
    _mm512_mask_range_round_pd::<IMM8, SAE>(_mm512_setzero_pd(), 0xff, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// double-precision (64-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src to dst if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_range_round_pd&ig_expand=5208)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangepd, IMM8 = 5, SAE = 8))]
#[rustc_legacy_const_generics(4, 5)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_range_round_pd<const IMM8: i32, const SAE: i32>(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
) -> __m512d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 4);
        static_assert_sae!(SAE);
        transmute(vrangepd_512(
            a.as_f64x8(),
            b.as_f64x8(),
            IMM8,
            src.as_f64x8(),
            k,
            SAE,
        ))
    }
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// double-precision (64-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_range_round_pd&ig_expand=5209)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangepd, IMM8 = 5, SAE = 8))]
#[rustc_legacy_const_generics(3, 4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_range_round_pd<const IMM8: i32, const SAE: i32>(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
) -> __m512d {
    static_assert_uimm_bits!(IMM8, 4);
    static_assert_sae!(SAE);
    _mm512_mask_range_round_pd::<IMM8, SAE>(_mm512_setzero_pd(), k, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_range_pd&ig_expand=5192)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vrangepd, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_range_pd<const IMM8: i32>(a: __m128d, b: __m128d) -> __m128d {
    static_assert_uimm_bits!(IMM8, 4);
    _mm_mask_range_pd::<IMM8>(_mm_setzero_pd(), 0xff, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// double-precision (64-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src to dst if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_range_pd&ig_expand=5190)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vrangepd, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_range_pd<const IMM8: i32>(
    src: __m128d,
    k: __mmask8,
    a: __m128d,
    b: __m128d,
) -> __m128d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 4);
        transmute(vrangepd_128(
            a.as_f64x2(),
            b.as_f64x2(),
            IMM8,
            src.as_f64x2(),
            k,
        ))
    }
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// double-precision (64-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_range_pd&ig_expand=5191)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vrangepd, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_range_pd<const IMM8: i32>(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    static_assert_uimm_bits!(IMM8, 4);
    _mm_mask_range_pd::<IMM8>(_mm_setzero_pd(), k, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_range_pd&ig_expand=5195)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vrangepd, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_range_pd<const IMM8: i32>(a: __m256d, b: __m256d) -> __m256d {
    static_assert_uimm_bits!(IMM8, 4);
    _mm256_mask_range_pd::<IMM8>(_mm256_setzero_pd(), 0xff, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// double-precision (64-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src to dst if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_range_pd&ig_expand=5193)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vrangepd, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_range_pd<const IMM8: i32>(
    src: __m256d,
    k: __mmask8,
    a: __m256d,
    b: __m256d,
) -> __m256d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 4);
        transmute(vrangepd_256(
            a.as_f64x4(),
            b.as_f64x4(),
            IMM8,
            src.as_f64x4(),
            k,
        ))
    }
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// double-precision (64-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_range_pd&ig_expand=5194)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vrangepd, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_range_pd<const IMM8: i32>(k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    static_assert_uimm_bits!(IMM8, 4);
    _mm256_mask_range_pd::<IMM8>(_mm256_setzero_pd(), k, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_range_pd&ig_expand=5198)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangepd, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_range_pd<const IMM8: i32>(a: __m512d, b: __m512d) -> __m512d {
    static_assert_uimm_bits!(IMM8, 4);
    _mm512_mask_range_pd::<IMM8>(_mm512_setzero_pd(), 0xff, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// double-precision (64-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src to dst if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_range_pd&ig_expand=5196)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangepd, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_range_pd<const IMM8: i32>(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
) -> __m512d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 4);
        transmute(vrangepd_512(
            a.as_f64x8(),
            b.as_f64x8(),
            IMM8,
            src.as_f64x8(),
            k,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// double-precision (64-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_range_pd&ig_expand=5197)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangepd, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_range_pd<const IMM8: i32>(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    static_assert_uimm_bits!(IMM8, 4);
    _mm512_mask_range_pd::<IMM8>(_mm512_setzero_pd(), k, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_range_round_ps&ig_expand=5213)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangeps, IMM8 = 5, SAE = 8))]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_range_round_ps<const IMM8: i32, const SAE: i32>(a: __m512, b: __m512) -> __m512 {
    static_assert_uimm_bits!(IMM8, 4);
    static_assert_sae!(SAE);
    _mm512_mask_range_round_ps::<IMM8, SAE>(_mm512_setzero_ps(), 0xffff, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// single-precision (32-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src to dst if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_range_round_ps&ig_expand=5211)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangeps, IMM8 = 5, SAE = 8))]
#[rustc_legacy_const_generics(4, 5)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_range_round_ps<const IMM8: i32, const SAE: i32>(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m512,
) -> __m512 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 4);
        static_assert_sae!(SAE);
        transmute(vrangeps_512(
            a.as_f32x16(),
            b.as_f32x16(),
            IMM8,
            src.as_f32x16(),
            k,
            SAE,
        ))
    }
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// single-precision (32-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_range_round_ps&ig_expand=5212)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangeps, IMM8 = 5, SAE = 8))]
#[rustc_legacy_const_generics(3, 4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_range_round_ps<const IMM8: i32, const SAE: i32>(
    k: __mmask16,
    a: __m512,
    b: __m512,
) -> __m512 {
    static_assert_uimm_bits!(IMM8, 4);
    static_assert_sae!(SAE);
    _mm512_mask_range_round_ps::<IMM8, SAE>(_mm512_setzero_ps(), k, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_range_ps&ig_expand=5201)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vrangeps, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_range_ps<const IMM8: i32>(a: __m128, b: __m128) -> __m128 {
    static_assert_uimm_bits!(IMM8, 4);
    _mm_mask_range_ps::<IMM8>(_mm_setzero_ps(), 0xff, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// single-precision (32-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src to dst if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_range_ps&ig_expand=5199)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vrangeps, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_range_ps<const IMM8: i32>(
    src: __m128,
    k: __mmask8,
    a: __m128,
    b: __m128,
) -> __m128 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 4);
        transmute(vrangeps_128(
            a.as_f32x4(),
            b.as_f32x4(),
            IMM8,
            src.as_f32x4(),
            k,
        ))
    }
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// single-precision (32-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_range_ps&ig_expand=5200)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vrangeps, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_range_ps<const IMM8: i32>(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    static_assert_uimm_bits!(IMM8, 4);
    _mm_mask_range_ps::<IMM8>(_mm_setzero_ps(), k, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_range_ps&ig_expand=5204)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vrangeps, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_range_ps<const IMM8: i32>(a: __m256, b: __m256) -> __m256 {
    static_assert_uimm_bits!(IMM8, 4);
    _mm256_mask_range_ps::<IMM8>(_mm256_setzero_ps(), 0xff, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// single-precision (32-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src to dst if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_range_ps&ig_expand=5202)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vrangeps, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_range_ps<const IMM8: i32>(
    src: __m256,
    k: __mmask8,
    a: __m256,
    b: __m256,
) -> __m256 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 4);
        transmute(vrangeps_256(
            a.as_f32x8(),
            b.as_f32x8(),
            IMM8,
            src.as_f32x8(),
            k,
        ))
    }
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// single-precision (32-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_range_ps&ig_expand=5203)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vrangeps, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_range_ps<const IMM8: i32>(k: __mmask8, a: __m256, b: __m256) -> __m256 {
    static_assert_uimm_bits!(IMM8, 4);
    _mm256_mask_range_ps::<IMM8>(_mm256_setzero_ps(), k, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_range_ps&ig_expand=5207)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangeps, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_range_ps<const IMM8: i32>(a: __m512, b: __m512) -> __m512 {
    static_assert_uimm_bits!(IMM8, 4);
    _mm512_mask_range_ps::<IMM8>(_mm512_setzero_ps(), 0xffff, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// single-precision (32-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src to dst if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_range_ps&ig_expand=5205)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangeps, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_range_ps<const IMM8: i32>(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m512,
) -> __m512 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 4);
        transmute(vrangeps_512(
            a.as_f32x16(),
            b.as_f32x16(),
            IMM8,
            src.as_f32x16(),
            k,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for packed
/// single-precision (32-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out if the corresponding mask bit is not set).
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_range_ps&ig_expand=5206)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangeps, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_range_ps<const IMM8: i32>(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    static_assert_uimm_bits!(IMM8, 4);
    _mm512_mask_range_ps::<IMM8>(_mm512_setzero_ps(), k, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for the lower
/// double-precision (64-bit) floating-point element in a and b, store the result in the lower element
/// of dst, and copy the upper element from a to the upper element of dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_range_round_sd&ig_expand=5216)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangesd, IMM8 = 5, SAE = 8))]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_range_round_sd<const IMM8: i32, const SAE: i32>(a: __m128d, b: __m128d) -> __m128d {
    static_assert_uimm_bits!(IMM8, 4);
    static_assert_sae!(SAE);
    _mm_mask_range_round_sd::<IMM8, SAE>(_mm_setzero_pd(), 0xff, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for the lower
/// double-precision (64-bit) floating-point element in a and b, store the result in the lower element
/// of dst using writemask k (the element is copied from src when mask bit 0 is not set), and copy the
/// upper element from a to the upper element of dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_range_round_sd&ig_expand=5214)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangesd, IMM8 = 5, SAE = 8))]
#[rustc_legacy_const_generics(4, 5)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_range_round_sd<const IMM8: i32, const SAE: i32>(
    src: __m128d,
    k: __mmask8,
    a: __m128d,
    b: __m128d,
) -> __m128d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 4);
        static_assert_sae!(SAE);
        transmute(vrangesd(
            a.as_f64x2(),
            b.as_f64x2(),
            src.as_f64x2(),
            k,
            IMM8,
            SAE,
        ))
    }
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for the lower
/// double-precision (64-bit) floating-point element in a and b, store the result in the lower element
/// of dst using zeromask k (the element is zeroed out when mask bit 0 is not set), and copy the upper
/// element from a to the upper element of dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_range_round_sd&ig_expand=5215)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangesd, IMM8 = 5, SAE = 8))]
#[rustc_legacy_const_generics(3, 4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_range_round_sd<const IMM8: i32, const SAE: i32>(
    k: __mmask8,
    a: __m128d,
    b: __m128d,
) -> __m128d {
    static_assert_uimm_bits!(IMM8, 4);
    static_assert_sae!(SAE);
    _mm_mask_range_round_sd::<IMM8, SAE>(_mm_setzero_pd(), k, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for the lower
/// double-precision (64-bit) floating-point element in a and b, store the result in the lower element
/// of dst using writemask k (the element is copied from src when mask bit 0 is not set), and copy the
/// upper element from a to the upper element of dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_range_sd&ig_expand=5220)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangesd, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_range_sd<const IMM8: i32>(
    src: __m128d,
    k: __mmask8,
    a: __m128d,
    b: __m128d,
) -> __m128d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 4);
        transmute(vrangesd(
            a.as_f64x2(),
            b.as_f64x2(),
            src.as_f64x2(),
            k,
            IMM8,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for the lower
/// double-precision (64-bit) floating-point element in a and b, store the result in the lower element
/// of dst using zeromask k (the element is zeroed out when mask bit 0 is not set), and copy the upper
/// element from a to the upper element of dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_range_sd&ig_expand=5221)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangesd, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_range_sd<const IMM8: i32>(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    static_assert_uimm_bits!(IMM8, 4);
    _mm_mask_range_sd::<IMM8>(_mm_setzero_pd(), k, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for the lower
/// single-precision (32-bit) floating-point element in a and b, store the result in the lower element
/// of dst, and copy the upper 3 packed elements from a to the upper elements of dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_range_round_ss&ig_expand=5219)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangess, IMM8 = 5, SAE = 8))]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_range_round_ss<const IMM8: i32, const SAE: i32>(a: __m128, b: __m128) -> __m128 {
    static_assert_uimm_bits!(IMM8, 4);
    static_assert_sae!(SAE);
    _mm_mask_range_round_ss::<IMM8, SAE>(_mm_setzero_ps(), 0xff, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for the lower
/// single-precision (32-bit) floating-point element in a and b, store the result in the lower element
/// of dst using writemask k (the element is copied from src when mask bit 0 is not set), and copy the
/// upper 3 packed elements from a to the upper elements of dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_range_round_ss&ig_expand=5217)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangess, IMM8 = 5, SAE = 8))]
#[rustc_legacy_const_generics(4, 5)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_range_round_ss<const IMM8: i32, const SAE: i32>(
    src: __m128,
    k: __mmask8,
    a: __m128,
    b: __m128,
) -> __m128 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 4);
        static_assert_sae!(SAE);
        transmute(vrangess(
            a.as_f32x4(),
            b.as_f32x4(),
            src.as_f32x4(),
            k,
            IMM8,
            SAE,
        ))
    }
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for the lower
/// single-precision (32-bit) floating-point element in a and b, store the result in the lower element
/// of dst using zeromask k (the element is zeroed out when mask bit 0 is not set), and copy the upper
/// 3 packed elements from a to the upper elements of dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_range_round_ss&ig_expand=5218)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangess, IMM8 = 5, SAE = 8))]
#[rustc_legacy_const_generics(3, 4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_range_round_ss<const IMM8: i32, const SAE: i32>(
    k: __mmask8,
    a: __m128,
    b: __m128,
) -> __m128 {
    static_assert_uimm_bits!(IMM8, 4);
    static_assert_sae!(SAE);
    _mm_mask_range_round_ss::<IMM8, SAE>(_mm_setzero_ps(), k, a, b)
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for the lower
/// single-precision (32-bit) floating-point element in a and b, store the result in the lower element
/// of dst using writemask k (the element is copied from src when mask bit 0 is not set), and copy the
/// upper 3 packed elements from a to the upper elements of dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_range_ss&ig_expand=5222)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangess, IMM8 = 5))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_range_ss<const IMM8: i32>(
    src: __m128,
    k: __mmask8,
    a: __m128,
    b: __m128,
) -> __m128 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 4);
        transmute(vrangess(
            a.as_f32x4(),
            b.as_f32x4(),
            src.as_f32x4(),
            k,
            IMM8,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
}

/// Calculate the max, min, absolute max, or absolute min (depending on control in imm8) for the lower
/// single-precision (32-bit) floating-point element in a and b, store the result in the lower element
/// of dst using zeromask k (the element is zeroed out when mask bit 0 is not set), and copy the upper
/// 3 packed elements from a to the upper elements of dst.
/// Lower 2 bits of IMM8 specifies the operation control:
///     00 = min, 01 = max, 10 = absolute min, 11 = absolute max.
/// Upper 2 bits of IMM8 specifies the sign control:
///     00 = sign from a, 01 = sign from compare result, 10 = clear sign bit, 11 = set sign bit.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_range_ss&ig_expand=5223)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vrangess, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_range_ss<const IMM8: i32>(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    static_assert_uimm_bits!(IMM8, 4);
    _mm_mask_range_ss::<IMM8>(_mm_setzero_ps(), k, a, b)
}

// Reduce

/// Extract the reduced argument of packed double-precision (64-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_reduce_round_pd&ig_expand=5438)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducepd, IMM8 = 0, SAE = 8))]
#[rustc_legacy_const_generics(1, 2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_reduce_round_pd<const IMM8: i32, const SAE: i32>(a: __m512d) -> __m512d {
    static_assert_uimm_bits!(IMM8, 8);
    static_assert_sae!(SAE);
    _mm512_mask_reduce_round_pd::<IMM8, SAE>(_mm512_undefined_pd(), 0xff, a)
}

/// Extract the reduced argument of packed double-precision (64-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using writemask k (elements are
/// copied from src to dst if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_reduce_round_pd&ig_expand=5436)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducepd, IMM8 = 0, SAE = 8))]
#[rustc_legacy_const_generics(3, 4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_reduce_round_pd<const IMM8: i32, const SAE: i32>(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
) -> __m512d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        static_assert_sae!(SAE);
        transmute(vreducepd_512(a.as_f64x8(), IMM8, src.as_f64x8(), k, SAE))
    }
}

/// Extract the reduced argument of packed double-precision (64-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using zeromask k (elements are
/// zeroed out if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_reduce_round_pd&ig_expand=5437)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducepd, IMM8 = 0, SAE = 8))]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_reduce_round_pd<const IMM8: i32, const SAE: i32>(
    k: __mmask8,
    a: __m512d,
) -> __m512d {
    static_assert_uimm_bits!(IMM8, 8);
    static_assert_sae!(SAE);
    _mm512_mask_reduce_round_pd::<IMM8, SAE>(_mm512_setzero_pd(), k, a)
}

/// Extract the reduced argument of packed double-precision (64-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_pd&ig_expand=5411)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vreducepd, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_pd<const IMM8: i32>(a: __m128d) -> __m128d {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_mask_reduce_pd::<IMM8>(_mm_undefined_pd(), 0xff, a)
}

/// Extract the reduced argument of packed double-precision (64-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using writemask k (elements are
/// copied from src to dst if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_pd&ig_expand=5409)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vreducepd, IMM8 = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_pd<const IMM8: i32>(src: __m128d, k: __mmask8, a: __m128d) -> __m128d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(vreducepd_128(a.as_f64x2(), IMM8, src.as_f64x2(), k))
    }
}

/// Extract the reduced argument of packed double-precision (64-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using zeromask k (elements are
/// zeroed out if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_reduce_pd&ig_expand=5410)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vreducepd, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_reduce_pd<const IMM8: i32>(k: __mmask8, a: __m128d) -> __m128d {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_mask_reduce_pd::<IMM8>(_mm_setzero_pd(), k, a)
}

/// Extract the reduced argument of packed double-precision (64-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_pd&ig_expand=5414)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vreducepd, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_pd<const IMM8: i32>(a: __m256d) -> __m256d {
    static_assert_uimm_bits!(IMM8, 8);
    _mm256_mask_reduce_pd::<IMM8>(_mm256_undefined_pd(), 0xff, a)
}

/// Extract the reduced argument of packed double-precision (64-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using writemask k (elements are
/// copied from src to dst if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_pd&ig_expand=5412)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vreducepd, IMM8 = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_pd<const IMM8: i32>(src: __m256d, k: __mmask8, a: __m256d) -> __m256d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(vreducepd_256(a.as_f64x4(), IMM8, src.as_f64x4(), k))
    }
}

/// Extract the reduced argument of packed double-precision (64-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using zeromask k (elements are
/// zeroed out if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_reduce_pd&ig_expand=5413)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vreducepd, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_reduce_pd<const IMM8: i32>(k: __mmask8, a: __m256d) -> __m256d {
    static_assert_uimm_bits!(IMM8, 8);
    _mm256_mask_reduce_pd::<IMM8>(_mm256_setzero_pd(), k, a)
}

/// Extract the reduced argument of packed double-precision (64-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_reduce_pd&ig_expand=5417)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducepd, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_reduce_pd<const IMM8: i32>(a: __m512d) -> __m512d {
    static_assert_uimm_bits!(IMM8, 8);
    _mm512_mask_reduce_pd::<IMM8>(_mm512_undefined_pd(), 0xff, a)
}

/// Extract the reduced argument of packed double-precision (64-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using writemask k (elements are
/// copied from src to dst if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_reduce_pd&ig_expand=5415)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducepd, IMM8 = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_reduce_pd<const IMM8: i32>(src: __m512d, k: __mmask8, a: __m512d) -> __m512d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(vreducepd_512(
            a.as_f64x8(),
            IMM8,
            src.as_f64x8(),
            k,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
}

/// Extract the reduced argument of packed double-precision (64-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using zeromask k (elements are
/// zeroed out if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_reduce_pd&ig_expand=5416)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducepd, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_reduce_pd<const IMM8: i32>(k: __mmask8, a: __m512d) -> __m512d {
    static_assert_uimm_bits!(IMM8, 8);
    _mm512_mask_reduce_pd::<IMM8>(_mm512_setzero_pd(), k, a)
}

/// Extract the reduced argument of packed single-precision (32-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_reduce_round_ps&ig_expand=5444)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreduceps, IMM8 = 0, SAE = 8))]
#[rustc_legacy_const_generics(1, 2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_reduce_round_ps<const IMM8: i32, const SAE: i32>(a: __m512) -> __m512 {
    static_assert_uimm_bits!(IMM8, 8);
    static_assert_sae!(SAE);
    _mm512_mask_reduce_round_ps::<IMM8, SAE>(_mm512_undefined_ps(), 0xffff, a)
}

/// Extract the reduced argument of packed single-precision (32-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using writemask k (elements are
/// copied from src to dst if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_reduce_round_ps&ig_expand=5442)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreduceps, IMM8 = 0, SAE = 8))]
#[rustc_legacy_const_generics(3, 4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_reduce_round_ps<const IMM8: i32, const SAE: i32>(
    src: __m512,
    k: __mmask16,
    a: __m512,
) -> __m512 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        static_assert_sae!(SAE);
        transmute(vreduceps_512(a.as_f32x16(), IMM8, src.as_f32x16(), k, SAE))
    }
}

/// Extract the reduced argument of packed single-precision (32-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using zeromask k (elements are
/// zeroed out if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_reduce_round_ps&ig_expand=5443)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreduceps, IMM8 = 0, SAE = 8))]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_reduce_round_ps<const IMM8: i32, const SAE: i32>(
    k: __mmask16,
    a: __m512,
) -> __m512 {
    static_assert_uimm_bits!(IMM8, 8);
    static_assert_sae!(SAE);
    _mm512_mask_reduce_round_ps::<IMM8, SAE>(_mm512_setzero_ps(), k, a)
}

/// Extract the reduced argument of packed single-precision (32-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_ps&ig_expand=5429)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vreduceps, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_ps<const IMM8: i32>(a: __m128) -> __m128 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_mask_reduce_ps::<IMM8>(_mm_undefined_ps(), 0xff, a)
}

/// Extract the reduced argument of packed single-precision (32-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using writemask k (elements are
/// copied from src to dst if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_ps&ig_expand=5427)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vreduceps, IMM8 = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_ps<const IMM8: i32>(src: __m128, k: __mmask8, a: __m128) -> __m128 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(vreduceps_128(a.as_f32x4(), IMM8, src.as_f32x4(), k))
    }
}

/// Extract the reduced argument of packed single-precision (32-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using zeromask k (elements are
/// zeroed out if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_reduce_ps&ig_expand=5428)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vreduceps, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_reduce_ps<const IMM8: i32>(k: __mmask8, a: __m128) -> __m128 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_mask_reduce_ps::<IMM8>(_mm_setzero_ps(), k, a)
}

/// Extract the reduced argument of packed single-precision (32-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_ps&ig_expand=5432)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vreduceps, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_ps<const IMM8: i32>(a: __m256) -> __m256 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm256_mask_reduce_ps::<IMM8>(_mm256_undefined_ps(), 0xff, a)
}

/// Extract the reduced argument of packed single-precision (32-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using writemask k (elements are
/// copied from src to dst if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_ps&ig_expand=5430)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vreduceps, IMM8 = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_ps<const IMM8: i32>(src: __m256, k: __mmask8, a: __m256) -> __m256 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(vreduceps_256(a.as_f32x8(), IMM8, src.as_f32x8(), k))
    }
}

/// Extract the reduced argument of packed single-precision (32-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using zeromask k (elements are
/// zeroed out if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_reduce_ps&ig_expand=5431)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vreduceps, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_maskz_reduce_ps<const IMM8: i32>(k: __mmask8, a: __m256) -> __m256 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm256_mask_reduce_ps::<IMM8>(_mm256_setzero_ps(), k, a)
}

/// Extract the reduced argument of packed single-precision (32-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_reduce_ps&ig_expand=5435)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreduceps, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_reduce_ps<const IMM8: i32>(a: __m512) -> __m512 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm512_mask_reduce_ps::<IMM8>(_mm512_undefined_ps(), 0xffff, a)
}

/// Extract the reduced argument of packed single-precision (32-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using writemask k (elements are
/// copied from src to dst if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_reduce_ps&ig_expand=5433)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreduceps, IMM8 = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_reduce_ps<const IMM8: i32>(src: __m512, k: __mmask16, a: __m512) -> __m512 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(vreduceps_512(
            a.as_f32x16(),
            IMM8,
            src.as_f32x16(),
            k,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
}

/// Extract the reduced argument of packed single-precision (32-bit) floating-point elements in a by
/// the number of bits specified by imm8, and store the results in dst using zeromask k (elements are
/// zeroed out if the corresponding mask bit is not set).
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_reduce_ps&ig_expand=5434)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreduceps, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_maskz_reduce_ps<const IMM8: i32>(k: __mmask16, a: __m512) -> __m512 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm512_mask_reduce_ps::<IMM8>(_mm512_setzero_ps(), k, a)
}

/// Extract the reduced argument of the lower double-precision (64-bit) floating-point element in b
/// by the number of bits specified by imm8, store the result in the lower element of dst, and copy
/// the upper element from a to the upper element of dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_round_sd&ig_expand=5447)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducesd, IMM8 = 0, SAE = 8))]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_round_sd<const IMM8: i32, const SAE: i32>(a: __m128d, b: __m128d) -> __m128d {
    static_assert_uimm_bits!(IMM8, 8);
    static_assert_sae!(SAE);
    _mm_mask_reduce_round_sd::<IMM8, SAE>(_mm_undefined_pd(), 0xff, a, b)
}

/// Extract the reduced argument of the lower double-precision (64-bit) floating-point element in b
/// by the number of bits specified by imm8, store the result in the lower element of dst using writemask
/// k (the element is copied from src when mask bit 0 is not set), and copy the upper element from a
/// to the upper element of dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_round_sd&ig_expand=5445)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducesd, IMM8 = 0, SAE = 8))]
#[rustc_legacy_const_generics(4, 5)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_round_sd<const IMM8: i32, const SAE: i32>(
    src: __m128d,
    k: __mmask8,
    a: __m128d,
    b: __m128d,
) -> __m128d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        static_assert_sae!(SAE);
        transmute(vreducesd(
            a.as_f64x2(),
            b.as_f64x2(),
            src.as_f64x2(),
            k,
            IMM8,
            SAE,
        ))
    }
}

/// Extract the reduced argument of the lower double-precision (64-bit) floating-point element in b
/// by the number of bits specified by imm8, store the result in the lower element of dst using zeromask
/// k (the element is zeroed out when mask bit 0 is not set), and copy the upper element from a
/// to the upper element of dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_reduce_round_sd&ig_expand=5446)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducesd, IMM8 = 0, SAE = 8))]
#[rustc_legacy_const_generics(3, 4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_reduce_round_sd<const IMM8: i32, const SAE: i32>(
    k: __mmask8,
    a: __m128d,
    b: __m128d,
) -> __m128d {
    static_assert_uimm_bits!(IMM8, 8);
    static_assert_sae!(SAE);
    _mm_mask_reduce_round_sd::<IMM8, SAE>(_mm_setzero_pd(), k, a, b)
}

/// Extract the reduced argument of the lower double-precision (64-bit) floating-point element in b
/// by the number of bits specified by imm8, store the result in the lower element of dst using, and
/// copy the upper element from a.
/// to the upper element of dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_sd&ig_expand=5456)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducesd, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_sd<const IMM8: i32>(a: __m128d, b: __m128d) -> __m128d {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_mask_reduce_sd::<IMM8>(_mm_undefined_pd(), 0xff, a, b)
}

/// Extract the reduced argument of the lower double-precision (64-bit) floating-point element in b
/// by the number of bits specified by imm8, store the result in the lower element of dst using writemask
/// k (the element is copied from src when mask bit 0 is not set), and copy the upper element from a
/// to the upper element of dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_sd&ig_expand=5454)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducesd, IMM8 = 0))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_sd<const IMM8: i32>(
    src: __m128d,
    k: __mmask8,
    a: __m128d,
    b: __m128d,
) -> __m128d {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(vreducesd(
            a.as_f64x2(),
            b.as_f64x2(),
            src.as_f64x2(),
            k,
            IMM8,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
}

/// Extract the reduced argument of the lower double-precision (64-bit) floating-point element in b
/// by the number of bits specified by imm8, store the result in the lower element of dst using zeromask
/// k (the element is zeroed out when mask bit 0 is not set), and copy the upper element from a
/// to the upper element of dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_reduce_sd&ig_expand=5455)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducesd, IMM8 = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_reduce_sd<const IMM8: i32>(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_mask_reduce_sd::<IMM8>(_mm_setzero_pd(), k, a, b)
}

/// Extract the reduced argument of the lower single-precision (32-bit) floating-point element in b
/// by the number of bits specified by imm8, store the result in the lower element of dst, and copy
/// the upper element from a.
/// to the upper element of dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_round_ss&ig_expand=5453)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducess, IMM8 = 0, SAE = 8))]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_round_ss<const IMM8: i32, const SAE: i32>(a: __m128, b: __m128) -> __m128 {
    static_assert_uimm_bits!(IMM8, 8);
    static_assert_sae!(SAE);
    _mm_mask_reduce_round_ss::<IMM8, SAE>(_mm_undefined_ps(), 0xff, a, b)
}

/// Extract the reduced argument of the lower single-precision (32-bit) floating-point element in b
/// by the number of bits specified by imm8, store the result in the lower element of dst using writemask
/// k (the element is copied from src when mask bit 0 is not set), and copy the upper element from a.
/// to the upper element of dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_round_ss&ig_expand=5451)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducess, IMM8 = 0, SAE = 8))]
#[rustc_legacy_const_generics(4, 5)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_round_ss<const IMM8: i32, const SAE: i32>(
    src: __m128,
    k: __mmask8,
    a: __m128,
    b: __m128,
) -> __m128 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        static_assert_sae!(SAE);
        transmute(vreducess(
            a.as_f32x4(),
            b.as_f32x4(),
            src.as_f32x4(),
            k,
            IMM8,
            SAE,
        ))
    }
}

/// Extract the reduced argument of the lower single-precision (32-bit) floating-point element in b
/// by the number of bits specified by imm8, store the result in the lower element of dst using zeromask
/// k (the element is zeroed out when mask bit 0 is not set), and copy the upper element from a.
/// to the upper element of dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_reduce_round_ss&ig_expand=5452)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducess, IMM8 = 0, SAE = 8))]
#[rustc_legacy_const_generics(3, 4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_reduce_round_ss<const IMM8: i32, const SAE: i32>(
    k: __mmask8,
    a: __m128,
    b: __m128,
) -> __m128 {
    static_assert_uimm_bits!(IMM8, 8);
    static_assert_sae!(SAE);
    _mm_mask_reduce_round_ss::<IMM8, SAE>(_mm_setzero_ps(), k, a, b)
}

/// Extract the reduced argument of the lower single-precision (32-bit) floating-point element in b
/// by the number of bits specified by imm8, store the result in the lower element of dst, and copy
/// the upper element from a.
/// to the upper element of dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_ss&ig_expand=5462)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducess, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_ss<const IMM8: i32>(a: __m128, b: __m128) -> __m128 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_mask_reduce_ss::<IMM8>(_mm_undefined_ps(), 0xff, a, b)
}

/// Extract the reduced argument of the lower single-precision (32-bit) floating-point element in b
/// by the number of bits specified by imm8, store the result in the lower element of dst using writemask
/// k (the element is copied from src when mask bit 0 is not set), and copy the upper element from a.
/// to the upper element of dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_ss&ig_expand=5460)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducess, IMM8 = 0))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_ss<const IMM8: i32>(
    src: __m128,
    k: __mmask8,
    a: __m128,
    b: __m128,
) -> __m128 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(vreducess(
            a.as_f32x4(),
            b.as_f32x4(),
            src.as_f32x4(),
            k,
            IMM8,
            _MM_FROUND_CUR_DIRECTION,
        ))
    }
}

/// Extract the reduced argument of the lower single-precision (32-bit) floating-point element in b
/// by the number of bits specified by imm8, store the result in the lower element of dst using zeromask
/// k (the element is zeroed out when mask bit 0 is not set), and copy the upper element from a.
/// to the upper element of dst.
/// Rounding is done according to the imm8 parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] : round to nearest
/// * [`_MM_FROUND_TO_NEG_INF`] : round down
/// * [`_MM_FROUND_TO_POS_INF`] : round up
/// * [`_MM_FROUND_TO_ZERO`] : truncate
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_reduce_ss&ig_expand=5461)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vreducess, IMM8 = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_maskz_reduce_ss<const IMM8: i32>(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_mask_reduce_ss::<IMM8>(_mm_setzero_ps(), k, a, b)
}

// FP-Class

/// Test packed double-precision (64-bit) floating-point elements in a for special categories specified
/// by imm8, and store the results in mask vector k.
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fpclass_pd_mask&ig_expand=3493)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vfpclasspd, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_fpclass_pd_mask<const IMM8: i32>(a: __m128d) -> __mmask8 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_mask_fpclass_pd_mask::<IMM8>(0xff, a)
}

/// Test packed double-precision (64-bit) floating-point elements in a for special categories specified
/// by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the
/// corresponding mask bit is not set).
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_fpclass_pd_mask&ig_expand=3494)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vfpclasspd, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_fpclass_pd_mask<const IMM8: i32>(k1: __mmask8, a: __m128d) -> __mmask8 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(vfpclasspd_128(a.as_f64x2(), IMM8, k1))
    }
}

/// Test packed double-precision (64-bit) floating-point elements in a for special categories specified
/// by imm8, and store the results in mask vector k.
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fpclass_pd_mask&ig_expand=3495)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vfpclasspd, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_fpclass_pd_mask<const IMM8: i32>(a: __m256d) -> __mmask8 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm256_mask_fpclass_pd_mask::<IMM8>(0xff, a)
}

/// Test packed double-precision (64-bit) floating-point elements in a for special categories specified
/// by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the
/// corresponding mask bit is not set).
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_fpclass_pd_mask&ig_expand=3496)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vfpclasspd, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_fpclass_pd_mask<const IMM8: i32>(k1: __mmask8, a: __m256d) -> __mmask8 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(vfpclasspd_256(a.as_f64x4(), IMM8, k1))
    }
}

/// Test packed double-precision (64-bit) floating-point elements in a for special categories specified
/// by imm8, and store the results in mask vector k.
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_fpclass_pd_mask&ig_expand=3497)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vfpclasspd, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_fpclass_pd_mask<const IMM8: i32>(a: __m512d) -> __mmask8 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm512_mask_fpclass_pd_mask::<IMM8>(0xff, a)
}

/// Test packed double-precision (64-bit) floating-point elements in a for special categories specified
/// by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the
/// corresponding mask bit is not set).
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_fpclass_pd_mask&ig_expand=3498)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vfpclasspd, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_fpclass_pd_mask<const IMM8: i32>(k1: __mmask8, a: __m512d) -> __mmask8 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(vfpclasspd_512(a.as_f64x8(), IMM8, k1))
    }
}

/// Test packed single-precision (32-bit) floating-point elements in a for special categories specified
/// by imm8, and store the results in mask vector k.
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fpclass_ps_mask&ig_expand=3505)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vfpclassps, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_fpclass_ps_mask<const IMM8: i32>(a: __m128) -> __mmask8 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_mask_fpclass_ps_mask::<IMM8>(0xff, a)
}

/// Test packed single-precision (32-bit) floating-point elements in a for special categories specified
/// by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the
/// corresponding mask bit is not set).
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_fpclass_ps_mask&ig_expand=3506)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vfpclassps, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_fpclass_ps_mask<const IMM8: i32>(k1: __mmask8, a: __m128) -> __mmask8 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(vfpclassps_128(a.as_f32x4(), IMM8, k1))
    }
}

/// Test packed single-precision (32-bit) floating-point elements in a for special categories specified
/// by imm8, and store the results in mask vector k.
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fpclass_ps_mask&ig_expand=3507)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vfpclassps, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_fpclass_ps_mask<const IMM8: i32>(a: __m256) -> __mmask8 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm256_mask_fpclass_ps_mask::<IMM8>(0xff, a)
}

/// Test packed single-precision (32-bit) floating-point elements in a for special categories specified
/// by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the
/// corresponding mask bit is not set).
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_fpclass_ps_mask&ig_expand=3508)
#[inline]
#[target_feature(enable = "avx512dq,avx512vl")]
#[cfg_attr(test, assert_instr(vfpclassps, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_fpclass_ps_mask<const IMM8: i32>(k1: __mmask8, a: __m256) -> __mmask8 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(vfpclassps_256(a.as_f32x8(), IMM8, k1))
    }
}

/// Test packed single-precision (32-bit) floating-point elements in a for special categories specified
/// by imm8, and store the results in mask vector k.
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_fpclass_ps_mask&ig_expand=3509)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vfpclassps, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_fpclass_ps_mask<const IMM8: i32>(a: __m512) -> __mmask16 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm512_mask_fpclass_ps_mask::<IMM8>(0xffff, a)
}

/// Test packed single-precision (32-bit) floating-point elements in a for special categories specified
/// by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the
/// corresponding mask bit is not set).
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_fpclass_ps_mask&ig_expand=3510)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vfpclassps, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm512_mask_fpclass_ps_mask<const IMM8: i32>(k1: __mmask16, a: __m512) -> __mmask16 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(vfpclassps_512(a.as_f32x16(), IMM8, k1))
    }
}

/// Test the lower double-precision (64-bit) floating-point element in a for special categories specified
/// by imm8, and store the results in mask vector k.
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fpclass_sd_mask&ig_expand=3511)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vfpclasssd, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_fpclass_sd_mask<const IMM8: i32>(a: __m128d) -> __mmask8 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_mask_fpclass_sd_mask::<IMM8>(0xff, a)
}

/// Test the lower double-precision (64-bit) floating-point element in a for special categories specified
/// by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the
/// corresponding mask bit is not set).
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_fpclass_sd_mask&ig_expand=3512)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vfpclasssd, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_fpclass_sd_mask<const IMM8: i32>(k1: __mmask8, a: __m128d) -> __mmask8 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        vfpclasssd(a.as_f64x2(), IMM8, k1)
    }
}

/// Test the lower single-precision (32-bit) floating-point element in a for special categories specified
/// by imm8, and store the results in mask vector k.
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_fpclass_ss_mask&ig_expand=3515)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vfpclassss, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_fpclass_ss_mask<const IMM8: i32>(a: __m128) -> __mmask8 {
    static_assert_uimm_bits!(IMM8, 8);
    _mm_mask_fpclass_ss_mask::<IMM8>(0xff, a)
}

/// Test the lower single-precision (32-bit) floating-point element in a for special categories specified
/// by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the
/// corresponding mask bit is not set).
/// imm can be a combination of:
///
///     - 0x01 // QNaN
///     - 0x02 // Positive Zero
///     - 0x04 // Negative Zero
///     - 0x08 // Positive Infinity
///     - 0x10 // Negative Infinity
///     - 0x20 // Denormal
///     - 0x40 // Negative
///     - 0x80 // SNaN
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_fpclass_ss_mask&ig_expand=3516)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vfpclassss, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_fpclass_ss_mask<const IMM8: i32>(k1: __mmask8, a: __m128) -> __mmask8 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        vfpclassss(a.as_f32x4(), IMM8, k1)
    }
}

#[allow(improper_ctypes)]
unsafe extern "C" {
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

    #[link_name = "llvm.x86.avx512.mask.range.pd.128"]
    fn vrangepd_128(a: f64x2, b: f64x2, imm8: i32, src: f64x2, k: __mmask8) -> f64x2;
    #[link_name = "llvm.x86.avx512.mask.range.pd.256"]
    fn vrangepd_256(a: f64x4, b: f64x4, imm8: i32, src: f64x4, k: __mmask8) -> f64x4;
    #[link_name = "llvm.x86.avx512.mask.range.pd.512"]
    fn vrangepd_512(a: f64x8, b: f64x8, imm8: i32, src: f64x8, k: __mmask8, sae: i32) -> f64x8;

    #[link_name = "llvm.x86.avx512.mask.range.ps.128"]
    fn vrangeps_128(a: f32x4, b: f32x4, imm8: i32, src: f32x4, k: __mmask8) -> f32x4;
    #[link_name = "llvm.x86.avx512.mask.range.ps.256"]
    fn vrangeps_256(a: f32x8, b: f32x8, imm8: i32, src: f32x8, k: __mmask8) -> f32x8;
    #[link_name = "llvm.x86.avx512.mask.range.ps.512"]
    fn vrangeps_512(a: f32x16, b: f32x16, imm8: i32, src: f32x16, k: __mmask16, sae: i32)
    -> f32x16;

    #[link_name = "llvm.x86.avx512.mask.range.sd"]
    fn vrangesd(a: f64x2, b: f64x2, src: f64x2, k: __mmask8, imm8: i32, sae: i32) -> f64x2;
    #[link_name = "llvm.x86.avx512.mask.range.ss"]
    fn vrangess(a: f32x4, b: f32x4, src: f32x4, k: __mmask8, imm8: i32, sae: i32) -> f32x4;

    #[link_name = "llvm.x86.avx512.mask.reduce.pd.128"]
    fn vreducepd_128(a: f64x2, imm8: i32, src: f64x2, k: __mmask8) -> f64x2;
    #[link_name = "llvm.x86.avx512.mask.reduce.pd.256"]
    fn vreducepd_256(a: f64x4, imm8: i32, src: f64x4, k: __mmask8) -> f64x4;
    #[link_name = "llvm.x86.avx512.mask.reduce.pd.512"]
    fn vreducepd_512(a: f64x8, imm8: i32, src: f64x8, k: __mmask8, sae: i32) -> f64x8;

    #[link_name = "llvm.x86.avx512.mask.reduce.ps.128"]
    fn vreduceps_128(a: f32x4, imm8: i32, src: f32x4, k: __mmask8) -> f32x4;
    #[link_name = "llvm.x86.avx512.mask.reduce.ps.256"]
    fn vreduceps_256(a: f32x8, imm8: i32, src: f32x8, k: __mmask8) -> f32x8;
    #[link_name = "llvm.x86.avx512.mask.reduce.ps.512"]
    fn vreduceps_512(a: f32x16, imm8: i32, src: f32x16, k: __mmask16, sae: i32) -> f32x16;

    #[link_name = "llvm.x86.avx512.mask.reduce.sd"]
    fn vreducesd(a: f64x2, b: f64x2, src: f64x2, k: __mmask8, imm8: i32, sae: i32) -> f64x2;
    #[link_name = "llvm.x86.avx512.mask.reduce.ss"]
    fn vreducess(a: f32x4, b: f32x4, src: f32x4, k: __mmask8, imm8: i32, sae: i32) -> f32x4;

    #[link_name = "llvm.x86.avx512.mask.fpclass.pd.128"]
    fn vfpclasspd_128(a: f64x2, imm8: i32, k: __mmask8) -> __mmask8;
    #[link_name = "llvm.x86.avx512.mask.fpclass.pd.256"]
    fn vfpclasspd_256(a: f64x4, imm8: i32, k: __mmask8) -> __mmask8;
    #[link_name = "llvm.x86.avx512.mask.fpclass.pd.512"]
    fn vfpclasspd_512(a: f64x8, imm8: i32, k: __mmask8) -> __mmask8;

    #[link_name = "llvm.x86.avx512.mask.fpclass.ps.128"]
    fn vfpclassps_128(a: f32x4, imm8: i32, k: __mmask8) -> __mmask8;
    #[link_name = "llvm.x86.avx512.mask.fpclass.ps.256"]
    fn vfpclassps_256(a: f32x8, imm8: i32, k: __mmask8) -> __mmask8;
    #[link_name = "llvm.x86.avx512.mask.fpclass.ps.512"]
    fn vfpclassps_512(a: f32x16, imm8: i32, k: __mmask16) -> __mmask16;

    #[link_name = "llvm.x86.avx512.mask.fpclass.sd"]
    fn vfpclasssd(a: f64x2, imm8: i32, k: __mmask8) -> __mmask8;
    #[link_name = "llvm.x86.avx512.mask.fpclass.ss"]
    fn vfpclassss(a: f32x4, imm8: i32, k: __mmask8) -> __mmask8;
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
        let e = _mm256_set_ps(3., 4., 3., 4., 3., 4., 3., 4.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_broadcast_f32x2() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm256_set_ps(5., 6., 7., 8., 9., 10., 11., 12.);
        let r = _mm256_mask_broadcast_f32x2(b, 0b01101001, a);
        let e = _mm256_set_ps(5., 4., 3., 8., 3., 10., 11., 4.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_broadcast_f32x2() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm256_maskz_broadcast_f32x2(0b01101001, a);
        let e = _mm256_set_ps(0., 4., 3., 0., 3., 0., 0., 4.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_broadcast_f32x2() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm512_broadcast_f32x2(a);
        let e = _mm512_set_ps(
            3., 4., 3., 4., 3., 4., 3., 4., 3., 4., 3., 4., 3., 4., 3., 4.,
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
            5., 4., 3., 8., 3., 10., 11., 4., 13., 14., 3., 4., 3., 4., 19., 20.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_broadcast_f32x2() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let r = _mm512_maskz_broadcast_f32x2(0b0110100100111100, a);
        let e = _mm512_set_ps(
            0., 4., 3., 0., 3., 0., 0., 4., 0., 0., 3., 4., 3., 4., 0., 0.,
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
        let e = _mm_set_epi32(3, 4, 3, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let b = _mm_set_epi32(5, 6, 7, 8);
        let r = _mm_mask_broadcast_i32x2(b, 0b0110, a);
        let e = _mm_set_epi32(5, 4, 3, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let r = _mm_maskz_broadcast_i32x2(0b0110, a);
        let e = _mm_set_epi32(0, 4, 3, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let r = _mm256_broadcast_i32x2(a);
        let e = _mm256_set_epi32(3, 4, 3, 4, 3, 4, 3, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let b = _mm256_set_epi32(5, 6, 7, 8, 9, 10, 11, 12);
        let r = _mm256_mask_broadcast_i32x2(b, 0b01101001, a);
        let e = _mm256_set_epi32(5, 4, 3, 8, 3, 10, 11, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let r = _mm256_maskz_broadcast_i32x2(0b01101001, a);
        let e = _mm256_set_epi32(0, 4, 3, 0, 3, 0, 0, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let r = _mm512_broadcast_i32x2(a);
        let e = _mm512_set_epi32(3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let b = _mm512_set_epi32(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
        let r = _mm512_mask_broadcast_i32x2(b, 0b0110100100111100, a);
        let e = _mm512_set_epi32(5, 4, 3, 8, 3, 10, 11, 4, 13, 14, 3, 4, 3, 4, 19, 20);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_broadcast_i32x2() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let r = _mm512_maskz_broadcast_i32x2(0b0110100100111100, a);
        let e = _mm512_set_epi32(0, 4, 3, 0, 3, 0, 0, 4, 0, 0, 3, 4, 3, 4, 0, 0);
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
        let e = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_extractf32x8_ps() {
        let a = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let b = _mm256_set_ps(17., 18., 19., 20., 21., 22., 23., 24.);
        let r = _mm512_mask_extractf32x8_ps::<1>(b, 0b01101001, a);
        let e = _mm256_set_ps(17., 2., 3., 20., 5., 22., 23., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_extractf32x8_ps() {
        let a = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let r = _mm512_maskz_extractf32x8_ps::<1>(0b01101001, a);
        let e = _mm256_set_ps(0., 2., 3., 0., 5., 0., 0., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_extractf64x2_pd() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_extractf64x2_pd::<1>(a);
        let e = _mm_set_pd(1., 2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_extractf64x2_pd() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm_set_pd(5., 6.);
        let r = _mm256_mask_extractf64x2_pd::<1>(b, 0b01, a);
        let e = _mm_set_pd(5., 2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_extractf64x2_pd() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let r = _mm256_maskz_extractf64x2_pd::<1>(0b01, a);
        let e = _mm_set_pd(0., 2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_extractf64x2_pd() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_extractf64x2_pd::<2>(a);
        let e = _mm_set_pd(3., 4.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_extractf64x2_pd() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm_set_pd(9., 10.);
        let r = _mm512_mask_extractf64x2_pd::<2>(b, 0b01, a);
        let e = _mm_set_pd(9., 4.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_extractf64x2_pd() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_maskz_extractf64x2_pd::<2>(0b01, a);
        let e = _mm_set_pd(0., 4.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_extracti32x8_epi32() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_extracti32x8_epi32::<1>(a);
        let e = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_extracti32x8_epi32() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm256_set_epi32(17, 18, 19, 20, 21, 22, 23, 24);
        let r = _mm512_mask_extracti32x8_epi32::<1>(b, 0b01101001, a);
        let e = _mm256_set_epi32(17, 2, 3, 20, 5, 22, 23, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_extracti32x8_epi32() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_maskz_extracti32x8_epi32::<1>(0b01101001, a);
        let e = _mm256_set_epi32(0, 2, 3, 0, 5, 0, 0, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_extracti64x2_epi64() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let r = _mm256_extracti64x2_epi64::<1>(a);
        let e = _mm_set_epi64x(1, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_extracti64x2_epi64() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm_set_epi64x(5, 6);
        let r = _mm256_mask_extracti64x2_epi64::<1>(b, 0b01, a);
        let e = _mm_set_epi64x(5, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_extracti64x2_epi64() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let r = _mm256_maskz_extracti64x2_epi64::<1>(0b01, a);
        let e = _mm_set_epi64x(0, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_extracti64x2_epi64() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_extracti64x2_epi64::<2>(a);
        let e = _mm_set_epi64x(3, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_extracti64x2_epi64() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_set_epi64x(9, 10);
        let r = _mm512_mask_extracti64x2_epi64::<2>(b, 0b01, a);
        let e = _mm_set_epi64x(9, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_extracti64x2_epi64() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm512_maskz_extracti64x2_epi64::<2>(0b01, a);
        let e = _mm_set_epi64x(0, 4);
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
            17., 18., 19., 20., 21., 22., 23., 24., 9., 10., 11., 12., 13., 14., 15., 16.,
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
            25., 18., 19., 28., 21., 30., 31., 24., 33., 34., 11., 12., 13., 14., 39., 40.,
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
            0., 18., 19., 0., 21., 0., 0., 24., 0., 0., 11., 12., 13., 14., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_insertf64x2() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm_set_pd(5., 6.);
        let r = _mm256_insertf64x2::<1>(a, b);
        let e = _mm256_set_pd(5., 6., 3., 4.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_insertf64x2() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm_set_pd(5., 6.);
        let src = _mm256_set_pd(7., 8., 9., 10.);
        let r = _mm256_mask_insertf64x2::<1>(src, 0b0110, a, b);
        let e = _mm256_set_pd(7., 6., 3., 10.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_insertf64x2() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm_set_pd(5., 6.);
        let r = _mm256_maskz_insertf64x2::<1>(0b0110, a, b);
        let e = _mm256_set_pd(0., 6., 3., 0.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_insertf64x2() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm_set_pd(9., 10.);
        let r = _mm512_insertf64x2::<2>(a, b);
        let e = _mm512_set_pd(1., 2., 9., 10., 5., 6., 7., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_insertf64x2() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm_set_pd(9., 10.);
        let src = _mm512_set_pd(11., 12., 13., 14., 15., 16., 17., 18.);
        let r = _mm512_mask_insertf64x2::<2>(src, 0b01101001, a, b);
        let e = _mm512_set_pd(11., 2., 9., 14., 5., 16., 17., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_insertf64x2() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm_set_pd(9., 10.);
        let r = _mm512_maskz_insertf64x2::<2>(0b01101001, a, b);
        let e = _mm512_set_pd(0., 2., 9., 0., 5., 0., 0., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_inserti32x8() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm256_set_epi32(17, 18, 19, 20, 21, 22, 23, 24);
        let r = _mm512_inserti32x8::<1>(a, b);
        let e = _mm512_set_epi32(
            17, 18, 19, 20, 21, 22, 23, 24, 9, 10, 11, 12, 13, 14, 15, 16,
        );
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
        let e = _mm512_set_epi32(
            25, 18, 19, 28, 21, 30, 31, 24, 33, 34, 11, 12, 13, 14, 39, 40,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_inserti32x8() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm256_set_epi32(17, 18, 19, 20, 21, 22, 23, 24);
        let r = _mm512_maskz_inserti32x8::<1>(0b0110100100111100, a, b);
        let e = _mm512_set_epi32(0, 18, 19, 0, 21, 0, 0, 24, 0, 0, 11, 12, 13, 14, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_inserti64x2() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm_set_epi64x(5, 6);
        let r = _mm256_inserti64x2::<1>(a, b);
        let e = _mm256_set_epi64x(5, 6, 3, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_inserti64x2() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm_set_epi64x(5, 6);
        let src = _mm256_set_epi64x(7, 8, 9, 10);
        let r = _mm256_mask_inserti64x2::<1>(src, 0b0110, a, b);
        let e = _mm256_set_epi64x(7, 6, 3, 10);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_inserti64x2() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm_set_epi64x(5, 6);
        let r = _mm256_maskz_inserti64x2::<1>(0b0110, a, b);
        let e = _mm256_set_epi64x(0, 6, 3, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_inserti64x2() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_set_epi64x(9, 10);
        let r = _mm512_inserti64x2::<2>(a, b);
        let e = _mm512_set_epi64(1, 2, 9, 10, 5, 6, 7, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_inserti64x2() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_set_epi64x(9, 10);
        let src = _mm512_set_epi64(11, 12, 13, 14, 15, 16, 17, 18);
        let r = _mm512_mask_inserti64x2::<2>(src, 0b01101001, a, b);
        let e = _mm512_set_epi64(11, 2, 9, 14, 5, 16, 17, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_inserti64x2() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_set_epi64x(9, 10);
        let r = _mm512_maskz_inserti64x2::<2>(0b01101001, a, b);
        let e = _mm512_set_epi64(0, 2, 9, 0, 5, 0, 0, 8);
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

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mullo_epi64() {
        let a = _mm_set_epi64x(1, 2);
        let b = _mm_set_epi64x(3, 4);
        let r = _mm_mullo_epi64(a, b);
        let e = _mm_set_epi64x(3, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_mullo_epi64() {
        let a = _mm_set_epi64x(1, 2);
        let b = _mm_set_epi64x(3, 4);
        let c = _mm_set_epi64x(5, 6);
        let r = _mm_mask_mullo_epi64(c, 0b01, a, b);
        let e = _mm_set_epi64x(5, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_mullo_epi64() {
        let a = _mm_set_epi64x(1, 2);
        let b = _mm_set_epi64x(3, 4);
        let r = _mm_maskz_mullo_epi64(0b01, a, b);
        let e = _mm_set_epi64x(0, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mullo_epi64() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm256_set_epi64x(5, 6, 7, 8);
        let r = _mm256_mullo_epi64(a, b);
        let e = _mm256_set_epi64x(5, 12, 21, 32);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_mullo_epi64() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm256_set_epi64x(5, 6, 7, 8);
        let c = _mm256_set_epi64x(9, 10, 11, 12);
        let r = _mm256_mask_mullo_epi64(c, 0b0110, a, b);
        let e = _mm256_set_epi64x(9, 12, 21, 12);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_mullo_epi64() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm256_set_epi64x(5, 6, 7, 8);
        let r = _mm256_maskz_mullo_epi64(0b0110, a, b);
        let e = _mm256_set_epi64x(0, 12, 21, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mullo_epi64() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_mullo_epi64(a, b);
        let e = _mm512_set_epi64(9, 20, 33, 48, 65, 84, 105, 128);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_mullo_epi64() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let c = _mm512_set_epi64(17, 18, 19, 20, 21, 22, 23, 24);
        let r = _mm512_mask_mullo_epi64(c, 0b01101001, a, b);
        let e = _mm512_set_epi64(17, 20, 33, 20, 65, 22, 23, 128);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_mullo_epi64() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm512_set_epi64(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm512_maskz_mullo_epi64(0b01101001, a, b);
        let e = _mm512_set_epi64(0, 20, 33, 0, 65, 0, 0, 128);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_cvtmask8_u32() {
        let a: __mmask8 = 0b01101001;
        let r = _cvtmask8_u32(a);
        let e: u32 = 0b01101001;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_cvtu32_mask8() {
        let a: u32 = 0b01101001;
        let r = _cvtu32_mask8(a);
        let e: __mmask8 = 0b01101001;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_kadd_mask16() {
        let a: __mmask16 = 27549;
        let b: __mmask16 = 23434;
        let r = _kadd_mask16(a, b);
        let e: __mmask16 = 50983;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_kadd_mask8() {
        let a: __mmask8 = 98;
        let b: __mmask8 = 117;
        let r = _kadd_mask8(a, b);
        let e: __mmask8 = 215;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_kand_mask8() {
        let a: __mmask8 = 0b01101001;
        let b: __mmask8 = 0b10110011;
        let r = _kand_mask8(a, b);
        let e: __mmask8 = 0b00100001;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_kandn_mask8() {
        let a: __mmask8 = 0b01101001;
        let b: __mmask8 = 0b10110011;
        let r = _kandn_mask8(a, b);
        let e: __mmask8 = 0b10010010;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_knot_mask8() {
        let a: __mmask8 = 0b01101001;
        let r = _knot_mask8(a);
        let e: __mmask8 = 0b10010110;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_kor_mask8() {
        let a: __mmask8 = 0b01101001;
        let b: __mmask8 = 0b10110011;
        let r = _kor_mask8(a, b);
        let e: __mmask8 = 0b11111011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_kxnor_mask8() {
        let a: __mmask8 = 0b01101001;
        let b: __mmask8 = 0b10110011;
        let r = _kxnor_mask8(a, b);
        let e: __mmask8 = 0b00100101;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_kxor_mask8() {
        let a: __mmask8 = 0b01101001;
        let b: __mmask8 = 0b10110011;
        let r = _kxor_mask8(a, b);
        let e: __mmask8 = 0b11011010;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_kortest_mask8_u8() {
        let a: __mmask8 = 0b01101001;
        let b: __mmask8 = 0b10110110;
        let mut all_ones: u8 = 0;
        let r = _kortest_mask8_u8(a, b, &mut all_ones);
        assert_eq!(r, 0);
        assert_eq!(all_ones, 1);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_kortestc_mask8_u8() {
        let a: __mmask8 = 0b01101001;
        let b: __mmask8 = 0b10110110;
        let r = _kortestc_mask8_u8(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_kortestz_mask8_u8() {
        let a: __mmask8 = 0b01101001;
        let b: __mmask8 = 0b10110110;
        let r = _kortestz_mask8_u8(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_kshiftli_mask8() {
        let a: __mmask8 = 0b01101001;
        let r = _kshiftli_mask8::<3>(a);
        let e: __mmask8 = 0b01001000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_kshiftri_mask8() {
        let a: __mmask8 = 0b01101001;
        let r = _kshiftri_mask8::<3>(a);
        let e: __mmask8 = 0b00001101;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_ktest_mask8_u8() {
        let a: __mmask8 = 0b01101001;
        let b: __mmask8 = 0b10010110;
        let mut and_not: u8 = 0;
        let r = _ktest_mask8_u8(a, b, &mut and_not);
        assert_eq!(r, 1);
        assert_eq!(and_not, 0);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_ktestc_mask8_u8() {
        let a: __mmask8 = 0b01101001;
        let b: __mmask8 = 0b10010110;
        let r = _ktestc_mask8_u8(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_ktestz_mask8_u8() {
        let a: __mmask8 = 0b01101001;
        let b: __mmask8 = 0b10010110;
        let r = _ktestz_mask8_u8(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_ktest_mask16_u8() {
        let a: __mmask16 = 0b0110100100111100;
        let b: __mmask16 = 0b1001011011000011;
        let mut and_not: u8 = 0;
        let r = _ktest_mask16_u8(a, b, &mut and_not);
        assert_eq!(r, 1);
        assert_eq!(and_not, 0);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_ktestc_mask16_u8() {
        let a: __mmask16 = 0b0110100100111100;
        let b: __mmask16 = 0b1001011011000011;
        let r = _ktestc_mask16_u8(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_ktestz_mask16_u8() {
        let a: __mmask16 = 0b0110100100111100;
        let b: __mmask16 = 0b1001011011000011;
        let r = _ktestz_mask16_u8(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_load_mask8() {
        let a: __mmask8 = 0b01101001;
        let r = _load_mask8(&a);
        let e: __mmask8 = 0b01101001;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_store_mask8() {
        let a: __mmask8 = 0b01101001;
        let mut r = 0;
        _store_mask8(&mut r, a);
        let e: __mmask8 = 0b01101001;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_movepi32_mask() {
        let a = _mm_set_epi32(0, -2, -3, 4);
        let r = _mm_movepi32_mask(a);
        let e = 0b0110;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_movepi32_mask() {
        let a = _mm256_set_epi32(0, -2, -3, 4, -5, 6, 7, -8);
        let r = _mm256_movepi32_mask(a);
        let e = 0b01101001;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_movepi32_mask() {
        let a = _mm512_set_epi32(
            0, -2, -3, 4, -5, 6, 7, -8, 9, 10, -11, -12, -13, -14, 15, 16,
        );
        let r = _mm512_movepi32_mask(a);
        let e = 0b0110100100111100;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_movepi64_mask() {
        let a = _mm_set_epi64x(0, -2);
        let r = _mm_movepi64_mask(a);
        let e = 0b01;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_movepi64_mask() {
        let a = _mm256_set_epi64x(0, -2, -3, 4);
        let r = _mm256_movepi64_mask(a);
        let e = 0b0110;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_movepi64_mask() {
        let a = _mm512_set_epi64(0, -2, -3, 4, -5, 6, 7, -8);
        let r = _mm512_movepi64_mask(a);
        let e = 0b01101001;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_movm_epi32() {
        let a = 0b0110;
        let r = _mm_movm_epi32(a);
        let e = _mm_set_epi32(0, -1, -1, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_movm_epi32() {
        let a = 0b01101001;
        let r = _mm256_movm_epi32(a);
        let e = _mm256_set_epi32(0, -1, -1, 0, -1, 0, 0, -1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_movm_epi32() {
        let a = 0b0110100100111100;
        let r = _mm512_movm_epi32(a);
        let e = _mm512_set_epi32(0, -1, -1, 0, -1, 0, 0, -1, 0, 0, -1, -1, -1, -1, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_movm_epi64() {
        let a = 0b01;
        let r = _mm_movm_epi64(a);
        let e = _mm_set_epi64x(0, -1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_movm_epi64() {
        let a = 0b0110;
        let r = _mm256_movm_epi64(a);
        let e = _mm256_set_epi64x(0, -1, -1, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_movm_epi64() {
        let a = 0b01101001;
        let r = _mm512_movm_epi64(a);
        let e = _mm512_set_epi64(0, -1, -1, 0, -1, 0, 0, -1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_range_round_pd() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_pd(2., 1., 4., 3., 6., 5., 8., 7.);
        let r = _mm512_range_round_pd::<0b0101, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm512_set_pd(2., 2., 4., 4., 6., 6., 8., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_range_round_pd() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_pd(2., 1., 4., 3., 6., 5., 8., 7.);
        let c = _mm512_set_pd(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm512_mask_range_round_pd::<0b0101, _MM_FROUND_NO_EXC>(c, 0b01101001, a, b);
        let e = _mm512_set_pd(9., 2., 4., 12., 6., 14., 15., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_range_round_pd() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_pd(2., 1., 4., 3., 6., 5., 8., 7.);
        let r = _mm512_maskz_range_round_pd::<0b0101, _MM_FROUND_NO_EXC>(0b01101001, a, b);
        let e = _mm512_set_pd(0., 2., 4., 0., 6., 0., 0., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_range_pd() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm_set_pd(2., 1.);
        let r = _mm_range_pd::<0b0101>(a, b);
        let e = _mm_set_pd(2., 2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_range_pd() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm_set_pd(2., 1.);
        let c = _mm_set_pd(3., 4.);
        let r = _mm_mask_range_pd::<0b0101>(c, 0b01, a, b);
        let e = _mm_set_pd(3., 2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_range_pd() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm_set_pd(2., 1.);
        let r = _mm_maskz_range_pd::<0b0101>(0b01, a, b);
        let e = _mm_set_pd(0., 2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_range_pd() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm256_set_pd(2., 1., 4., 3.);
        let r = _mm256_range_pd::<0b0101>(a, b);
        let e = _mm256_set_pd(2., 2., 4., 4.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_range_pd() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm256_set_pd(2., 1., 4., 3.);
        let c = _mm256_set_pd(5., 6., 7., 8.);
        let r = _mm256_mask_range_pd::<0b0101>(c, 0b0110, a, b);
        let e = _mm256_set_pd(5., 2., 4., 8.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_range_pd() {
        let a = _mm256_set_pd(1., 2., 3., 4.);
        let b = _mm256_set_pd(2., 1., 4., 3.);
        let r = _mm256_maskz_range_pd::<0b0101>(0b0110, a, b);
        let e = _mm256_set_pd(0., 2., 4., 0.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_range_pd() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_pd(2., 1., 4., 3., 6., 5., 8., 7.);
        let r = _mm512_range_pd::<0b0101>(a, b);
        let e = _mm512_set_pd(2., 2., 4., 4., 6., 6., 8., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_range_pd() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_pd(2., 1., 4., 3., 6., 5., 8., 7.);
        let c = _mm512_set_pd(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm512_mask_range_pd::<0b0101>(c, 0b01101001, a, b);
        let e = _mm512_set_pd(9., 2., 4., 12., 6., 14., 15., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_range_pd() {
        let a = _mm512_set_pd(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm512_set_pd(2., 1., 4., 3., 6., 5., 8., 7.);
        let r = _mm512_maskz_range_pd::<0b0101>(0b01101001, a, b);
        let e = _mm512_set_pd(0., 2., 4., 0., 6., 0., 0., 8.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_range_round_ps() {
        let a = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let b = _mm512_set_ps(
            2., 1., 4., 3., 6., 5., 8., 7., 10., 9., 12., 11., 14., 13., 16., 15.,
        );
        let r = _mm512_range_round_ps::<0b0101, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm512_set_ps(
            2., 2., 4., 4., 6., 6., 8., 8., 10., 10., 12., 12., 14., 14., 16., 16.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_range_round_ps() {
        let a = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let b = _mm512_set_ps(
            2., 1., 4., 3., 6., 5., 8., 7., 10., 9., 12., 11., 14., 13., 16., 15.,
        );
        let c = _mm512_set_ps(
            17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.,
        );
        let r =
            _mm512_mask_range_round_ps::<0b0101, _MM_FROUND_NO_EXC>(c, 0b0110100100111100, a, b);
        let e = _mm512_set_ps(
            17., 2., 4., 20., 6., 22., 23., 8., 25., 26., 12., 12., 14., 14., 31., 32.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_range_round_ps() {
        let a = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let b = _mm512_set_ps(
            2., 1., 4., 3., 6., 5., 8., 7., 10., 9., 12., 11., 14., 13., 16., 15.,
        );
        let r = _mm512_maskz_range_round_ps::<0b0101, _MM_FROUND_NO_EXC>(0b0110100100111100, a, b);
        let e = _mm512_set_ps(
            0., 2., 4., 0., 6., 0., 0., 8., 0., 0., 12., 12., 14., 14., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_range_ps() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm_set_ps(2., 1., 4., 3.);
        let r = _mm_range_ps::<0b0101>(a, b);
        let e = _mm_set_ps(2., 2., 4., 4.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_range_ps() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm_set_ps(2., 1., 4., 3.);
        let c = _mm_set_ps(5., 6., 7., 8.);
        let r = _mm_mask_range_ps::<0b0101>(c, 0b0110, a, b);
        let e = _mm_set_ps(5., 2., 4., 8.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_range_ps() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm_set_ps(2., 1., 4., 3.);
        let r = _mm_maskz_range_ps::<0b0101>(0b0110, a, b);
        let e = _mm_set_ps(0., 2., 4., 0.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_range_ps() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm256_set_ps(2., 1., 4., 3., 6., 5., 8., 7.);
        let r = _mm256_range_ps::<0b0101>(a, b);
        let e = _mm256_set_ps(2., 2., 4., 4., 6., 6., 8., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_range_ps() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm256_set_ps(2., 1., 4., 3., 6., 5., 8., 7.);
        let c = _mm256_set_ps(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm256_mask_range_ps::<0b0101>(c, 0b01101001, a, b);
        let e = _mm256_set_ps(9., 2., 4., 12., 6., 14., 15., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_range_ps() {
        let a = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm256_set_ps(2., 1., 4., 3., 6., 5., 8., 7.);
        let r = _mm256_maskz_range_ps::<0b0101>(0b01101001, a, b);
        let e = _mm256_set_ps(0., 2., 4., 0., 6., 0., 0., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_range_ps() {
        let a = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let b = _mm512_set_ps(
            2., 1., 4., 3., 6., 5., 8., 7., 10., 9., 12., 11., 14., 13., 16., 15.,
        );
        let r = _mm512_range_ps::<0b0101>(a, b);
        let e = _mm512_set_ps(
            2., 2., 4., 4., 6., 6., 8., 8., 10., 10., 12., 12., 14., 14., 16., 16.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_range_ps() {
        let a = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let b = _mm512_set_ps(
            2., 1., 4., 3., 6., 5., 8., 7., 10., 9., 12., 11., 14., 13., 16., 15.,
        );
        let c = _mm512_set_ps(
            17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.,
        );
        let r = _mm512_mask_range_ps::<0b0101>(c, 0b0110100100111100, a, b);
        let e = _mm512_set_ps(
            17., 2., 4., 20., 6., 22., 23., 8., 25., 26., 12., 12., 14., 14., 31., 32.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_range_ps() {
        let a = _mm512_set_ps(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let b = _mm512_set_ps(
            2., 1., 4., 3., 6., 5., 8., 7., 10., 9., 12., 11., 14., 13., 16., 15.,
        );
        let r = _mm512_maskz_range_ps::<0b0101>(0b0110100100111100, a, b);
        let e = _mm512_set_ps(
            0., 2., 4., 0., 6., 0., 0., 8., 0., 0., 12., 12., 14., 14., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_range_round_sd() {
        let a = _mm_set_sd(1.);
        let b = _mm_set_sd(2.);
        let r = _mm_range_round_sd::<0b0101, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm_set_sd(2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_mask_range_round_sd() {
        let a = _mm_set_sd(1.);
        let b = _mm_set_sd(2.);
        let c = _mm_set_sd(3.);
        let r = _mm_mask_range_round_sd::<0b0101, _MM_FROUND_NO_EXC>(c, 0b0, a, b);
        let e = _mm_set_sd(3.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_maskz_range_round_sd() {
        let a = _mm_set_sd(1.);
        let b = _mm_set_sd(2.);
        let r = _mm_maskz_range_round_sd::<0b0101, _MM_FROUND_NO_EXC>(0b0, a, b);
        let e = _mm_set_sd(0.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_mask_range_sd() {
        let a = _mm_set_sd(1.);
        let b = _mm_set_sd(2.);
        let c = _mm_set_sd(3.);
        let r = _mm_mask_range_sd::<0b0101>(c, 0b0, a, b);
        let e = _mm_set_sd(3.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_maskz_range_sd() {
        let a = _mm_set_sd(1.);
        let b = _mm_set_sd(2.);
        let r = _mm_maskz_range_sd::<0b0101>(0b0, a, b);
        let e = _mm_set_sd(0.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_range_round_ss() {
        let a = _mm_set_ss(1.);
        let b = _mm_set_ss(2.);
        let r = _mm_range_round_ss::<0b0101, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm_set_ss(2.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_mask_range_round_ss() {
        let a = _mm_set_ss(1.);
        let b = _mm_set_ss(2.);
        let c = _mm_set_ss(3.);
        let r = _mm_mask_range_round_ss::<0b0101, _MM_FROUND_NO_EXC>(c, 0b0, a, b);
        let e = _mm_set_ss(3.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_maskz_range_round_ss() {
        let a = _mm_set_ss(1.);
        let b = _mm_set_ss(2.);
        let r = _mm_maskz_range_round_ss::<0b0101, _MM_FROUND_NO_EXC>(0b0, a, b);
        let e = _mm_set_ss(0.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_mask_range_ss() {
        let a = _mm_set_ss(1.);
        let b = _mm_set_ss(2.);
        let c = _mm_set_ss(3.);
        let r = _mm_mask_range_ss::<0b0101>(c, 0b0, a, b);
        let e = _mm_set_ss(3.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_maskz_range_ss() {
        let a = _mm_set_ss(1.);
        let b = _mm_set_ss(2.);
        let r = _mm_maskz_range_ss::<0b0101>(0b0, a, b);
        let e = _mm_set_ss(0.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_reduce_round_pd() {
        let a = _mm512_set_pd(0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0);
        let r = _mm512_reduce_round_pd::<{ 16 | _MM_FROUND_TO_ZERO }, _MM_FROUND_NO_EXC>(a);
        let e = _mm512_set_pd(0.25, 0., 0.25, 0., 0.25, 0., 0.25, 0.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_reduce_round_pd() {
        let a = _mm512_set_pd(0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0);
        let src = _mm512_set_pd(3., 4., 5., 6., 7., 8., 9., 10.);
        let r = _mm512_mask_reduce_round_pd::<{ 16 | _MM_FROUND_TO_ZERO }, _MM_FROUND_NO_EXC>(
            src, 0b01101001, a,
        );
        let e = _mm512_set_pd(3., 0., 0.25, 6., 0.25, 8., 9., 0.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_reduce_round_pd() {
        let a = _mm512_set_pd(0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0);
        let r = _mm512_maskz_reduce_round_pd::<{ 16 | _MM_FROUND_TO_ZERO }, _MM_FROUND_NO_EXC>(
            0b01101001, a,
        );
        let e = _mm512_set_pd(0., 0., 0.25, 0., 0.25, 0., 0., 0.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_reduce_pd() {
        let a = _mm_set_pd(0.25, 0.50);
        let r = _mm_reduce_pd::<{ 16 | _MM_FROUND_TO_ZERO }>(a);
        let e = _mm_set_pd(0.25, 0.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_reduce_pd() {
        let a = _mm_set_pd(0.25, 0.50);
        let src = _mm_set_pd(3., 4.);
        let r = _mm_mask_reduce_pd::<{ 16 | _MM_FROUND_TO_ZERO }>(src, 0b01, a);
        let e = _mm_set_pd(3., 0.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_reduce_pd() {
        let a = _mm_set_pd(0.25, 0.50);
        let r = _mm_maskz_reduce_pd::<{ 16 | _MM_FROUND_TO_ZERO }>(0b01, a);
        let e = _mm_set_pd(0., 0.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_reduce_pd() {
        let a = _mm256_set_pd(0.25, 0.50, 0.75, 1.0);
        let r = _mm256_reduce_pd::<{ 16 | _MM_FROUND_TO_ZERO }>(a);
        let e = _mm256_set_pd(0.25, 0., 0.25, 0.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_reduce_pd() {
        let a = _mm256_set_pd(0.25, 0.50, 0.75, 1.0);
        let src = _mm256_set_pd(3., 4., 5., 6.);
        let r = _mm256_mask_reduce_pd::<{ 16 | _MM_FROUND_TO_ZERO }>(src, 0b0110, a);
        let e = _mm256_set_pd(3., 0., 0.25, 6.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_reduce_pd() {
        let a = _mm256_set_pd(0.25, 0.50, 0.75, 1.0);
        let r = _mm256_maskz_reduce_pd::<{ 16 | _MM_FROUND_TO_ZERO }>(0b0110, a);
        let e = _mm256_set_pd(0., 0., 0.25, 0.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_reduce_pd() {
        let a = _mm512_set_pd(0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0);
        let r = _mm512_reduce_pd::<{ 16 | _MM_FROUND_TO_ZERO }>(a);
        let e = _mm512_set_pd(0.25, 0., 0.25, 0., 0.25, 0., 0.25, 0.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_reduce_pd() {
        let a = _mm512_set_pd(0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0);
        let src = _mm512_set_pd(3., 4., 5., 6., 7., 8., 9., 10.);
        let r = _mm512_mask_reduce_pd::<{ 16 | _MM_FROUND_TO_ZERO }>(src, 0b01101001, a);
        let e = _mm512_set_pd(3., 0., 0.25, 6., 0.25, 8., 9., 0.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_reduce_pd() {
        let a = _mm512_set_pd(0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0);
        let r = _mm512_maskz_reduce_pd::<{ 16 | _MM_FROUND_TO_ZERO }>(0b01101001, a);
        let e = _mm512_set_pd(0., 0., 0.25, 0., 0.25, 0., 0., 0.);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_reduce_round_ps() {
        let a = _mm512_set_ps(
            0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0, 3.25, 3.50, 3.75,
            4.0,
        );
        let r = _mm512_reduce_round_ps::<{ 16 | _MM_FROUND_TO_ZERO }, _MM_FROUND_NO_EXC>(a);
        let e = _mm512_set_ps(
            0.25, 0., 0.25, 0., 0.25, 0., 0.25, 0., 0.25, 0., 0.25, 0., 0.25, 0., 0.25, 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_reduce_round_ps() {
        let a = _mm512_set_ps(
            0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0, 3.25, 3.50, 3.75,
            4.0,
        );
        let src = _mm512_set_ps(
            5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
        );
        let r = _mm512_mask_reduce_round_ps::<{ 16 | _MM_FROUND_TO_ZERO }, _MM_FROUND_NO_EXC>(
            src,
            0b0110100100111100,
            a,
        );
        let e = _mm512_set_ps(
            5., 0., 0.25, 8., 0.25, 10., 11., 0., 13., 14., 0.25, 0., 0.25, 0., 19., 20.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_reduce_round_ps() {
        let a = _mm512_set_ps(
            0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0, 3.25, 3.50, 3.75,
            4.0,
        );
        let r = _mm512_maskz_reduce_round_ps::<{ 16 | _MM_FROUND_TO_ZERO }, _MM_FROUND_NO_EXC>(
            0b0110100100111100,
            a,
        );
        let e = _mm512_set_ps(
            0., 0., 0.25, 0., 0.25, 0., 0., 0., 0., 0., 0.25, 0., 0.25, 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_reduce_ps() {
        let a = _mm_set_ps(0.25, 0.50, 0.75, 1.0);
        let r = _mm_reduce_ps::<{ 16 | _MM_FROUND_TO_ZERO }>(a);
        let e = _mm_set_ps(0.25, 0., 0.25, 0.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_reduce_ps() {
        let a = _mm_set_ps(0.25, 0.50, 0.75, 1.0);
        let src = _mm_set_ps(2., 3., 4., 5.);
        let r = _mm_mask_reduce_ps::<{ 16 | _MM_FROUND_TO_ZERO }>(src, 0b0110, a);
        let e = _mm_set_ps(2., 0., 0.25, 5.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_maskz_reduce_ps() {
        let a = _mm_set_ps(0.25, 0.50, 0.75, 1.0);
        let r = _mm_maskz_reduce_ps::<{ 16 | _MM_FROUND_TO_ZERO }>(0b0110, a);
        let e = _mm_set_ps(0., 0., 0.25, 0.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_reduce_ps() {
        let a = _mm256_set_ps(0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0);
        let r = _mm256_reduce_ps::<{ 16 | _MM_FROUND_TO_ZERO }>(a);
        let e = _mm256_set_ps(0.25, 0., 0.25, 0., 0.25, 0., 0.25, 0.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_reduce_ps() {
        let a = _mm256_set_ps(0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0);
        let src = _mm256_set_ps(3., 4., 5., 6., 7., 8., 9., 10.);
        let r = _mm256_mask_reduce_ps::<{ 16 | _MM_FROUND_TO_ZERO }>(src, 0b01101001, a);
        let e = _mm256_set_ps(3., 0., 0.25, 6., 0.25, 8., 9., 0.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_maskz_reduce_ps() {
        let a = _mm256_set_ps(0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0);
        let r = _mm256_maskz_reduce_ps::<{ 16 | _MM_FROUND_TO_ZERO }>(0b01101001, a);
        let e = _mm256_set_ps(0., 0., 0.25, 0., 0.25, 0., 0., 0.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_reduce_ps() {
        let a = _mm512_set_ps(
            0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0, 3.25, 3.50, 3.75,
            4.0,
        );
        let r = _mm512_reduce_ps::<{ 16 | _MM_FROUND_TO_ZERO }>(a);
        let e = _mm512_set_ps(
            0.25, 0., 0.25, 0., 0.25, 0., 0.25, 0., 0.25, 0., 0.25, 0., 0.25, 0., 0.25, 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_reduce_ps() {
        let a = _mm512_set_ps(
            0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0, 3.25, 3.50, 3.75,
            4.0,
        );
        let src = _mm512_set_ps(
            5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
        );
        let r = _mm512_mask_reduce_ps::<{ 16 | _MM_FROUND_TO_ZERO }>(src, 0b0110100100111100, a);
        let e = _mm512_set_ps(
            5., 0., 0.25, 8., 0.25, 10., 11., 0., 13., 14., 0.25, 0., 0.25, 0., 19., 20.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_maskz_reduce_ps() {
        let a = _mm512_set_ps(
            0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0, 3.25, 3.50, 3.75,
            4.0,
        );
        let r = _mm512_maskz_reduce_ps::<{ 16 | _MM_FROUND_TO_ZERO }>(0b0110100100111100, a);
        let e = _mm512_set_ps(
            0., 0., 0.25, 0., 0.25, 0., 0., 0., 0., 0., 0.25, 0., 0.25, 0., 0., 0.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_reduce_round_sd() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm_set_sd(0.25);
        let r = _mm_reduce_round_sd::<{ 16 | _MM_FROUND_TO_ZERO }, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm_set_pd(1., 0.25);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_mask_reduce_round_sd() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm_set_sd(0.25);
        let c = _mm_set_pd(3., 4.);
        let r = _mm_mask_reduce_round_sd::<{ 16 | _MM_FROUND_TO_ZERO }, _MM_FROUND_NO_EXC>(
            c, 0b0, a, b,
        );
        let e = _mm_set_pd(1., 4.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_maskz_reduce_round_sd() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm_set_sd(0.25);
        let r =
            _mm_maskz_reduce_round_sd::<{ 16 | _MM_FROUND_TO_ZERO }, _MM_FROUND_NO_EXC>(0b0, a, b);
        let e = _mm_set_pd(1., 0.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_reduce_sd() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm_set_sd(0.25);
        let r = _mm_reduce_sd::<{ 16 | _MM_FROUND_TO_ZERO }>(a, b);
        let e = _mm_set_pd(1., 0.25);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_mask_reduce_sd() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm_set_sd(0.25);
        let c = _mm_set_pd(3., 4.);
        let r = _mm_mask_reduce_sd::<{ 16 | _MM_FROUND_TO_ZERO }>(c, 0b0, a, b);
        let e = _mm_set_pd(1., 4.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_maskz_reduce_sd() {
        let a = _mm_set_pd(1., 2.);
        let b = _mm_set_sd(0.25);
        let r = _mm_maskz_reduce_sd::<{ 16 | _MM_FROUND_TO_ZERO }>(0b0, a, b);
        let e = _mm_set_pd(1., 0.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_reduce_round_ss() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm_set_ss(0.25);
        let r = _mm_reduce_round_ss::<{ 16 | _MM_FROUND_TO_ZERO }, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm_set_ps(1., 2., 3., 0.25);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_mask_reduce_round_ss() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm_set_ss(0.25);
        let c = _mm_set_ps(5., 6., 7., 8.);
        let r = _mm_mask_reduce_round_ss::<{ 16 | _MM_FROUND_TO_ZERO }, _MM_FROUND_NO_EXC>(
            c, 0b0, a, b,
        );
        let e = _mm_set_ps(1., 2., 3., 8.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_maskz_reduce_round_ss() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm_set_ss(0.25);
        let r =
            _mm_maskz_reduce_round_ss::<{ 16 | _MM_FROUND_TO_ZERO }, _MM_FROUND_NO_EXC>(0b0, a, b);
        let e = _mm_set_ps(1., 2., 3., 0.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_reduce_ss() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm_set_ss(0.25);
        let r = _mm_reduce_ss::<{ 16 | _MM_FROUND_TO_ZERO }>(a, b);
        let e = _mm_set_ps(1., 2., 3., 0.25);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_mask_reduce_ss() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm_set_ss(0.25);
        let c = _mm_set_ps(5., 6., 7., 8.);
        let r = _mm_mask_reduce_ss::<{ 16 | _MM_FROUND_TO_ZERO }>(c, 0b0, a, b);
        let e = _mm_set_ps(1., 2., 3., 8.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_maskz_reduce_ss() {
        let a = _mm_set_ps(1., 2., 3., 4.);
        let b = _mm_set_ss(0.25);
        let r = _mm_maskz_reduce_ss::<{ 16 | _MM_FROUND_TO_ZERO }>(0b0, a, b);
        let e = _mm_set_ps(1., 2., 3., 0.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_fpclass_pd_mask() {
        let a = _mm_set_pd(1., f64::INFINITY);
        let r = _mm_fpclass_pd_mask::<0x18>(a);
        let e = 0b01;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_fpclass_pd_mask() {
        let a = _mm_set_pd(1., f64::INFINITY);
        let r = _mm_mask_fpclass_pd_mask::<0x18>(0b10, a);
        let e = 0b00;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_fpclass_pd_mask() {
        let a = _mm256_set_pd(1., f64::INFINITY, f64::NEG_INFINITY, 0.0);
        let r = _mm256_fpclass_pd_mask::<0x18>(a);
        let e = 0b0110;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_fpclass_pd_mask() {
        let a = _mm256_set_pd(1., f64::INFINITY, f64::NEG_INFINITY, 0.0);
        let r = _mm256_mask_fpclass_pd_mask::<0x18>(0b1010, a);
        let e = 0b0010;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_fpclass_pd_mask() {
        let a = _mm512_set_pd(
            1.,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            -2.0,
            f64::NAN,
            1.0e-308,
        );
        let r = _mm512_fpclass_pd_mask::<0x18>(a);
        let e = 0b01100000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_fpclass_pd_mask() {
        let a = _mm512_set_pd(
            1.,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            -2.0,
            f64::NAN,
            1.0e-308,
        );
        let r = _mm512_mask_fpclass_pd_mask::<0x18>(0b10101010, a);
        let e = 0b00100000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_fpclass_ps_mask() {
        let a = _mm_set_ps(1., f32::INFINITY, f32::NEG_INFINITY, 0.0);
        let r = _mm_fpclass_ps_mask::<0x18>(a);
        let e = 0b0110;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm_mask_fpclass_ps_mask() {
        let a = _mm_set_ps(1., f32::INFINITY, f32::NEG_INFINITY, 0.0);
        let r = _mm_mask_fpclass_ps_mask::<0x18>(0b1010, a);
        let e = 0b0010;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_fpclass_ps_mask() {
        let a = _mm256_set_ps(
            1.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            -2.0,
            f32::NAN,
            1.0e-38,
        );
        let r = _mm256_fpclass_ps_mask::<0x18>(a);
        let e = 0b01100000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq,avx512vl")]
    unsafe fn test_mm256_mask_fpclass_ps_mask() {
        let a = _mm256_set_ps(
            1.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            -2.0,
            f32::NAN,
            1.0e-38,
        );
        let r = _mm256_mask_fpclass_ps_mask::<0x18>(0b10101010, a);
        let e = 0b00100000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_fpclass_ps_mask() {
        let a = _mm512_set_ps(
            1.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            -2.0,
            f32::NAN,
            1.0e-38,
            -1.,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -0.0,
            0.0,
            2.0,
            f32::NAN,
            -1.0e-38,
        );
        let r = _mm512_fpclass_ps_mask::<0x18>(a);
        let e = 0b0110000001100000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm512_mask_fpclass_ps_mask() {
        let a = _mm512_set_ps(
            1.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            -2.0,
            f32::NAN,
            1.0e-38,
            -1.,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -0.0,
            0.0,
            2.0,
            f32::NAN,
            -1.0e-38,
        );
        let r = _mm512_mask_fpclass_ps_mask::<0x18>(0b1010101010101010, a);
        let e = 0b0010000000100000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_fpclass_sd_mask() {
        let a = _mm_set_pd(1., f64::INFINITY);
        let r = _mm_fpclass_sd_mask::<0x18>(a);
        let e = 0b1;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_mask_fpclass_sd_mask() {
        let a = _mm_set_sd(f64::INFINITY);
        let r = _mm_mask_fpclass_sd_mask::<0x18>(0b0, a);
        let e = 0b0;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_fpclass_ss_mask() {
        let a = _mm_set_ss(f32::INFINITY);
        let r = _mm_fpclass_ss_mask::<0x18>(a);
        let e = 0b1;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512dq")]
    unsafe fn test_mm_mask_fpclass_ss_mask() {
        let a = _mm_set_ss(f32::INFINITY);
        let r = _mm_mask_fpclass_ss_mask::<0x18>(0b0, a);
        let e = 0b0;
        assert_eq!(r, e);
    }
}
