use crate::arch::asm;
use crate::core_arch::{simd::*, x86::*};
use crate::intrinsics::simd::*;
use crate::ptr;

/// Set packed half-precision (16-bit) floating-point elements in dst with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_set_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_set_ph(
    e7: f16,
    e6: f16,
    e5: f16,
    e4: f16,
    e3: f16,
    e2: f16,
    e1: f16,
    e0: f16,
) -> __m128h {
    __m128h(e0, e1, e2, e3, e4, e5, e6, e7)
}

/// Set packed half-precision (16-bit) floating-point elements in dst with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_set_ph(
    e15: f16,
    e14: f16,
    e13: f16,
    e12: f16,
    e11: f16,
    e10: f16,
    e9: f16,
    e8: f16,
    e7: f16,
    e6: f16,
    e5: f16,
    e4: f16,
    e3: f16,
    e2: f16,
    e1: f16,
    e0: f16,
) -> __m256h {
    __m256h(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
    )
}

/// Set packed half-precision (16-bit) floating-point elements in dst with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_set_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_set_ph(
    e31: f16,
    e30: f16,
    e29: f16,
    e28: f16,
    e27: f16,
    e26: f16,
    e25: f16,
    e24: f16,
    e23: f16,
    e22: f16,
    e21: f16,
    e20: f16,
    e19: f16,
    e18: f16,
    e17: f16,
    e16: f16,
    e15: f16,
    e14: f16,
    e13: f16,
    e12: f16,
    e11: f16,
    e10: f16,
    e9: f16,
    e8: f16,
    e7: f16,
    e6: f16,
    e5: f16,
    e4: f16,
    e3: f16,
    e2: f16,
    e1: f16,
    e0: f16,
) -> __m512h {
    __m512h(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19,
        e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31,
    )
}

/// Copy half-precision (16-bit) floating-point elements from a to the lower element of dst and zero
/// the upper 7 elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_set_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_set_sh(a: f16) -> __m128h {
    __m128h(a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
}

/// Broadcast the half-precision (16-bit) floating-point value a to all elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_set1_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_set1_ph(a: f16) -> __m128h {
    transmute(f16x8::splat(a))
}

/// Broadcast the half-precision (16-bit) floating-point value a to all elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set1_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_set1_ph(a: f16) -> __m256h {
    transmute(f16x16::splat(a))
}

/// Broadcast the half-precision (16-bit) floating-point value a to all elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_set1_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_set1_ph(a: f16) -> __m512h {
    transmute(f16x32::splat(a))
}

/// Set packed half-precision (16-bit) floating-point elements in dst with the supplied values in reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_setr_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_setr_ph(
    e0: f16,
    e1: f16,
    e2: f16,
    e3: f16,
    e4: f16,
    e5: f16,
    e6: f16,
    e7: f16,
) -> __m128h {
    __m128h(e0, e1, e2, e3, e4, e5, e6, e7)
}

/// Set packed half-precision (16-bit) floating-point elements in dst with the supplied values in reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setr_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_setr_ph(
    e0: f16,
    e1: f16,
    e2: f16,
    e3: f16,
    e4: f16,
    e5: f16,
    e6: f16,
    e7: f16,
    e8: f16,
    e9: f16,
    e10: f16,
    e11: f16,
    e12: f16,
    e13: f16,
    e14: f16,
    e15: f16,
) -> __m256h {
    __m256h(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
    )
}

/// Set packed half-precision (16-bit) floating-point elements in dst with the supplied values in reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_setr_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_setr_ph(
    e0: f16,
    e1: f16,
    e2: f16,
    e3: f16,
    e4: f16,
    e5: f16,
    e6: f16,
    e7: f16,
    e8: f16,
    e9: f16,
    e10: f16,
    e11: f16,
    e12: f16,
    e13: f16,
    e14: f16,
    e15: f16,
    e16: f16,
    e17: f16,
    e18: f16,
    e19: f16,
    e20: f16,
    e21: f16,
    e22: f16,
    e23: f16,
    e24: f16,
    e25: f16,
    e26: f16,
    e27: f16,
    e28: f16,
    e29: f16,
    e30: f16,
    e31: f16,
) -> __m512h {
    __m512h(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19,
        e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31,
    )
}

/// Return vector of type __m128h with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_setzero_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_setzero_ph() -> __m128h {
    transmute(f16x8::splat(0.0))
}

/// Return vector of type __m256h with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setzero_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_setzero_ph() -> __m256h {
    transmute(f16x16::splat(0.0))
}

/// Return vector of type __m512h with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_setzero_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_setzero_ph() -> __m512h {
    transmute(f16x32::splat(0.0))
}

/// Return vector of type `__m128h` with undefined elements. In practice, this returns the all-zero
/// vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_undefined_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_undefined_ph() -> __m128h {
    transmute(f16x8::splat(0.0))
}

/// Return vector of type `__m256h` with undefined elements. In practice, this returns the all-zero
/// vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_undefined_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_undefined_ph() -> __m256h {
    transmute(f16x16::splat(0.0))
}

/// Return vector of type `__m512h` with undefined elements. In practice, this returns the all-zero
/// vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_undefined_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_undefined_ph() -> __m512h {
    transmute(f16x32::splat(0.0))
}

/// Cast vector of type `__m128d` to type `__m128h`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_castpd_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_castpd_ph(a: __m128d) -> __m128h {
    transmute(a)
}

/// Cast vector of type `__m256d` to type `__m256h`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castpd_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_castpd_ph(a: __m256d) -> __m256h {
    transmute(a)
}

/// Cast vector of type `__m512d` to type `__m512h`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_castpd_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_castpd_ph(a: __m512d) -> __m512h {
    transmute(a)
}

/// Cast vector of type `__m128h` to type `__m128d`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_castph_pd)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_castph_pd(a: __m128h) -> __m128d {
    transmute(a)
}

/// Cast vector of type `__m256h` to type `__m256d`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castph_pd)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_castph_pd(a: __m256h) -> __m256d {
    transmute(a)
}

/// Cast vector of type `__m512h` to type `__m512d`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_castph_pd)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_castph_pd(a: __m512h) -> __m512d {
    transmute(a)
}

/// Cast vector of type `__m128` to type `__m128h`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_castps_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_castps_ph(a: __m128) -> __m128h {
    transmute(a)
}

/// Cast vector of type `__m256` to type `__m256h`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castps_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_castps_ph(a: __m256) -> __m256h {
    transmute(a)
}

/// Cast vector of type `__m512` to type `__m512h`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_castps_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_castps_ph(a: __m512) -> __m512h {
    transmute(a)
}

/// Cast vector of type `__m128h` to type `__m128`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_castph_ps)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_castph_ps(a: __m128h) -> __m128 {
    transmute(a)
}

/// Cast vector of type `__m256h` to type `__m256`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castph_ps)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_castph_ps(a: __m256h) -> __m256 {
    transmute(a)
}

/// Cast vector of type `__m512h` to type `__m512`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_castph_ps)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_castph_ps(a: __m512h) -> __m512 {
    transmute(a)
}

/// Cast vector of type `__m128i` to type `__m128h`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_castsi128_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_castsi128_ph(a: __m128i) -> __m128h {
    transmute(a)
}

/// Cast vector of type `__m256i` to type `__m256h`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castsi256_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_castsi256_ph(a: __m256i) -> __m256h {
    transmute(a)
}

/// Cast vector of type `__m512i` to type `__m512h`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_castsi512_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_castsi512_ph(a: __m512i) -> __m512h {
    transmute(a)
}

/// Cast vector of type `__m128h` to type `__m128i`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_castph_si128)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_castph_si128(a: __m128h) -> __m128i {
    transmute(a)
}

/// Cast vector of type `__m256h` to type `__m256i`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castph_si256)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_castph_si256(a: __m256h) -> __m256i {
    transmute(a)
}

/// Cast vector of type `__m512h` to type `__m512i`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_castph_si512)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_castph_si512(a: __m512h) -> __m512i {
    transmute(a)
}

/// Cast vector of type `__m256h` to type `__m128h`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castph256_ph128)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_castph256_ph128(a: __m256h) -> __m128h {
    simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Cast vector of type `__m512h` to type `__m128h`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_castph512_ph128)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_castph512_ph128(a: __m512h) -> __m128h {
    simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Cast vector of type `__m512h` to type `__m256h`. This intrinsic is only used for compilation and
/// does not generate any instructions, thus it has zero latency.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_castph512_ph256)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_castph512_ph256(a: __m512h) -> __m256h {
    simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Cast vector of type `__m128h` to type `__m256h`. The upper 8 elements of the result are undefined.
/// In practice, the upper elements are zeroed. This intrinsic can generate the `vzeroupper` instruction,
/// but most of the time it does not generate any instructions.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castph128_ph256)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_castph128_ph256(a: __m128h) -> __m256h {
    simd_shuffle!(
        a,
        _mm_undefined_ph(),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8]
    )
}

/// Cast vector of type `__m128h` to type `__m512h`. The upper 24 elements of the result are undefined.
/// In practice, the upper elements are zeroed. This intrinsic can generate the `vzeroupper` instruction,
/// but most of the time it does not generate any instructions.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_castph128_ph512)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_castph128_ph512(a: __m128h) -> __m512h {
    simd_shuffle!(
        a,
        _mm_undefined_ph(),
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            8, 8, 8
        ]
    )
}

/// Cast vector of type `__m256h` to type `__m512h`. The upper 16 elements of the result are undefined.
/// In practice, the upper elements are zeroed. This intrinsic can generate the `vzeroupper` instruction,
/// but most of the time it does not generate any instructions.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_castph256_ph512)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_castph256_ph512(a: __m256h) -> __m512h {
    simd_shuffle!(
        a,
        _mm256_undefined_ph(),
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 16, 16, 16, 16
        ]
    )
}

/// Cast vector of type `__m256h` to type `__m128h`. The upper 8 elements of the result are zeroed.
/// This intrinsic can generate the `vzeroupper` instruction, but most of the time it does not generate
/// any instructions.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_zextph128_ph256)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_zextph128_ph256(a: __m128h) -> __m256h {
    simd_shuffle!(
        a,
        _mm_setzero_ph(),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8]
    )
}

/// Cast vector of type `__m128h` to type `__m512h`. The upper 24 elements of the result are zeroed.
/// This intrinsic can generate the `vzeroupper` instruction, but most of the time it does not generate
/// any instructions.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_zextph128_ph512)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_zextph128_ph512(a: __m128h) -> __m512h {
    simd_shuffle!(
        a,
        _mm_setzero_ph(),
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            8, 8, 8
        ]
    )
}

/// Cast vector of type `__m256h` to type `__m512h`. The upper 16 elements of the result are zeroed.
/// This intrinsic can generate the `vzeroupper` instruction, but most of the time it does not generate
/// any instructions.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_zextph256_ph512)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_zextph256_ph512(a: __m256h) -> __m512h {
    simd_shuffle!(
        a,
        _mm256_setzero_ph(),
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 16, 16, 16, 16
        ]
    )
}

/// Compare the lower half-precision (16-bit) floating-point elements in a and b based on the comparison
/// operand specified by imm8, and return the boolean result (0 or 1).
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_comi_round_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_comi_round_sh<const IMM8: i32, const SAE: i32>(a: __m128h, b: __m128h) -> i32 {
    static_assert_sae!(SAE);
    vcomish(a, b, IMM8, SAE)
}

/// Compare the lower half-precision (16-bit) floating-point elements in a and b based on the comparison
/// operand specified by imm8, and return the boolean result (0 or 1).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_comi_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_comi_sh<const IMM8: i32>(a: __m128h, b: __m128h) -> i32 {
    _mm_comi_round_sh::<IMM8, _MM_FROUND_CUR_DIRECTION>(a, b)
}

/// Compare the lower half-precision (16-bit) floating-point elements in a and b for equality, and return
/// the boolean result (0 or 1).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_comieq_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_comieq_sh(a: __m128h, b: __m128h) -> i32 {
    _mm_comi_sh::<_CMP_EQ_OS>(a, b)
}

/// Compare the lower half-precision (16-bit) floating-point elements in a and b for greater-than-or-equal,
/// and return the boolean result (0 or 1).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_comige_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_comige_sh(a: __m128h, b: __m128h) -> i32 {
    _mm_comi_sh::<_CMP_GE_OS>(a, b)
}

/// Compare the lower half-precision (16-bit) floating-point elements in a and b for greater-than, and return
/// the boolean result (0 or 1).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_comigt_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_comigt_sh(a: __m128h, b: __m128h) -> i32 {
    _mm_comi_sh::<_CMP_GT_OS>(a, b)
}

/// Compare the lower half-precision (16-bit) floating-point elements in a and b for less-than-or-equal, and
/// return the boolean result (0 or 1).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_comile_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_comile_sh(a: __m128h, b: __m128h) -> i32 {
    _mm_comi_sh::<_CMP_LE_OS>(a, b)
}

/// Compare the lower half-precision (16-bit) floating-point elements in a and b for less-than, and return
/// the boolean result (0 or 1).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_comilt_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_comilt_sh(a: __m128h, b: __m128h) -> i32 {
    _mm_comi_sh::<_CMP_LT_OS>(a, b)
}

/// Compare the lower half-precision (16-bit) floating-point elements in a and b for not-equal, and return
/// the boolean result (0 or 1).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_comineq_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_comineq_sh(a: __m128h, b: __m128h) -> i32 {
    _mm_comi_sh::<_CMP_NEQ_OS>(a, b)
}

/// Compare the lower half-precision (16-bit) floating-point elements in a and b for equality, and
/// return the boolean result (0 or 1). This instruction will not signal an exception for QNaNs.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_ucomieq_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_ucomieq_sh(a: __m128h, b: __m128h) -> i32 {
    _mm_comi_sh::<_CMP_EQ_OQ>(a, b)
}

/// Compare the lower half-precision (16-bit) floating-point elements in a and b for greater-than-or-equal,
/// and return the boolean result (0 or 1). This instruction will not signal an exception for QNaNs.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_ucomige_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_ucomige_sh(a: __m128h, b: __m128h) -> i32 {
    _mm_comi_sh::<_CMP_GE_OQ>(a, b)
}

/// Compare the lower half-precision (16-bit) floating-point elements in a and b for greater-than, and return
/// the boolean result (0 or 1). This instruction will not signal an exception for QNaNs.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_ucomigt_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_ucomigt_sh(a: __m128h, b: __m128h) -> i32 {
    _mm_comi_sh::<_CMP_GT_OQ>(a, b)
}

/// Compare the lower half-precision (16-bit) floating-point elements in a and b for less-than-or-equal, and
/// return the boolean result (0 or 1). This instruction will not signal an exception for QNaNs.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_ucomile_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_ucomile_sh(a: __m128h, b: __m128h) -> i32 {
    _mm_comi_sh::<_CMP_LE_OQ>(a, b)
}

/// Compare the lower half-precision (16-bit) floating-point elements in a and b for less-than, and return
/// the boolean result (0 or 1). This instruction will not signal an exception for QNaNs.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_ucomilt_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_ucomilt_sh(a: __m128h, b: __m128h) -> i32 {
    _mm_comi_sh::<_CMP_LT_OQ>(a, b)
}

/// Compare the lower half-precision (16-bit) floating-point elements in a and b for not-equal, and return
/// the boolean result (0 or 1). This instruction will not signal an exception for QNaNs.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_ucomineq_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_ucomineq_sh(a: __m128h, b: __m128h) -> i32 {
    _mm_comi_sh::<_CMP_NEQ_OQ>(a, b)
}

/// Load 128-bits (composed of 8 packed half-precision (16-bit) floating-point elements) from memory into
/// a new vector. The address must be aligned to 16 bytes or a general-protection exception may be generated.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_load_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_load_ph(mem_addr: *const f16) -> __m128h {
    *mem_addr.cast()
}

/// Load 256-bits (composed of 16 packed half-precision (16-bit) floating-point elements) from memory into
/// a new vector. The address must be aligned to 32 bytes or a general-protection exception may be generated.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_load_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_load_ph(mem_addr: *const f16) -> __m256h {
    *mem_addr.cast()
}

/// Load 512-bits (composed of 32 packed half-precision (16-bit) floating-point elements) from memory into
/// a new vector. The address must be aligned to 64 bytes or a general-protection exception may be generated.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_load_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_load_ph(mem_addr: *const f16) -> __m512h {
    *mem_addr.cast()
}

/// Load a half-precision (16-bit) floating-point element from memory into the lower element of a new vector,
/// and zero the upper elements
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_load_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_load_sh(mem_addr: *const f16) -> __m128h {
    _mm_set_sh(*mem_addr)
}

/// Load a half-precision (16-bit) floating-point element from memory into the lower element of a new vector
/// using writemask k (the element is copied from src when mask bit 0 is not set), and zero the upper elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_load_sh)
#[inline]
#[target_feature(enable = "avx512fp16,sse,avx512f")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_load_sh(src: __m128h, k: __mmask8, mem_addr: *const f16) -> __m128h {
    let mut dst = src;
    asm!(
        vpl!("vmovsh {dst}{{{k}}}"),
        dst = inout(xmm_reg) dst,
        k = in(kreg) k,
        p = in(reg) mem_addr,
        options(pure, nomem, nostack, preserves_flags)
    );
    dst
}

/// Load a half-precision (16-bit) floating-point element from memory into the lower element of a new vector
/// using zeromask k (the element is zeroed out when mask bit 0 is not set), and zero the upper elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_load_sh)
#[inline]
#[target_feature(enable = "avx512fp16,sse,avx512f")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_maskz_load_sh(k: __mmask8, mem_addr: *const f16) -> __m128h {
    let mut dst: __m128h;
    asm!(
        vpl!("vmovsh {dst}{{{k}}}{{z}}"),
        dst = out(xmm_reg) dst,
        k = in(kreg) k,
        p = in(reg) mem_addr,
        options(pure, nomem, nostack, preserves_flags)
    );
    dst
}

/// Load 128-bits (composed of 8 packed half-precision (16-bit) floating-point elements) from memory into
/// a new vector. The address does not need to be aligned to any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_loadu_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_loadu_ph(mem_addr: *const f16) -> __m128h {
    ptr::read_unaligned(mem_addr.cast())
}

/// Load 256-bits (composed of 16 packed half-precision (16-bit) floating-point elements) from memory into
/// a new vector. The address does not need to be aligned to any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_loadu_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_loadu_ph(mem_addr: *const f16) -> __m256h {
    ptr::read_unaligned(mem_addr.cast())
}

/// Load 512-bits (composed of 32 packed half-precision (16-bit) floating-point elements) from memory into
/// a new vector. The address does not need to be aligned to any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_loadu_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_loadu_ph(mem_addr: *const f16) -> __m512h {
    ptr::read_unaligned(mem_addr.cast())
}

/// Move the lower half-precision (16-bit) floating-point element from b to the lower element of dst
/// using writemask k (the element is copied from src when mask bit 0 is not set), and copy the upper
/// 7 packed elements from a to the upper elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_move_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_move_sh(src: __m128h, k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    let mut mov: f16 = simd_extract!(src, 0);
    if (k & 1) != 0 {
        mov = simd_extract!(b, 0);
    }
    simd_insert!(a, 0, mov)
}

/// Move the lower half-precision (16-bit) floating-point element from b to the lower element of dst
/// using zeromask k (the element is zeroed out when mask bit 0 is not set), and copy the upper 7 packed
/// elements from a to the upper elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_move_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_maskz_move_sh(k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    let mut mov: f16 = 0.;
    if (k & 1) != 0 {
        mov = simd_extract!(b, 0);
    }
    simd_insert!(a, 0, mov)
}

/// Move the lower half-precision (16-bit) floating-point element from b to the lower element of dst,
/// and copy the upper 7 packed elements from a to the upper elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_move_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_move_sh(a: __m128h, b: __m128h) -> __m128h {
    let mov: f16 = simd_extract!(b, 0);
    simd_insert!(a, 0, mov)
}

/// Store 128-bits (composed of 8 packed half-precision (16-bit) floating-point elements) from a into memory.
/// The address must be aligned to 16 bytes or a general-protection exception may be generated.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_store_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_store_ph(mem_addr: *mut f16, a: __m128h) {
    *mem_addr.cast() = a;
}

/// Store 256-bits (composed of 16 packed half-precision (16-bit) floating-point elements) from a into memory.
/// The address must be aligned to 32 bytes or a general-protection exception may be generated.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_store_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_store_ph(mem_addr: *mut f16, a: __m256h) {
    *mem_addr.cast() = a;
}

/// Store 512-bits (composed of 32 packed half-precision (16-bit) floating-point elements) from a into memory.
/// The address must be aligned to 64 bytes or a general-protection exception may be generated.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_store_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_store_ph(mem_addr: *mut f16, a: __m512h) {
    *mem_addr.cast() = a;
}

/// Store the lower half-precision (16-bit) floating-point element from a into memory.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_store_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_store_sh(mem_addr: *mut f16, a: __m128h) {
    *mem_addr = simd_extract!(a, 0);
}

/// Store the lower half-precision (16-bit) floating-point element from a into memory using writemask k
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_store_sh)
#[inline]
#[target_feature(enable = "avx512fp16,sse,avx512f")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_store_sh(mem_addr: *mut f16, k: __mmask8, a: __m128h) {
    asm!(
        vps!("vmovdqu16", "{{{k}}}, {src}"),
        p = in(reg) mem_addr,
        k = in(kreg) k,
        src = in(xmm_reg) a,
        options(nostack, preserves_flags)
    );
}

/// Store 128-bits (composed of 8 packed half-precision (16-bit) floating-point elements) from a into memory.
/// The address does not need to be aligned to any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_storeu_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_storeu_ph(mem_addr: *mut f16, a: __m128h) {
    ptr::write_unaligned(mem_addr.cast(), a);
}

/// Store 256-bits (composed of 16 packed half-precision (16-bit) floating-point elements) from a into memory.
/// The address does not need to be aligned to any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_storeu_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_storeu_ph(mem_addr: *mut f16, a: __m256h) {
    ptr::write_unaligned(mem_addr.cast(), a);
}

/// Store 512-bits (composed of 32 packed half-precision (16-bit) floating-point elements) from a into memory.
/// The address does not need to be aligned to any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_storeu_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_storeu_ph(mem_addr: *mut f16, a: __m512h) {
    ptr::write_unaligned(mem_addr.cast(), a);
}

/// Add packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_add_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vaddph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_add_ph(a: __m128h, b: __m128h) -> __m128h {
    simd_add(a, b)
}

/// Add packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_add_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vaddph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_add_ph(src: __m128h, k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    let r = _mm_add_ph(a, b);
    simd_select_bitmask(k, r, src)
}

/// Add packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_add_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vaddph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_maskz_add_ph(k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    let r = _mm_add_ph(a, b);
    simd_select_bitmask(k, r, _mm_setzero_ph())
}

/// Add packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_add_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vaddph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_add_ph(a: __m256h, b: __m256h) -> __m256h {
    simd_add(a, b)
}

/// Add packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_add_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vaddph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_mask_add_ph(src: __m256h, k: __mmask16, a: __m256h, b: __m256h) -> __m256h {
    let r = _mm256_add_ph(a, b);
    simd_select_bitmask(k, r, src)
}

/// Add packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_add_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vaddph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_maskz_add_ph(k: __mmask16, a: __m256h, b: __m256h) -> __m256h {
    let r = _mm256_add_ph(a, b);
    simd_select_bitmask(k, r, _mm256_setzero_ph())
}

/// Add packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_add_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vaddph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_add_ph(a: __m512h, b: __m512h) -> __m512h {
    simd_add(a, b)
}

/// Add packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_add_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vaddph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_mask_add_ph(src: __m512h, k: __mmask32, a: __m512h, b: __m512h) -> __m512h {
    let r = _mm512_add_ph(a, b);
    simd_select_bitmask(k, r, src)
}

/// Add packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_add_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vaddph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_maskz_add_ph(k: __mmask32, a: __m512h, b: __m512h) -> __m512h {
    let r = _mm512_add_ph(a, b);
    simd_select_bitmask(k, r, _mm512_setzero_ph())
}

/// Add packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst.
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_add_round_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vaddph, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_add_round_ph<const ROUNDING: i32>(a: __m512h, b: __m512h) -> __m512h {
    static_assert_rounding!(ROUNDING);
    vaddph(a, b, ROUNDING)
}

/// Add packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_add_round_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vaddph, ROUNDING = 8))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_mask_add_round_ph<const ROUNDING: i32>(
    src: __m512h,
    k: __mmask32,
    a: __m512h,
    b: __m512h,
) -> __m512h {
    static_assert_rounding!(ROUNDING);
    let r = _mm512_add_round_ph::<ROUNDING>(a, b);
    simd_select_bitmask(k, r, src)
}

/// Add packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_add_round_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vaddph, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_maskz_add_round_ph<const ROUNDING: i32>(
    k: __mmask32,
    a: __m512h,
    b: __m512h,
) -> __m512h {
    static_assert_rounding!(ROUNDING);
    let r = _mm512_add_round_ph::<ROUNDING>(a, b);
    simd_select_bitmask(k, r, _mm512_setzero_ph())
}

/// Add the lower half-precision (16-bit) floating-point elements in a and b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst.
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_add_round_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vaddsh, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_add_round_sh<const ROUNDING: i32>(a: __m128h, b: __m128h) -> __m128h {
    static_assert_rounding!(ROUNDING);
    _mm_mask_add_round_sh::<ROUNDING>(_mm_undefined_ph(), 0xff, a, b)
}

/// Add the lower half-precision (16-bit) floating-point elements in a and b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// writemask k (the element is copied from src when mask bit 0 is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_add_round_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vaddsh, ROUNDING = 8))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_add_round_sh<const ROUNDING: i32>(
    src: __m128h,
    k: __mmask8,
    a: __m128h,
    b: __m128h,
) -> __m128h {
    static_assert_rounding!(ROUNDING);
    vaddsh(a, b, src, k, ROUNDING)
}

/// Add the lower half-precision (16-bit) floating-point elements in a and b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// zeromask k (the element is zeroed out when mask bit 0 is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_add_round_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vaddsh, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_maskz_add_round_sh<const ROUNDING: i32>(
    k: __mmask8,
    a: __m128h,
    b: __m128h,
) -> __m128h {
    static_assert_rounding!(ROUNDING);
    _mm_mask_add_round_sh::<ROUNDING>(_mm_setzero_ph(), k, a, b)
}

/// Add the lower half-precision (16-bit) floating-point elements in a and b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_add_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vaddsh))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_add_sh(a: __m128h, b: __m128h) -> __m128h {
    _mm_add_round_sh::<_MM_FROUND_CUR_DIRECTION>(a, b)
}

/// Add the lower half-precision (16-bit) floating-point elements in a and b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// writemask k (the element is copied from src when mask bit 0 is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_add_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vaddsh))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_add_sh(src: __m128h, k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    _mm_mask_add_round_sh::<_MM_FROUND_CUR_DIRECTION>(src, k, a, b)
}

/// Add the lower half-precision (16-bit) floating-point elements in a and b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// zeromask k (the element is zeroed out when mask bit 0 is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_add_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vaddsh))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_maskz_add_sh(k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    _mm_maskz_add_round_sh::<_MM_FROUND_CUR_DIRECTION>(k, a, b)
}

/// Subtract packed half-precision (16-bit) floating-point elements in b from a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sub_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vsubph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_sub_ph(a: __m128h, b: __m128h) -> __m128h {
    simd_sub(a, b)
}

/// Subtract packed half-precision (16-bit) floating-point elements in b from a, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_sub_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vsubph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_sub_ph(src: __m128h, k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    let r = _mm_sub_ph(a, b);
    simd_select_bitmask(k, r, src)
}

/// Subtract packed half-precision (16-bit) floating-point elements in b from a, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_sub_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vsubph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_maskz_sub_ph(k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    let r = _mm_sub_ph(a, b);
    simd_select_bitmask(k, r, _mm_setzero_ph())
}

/// Subtract packed half-precision (16-bit) floating-point elements in b from a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sub_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vsubph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_sub_ph(a: __m256h, b: __m256h) -> __m256h {
    simd_sub(a, b)
}

/// Subtract packed half-precision (16-bit) floating-point elements in b from a, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_sub_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vsubph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_mask_sub_ph(src: __m256h, k: __mmask16, a: __m256h, b: __m256h) -> __m256h {
    let r = _mm256_sub_ph(a, b);
    simd_select_bitmask(k, r, src)
}

/// Subtract packed half-precision (16-bit) floating-point elements in b from a, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_sub_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vsubph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_maskz_sub_ph(k: __mmask16, a: __m256h, b: __m256h) -> __m256h {
    let r = _mm256_sub_ph(a, b);
    simd_select_bitmask(k, r, _mm256_setzero_ph())
}

/// Subtract packed half-precision (16-bit) floating-point elements in b from a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_sub_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vsubph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_sub_ph(a: __m512h, b: __m512h) -> __m512h {
    simd_sub(a, b)
}

/// Subtract packed half-precision (16-bit) floating-point elements in b from a, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_sub_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vsubph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_mask_sub_ph(src: __m512h, k: __mmask32, a: __m512h, b: __m512h) -> __m512h {
    let r = _mm512_sub_ph(a, b);
    simd_select_bitmask(k, r, src)
}

/// Subtract packed half-precision (16-bit) floating-point elements in b from a, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_sub_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vsubph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_maskz_sub_ph(k: __mmask32, a: __m512h, b: __m512h) -> __m512h {
    let r = _mm512_sub_ph(a, b);
    simd_select_bitmask(k, r, _mm512_setzero_ph())
}

/// Subtract packed half-precision (16-bit) floating-point elements in b from a, and store the results in dst.
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_sub_round_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vsubph, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_sub_round_ph<const ROUNDING: i32>(a: __m512h, b: __m512h) -> __m512h {
    static_assert_rounding!(ROUNDING);
    vsubph(a, b, ROUNDING)
}

/// Subtract packed half-precision (16-bit) floating-point elements in b from a, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_sub_round_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vsubph, ROUNDING = 8))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_mask_sub_round_ph<const ROUNDING: i32>(
    src: __m512h,
    k: __mmask32,
    a: __m512h,
    b: __m512h,
) -> __m512h {
    static_assert_rounding!(ROUNDING);
    let r = _mm512_sub_round_ph::<ROUNDING>(a, b);
    simd_select_bitmask(k, r, src)
}

/// Subtract packed half-precision (16-bit) floating-point elements in b from a, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_sub_round_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vsubph, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_maskz_sub_round_ph<const ROUNDING: i32>(
    k: __mmask32,
    a: __m512h,
    b: __m512h,
) -> __m512h {
    static_assert_rounding!(ROUNDING);
    let r = _mm512_sub_round_ph::<ROUNDING>(a, b);
    simd_select_bitmask(k, r, _mm512_setzero_ph())
}

/// Subtract the lower half-precision (16-bit) floating-point elements in b from a, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst.
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sub_round_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vsubsh, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_sub_round_sh<const ROUNDING: i32>(a: __m128h, b: __m128h) -> __m128h {
    static_assert_rounding!(ROUNDING);
    _mm_mask_sub_round_sh::<ROUNDING>(_mm_undefined_ph(), 0xff, a, b)
}

/// Subtract the lower half-precision (16-bit) floating-point elements in b from a, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// writemask k (the element is copied from src when mask bit 0 is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_sub_round_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vsubsh, ROUNDING = 8))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_sub_round_sh<const ROUNDING: i32>(
    src: __m128h,
    k: __mmask8,
    a: __m128h,
    b: __m128h,
) -> __m128h {
    static_assert_rounding!(ROUNDING);
    vsubsh(a, b, src, k, ROUNDING)
}

/// Subtract the lower half-precision (16-bit) floating-point elements in b from a, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// zeromask k (the element is zeroed out when mask bit 0 is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_sub_round_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vsubsh, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_maskz_sub_round_sh<const ROUNDING: i32>(
    k: __mmask8,
    a: __m128h,
    b: __m128h,
) -> __m128h {
    static_assert_rounding!(ROUNDING);
    _mm_mask_sub_round_sh::<ROUNDING>(_mm_setzero_ph(), k, a, b)
}

/// Subtract the lower half-precision (16-bit) floating-point elements in b from a, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sub_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vsubsh))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_sub_sh(a: __m128h, b: __m128h) -> __m128h {
    _mm_sub_round_sh::<_MM_FROUND_CUR_DIRECTION>(a, b)
}

/// Subtract the lower half-precision (16-bit) floating-point elements in b from a, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// writemask k (the element is copied from src when mask bit 0 is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_sub_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vsubsh))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_sub_sh(src: __m128h, k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    _mm_mask_sub_round_sh::<_MM_FROUND_CUR_DIRECTION>(src, k, a, b)
}

/// Subtract the lower half-precision (16-bit) floating-point elements in b from a, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// zeromask k (the element is zeroed out when mask bit 0 is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_sub_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vsubsh))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_maskz_sub_sh(k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    _mm_maskz_sub_round_sh::<_MM_FROUND_CUR_DIRECTION>(k, a, b)
}

/// Multiply packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mul_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vmulph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mul_ph(a: __m128h, b: __m128h) -> __m128h {
    simd_mul(a, b)
}

/// Multiply packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_mul_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vmulph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_mul_ph(src: __m128h, k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    let r = _mm_mul_ph(a, b);
    simd_select_bitmask(k, r, src)
}

/// Multiply packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_mul_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vmulph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_maskz_mul_ph(k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    let r = _mm_mul_ph(a, b);
    simd_select_bitmask(k, r, _mm_setzero_ph())
}

/// Multiply packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mul_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vmulph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_mul_ph(a: __m256h, b: __m256h) -> __m256h {
    simd_mul(a, b)
}

/// Multiply packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_mul_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vmulph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_mask_mul_ph(src: __m256h, k: __mmask16, a: __m256h, b: __m256h) -> __m256h {
    let r = _mm256_mul_ph(a, b);
    simd_select_bitmask(k, r, src)
}

/// Multiply packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_mul_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vmulph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_maskz_mul_ph(k: __mmask16, a: __m256h, b: __m256h) -> __m256h {
    let r = _mm256_mul_ph(a, b);
    simd_select_bitmask(k, r, _mm256_setzero_ph())
}

/// Multiply packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mul_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vmulph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_mul_ph(a: __m512h, b: __m512h) -> __m512h {
    simd_mul(a, b)
}

/// Multiply packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_mul_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vmulph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_mask_mul_ph(src: __m512h, k: __mmask32, a: __m512h, b: __m512h) -> __m512h {
    let r = _mm512_mul_ph(a, b);
    simd_select_bitmask(k, r, src)
}

/// Multiply packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_mul_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vmulph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_maskz_mul_ph(k: __mmask32, a: __m512h, b: __m512h) -> __m512h {
    let r = _mm512_mul_ph(a, b);
    simd_select_bitmask(k, r, _mm512_setzero_ph())
}

/// Multiply packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst.
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mul_round_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vmulph, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_mul_round_ph<const ROUNDING: i32>(a: __m512h, b: __m512h) -> __m512h {
    static_assert_rounding!(ROUNDING);
    vmulph(a, b, ROUNDING)
}

/// Multiply packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_mul_round_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vmulph, ROUNDING = 8))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_mask_mul_round_ph<const ROUNDING: i32>(
    src: __m512h,
    k: __mmask32,
    a: __m512h,
    b: __m512h,
) -> __m512h {
    static_assert_rounding!(ROUNDING);
    let r = _mm512_mul_round_ph::<ROUNDING>(a, b);
    simd_select_bitmask(k, r, src)
}

/// Multiply packed half-precision (16-bit) floating-point elements in a and b, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_mul_round_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vmulph, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_maskz_mul_round_ph<const ROUNDING: i32>(
    k: __mmask32,
    a: __m512h,
    b: __m512h,
) -> __m512h {
    static_assert_rounding!(ROUNDING);
    let r = _mm512_mul_round_ph::<ROUNDING>(a, b);
    simd_select_bitmask(k, r, _mm512_setzero_ph())
}

/// Multiply the lower half-precision (16-bit) floating-point elements in a and b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst.
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mul_round_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vmulsh, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mul_round_sh<const ROUNDING: i32>(a: __m128h, b: __m128h) -> __m128h {
    static_assert_rounding!(ROUNDING);
    _mm_mask_mul_round_sh::<ROUNDING>(_mm_undefined_ph(), 0xff, a, b)
}

/// Multiply the lower half-precision (16-bit) floating-point elements in a and b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// writemask k (the element is copied from src when mask bit 0 is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_mul_round_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vmulsh, ROUNDING = 8))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_mul_round_sh<const ROUNDING: i32>(
    src: __m128h,
    k: __mmask8,
    a: __m128h,
    b: __m128h,
) -> __m128h {
    static_assert_rounding!(ROUNDING);
    vmulsh(a, b, src, k, ROUNDING)
}

/// Multiply the lower half-precision (16-bit) floating-point elements in a and b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// zeromask k (the element is zeroed out when mask bit 0 is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_mul_round_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vmulsh, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_maskz_mul_round_sh<const ROUNDING: i32>(
    k: __mmask8,
    a: __m128h,
    b: __m128h,
) -> __m128h {
    static_assert_rounding!(ROUNDING);
    _mm_mask_mul_round_sh::<ROUNDING>(_mm_setzero_ph(), k, a, b)
}

/// Multiply the lower half-precision (16-bit) floating-point elements in a and b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mul_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vmulsh))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mul_sh(a: __m128h, b: __m128h) -> __m128h {
    _mm_mul_round_sh::<_MM_FROUND_CUR_DIRECTION>(a, b)
}

/// Multiply the lower half-precision (16-bit) floating-point elements in a and b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// writemask k (the element is copied from src when mask bit 0 is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_mul_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vmulsh))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_mul_sh(src: __m128h, k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    _mm_mask_mul_round_sh::<_MM_FROUND_CUR_DIRECTION>(src, k, a, b)
}

/// Multiply the lower half-precision (16-bit) floating-point elements in a and b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// zeromask k (the element is zeroed out when mask bit 0 is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_mul_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vmulsh))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_maskz_mul_sh(k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    _mm_maskz_mul_round_sh::<_MM_FROUND_CUR_DIRECTION>(k, a, b)
}

/// Divide packed half-precision (16-bit) floating-point elements in a by b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_div_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vdivph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_div_ph(a: __m128h, b: __m128h) -> __m128h {
    simd_div(a, b)
}

/// Divide packed half-precision (16-bit) floating-point elements in a by b, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_div_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vdivph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_div_ph(src: __m128h, k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    let r = _mm_div_ph(a, b);
    simd_select_bitmask(k, r, src)
}

/// Divide packed half-precision (16-bit) floating-point elements in a by b, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_div_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vdivph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_maskz_div_ph(k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    let r = _mm_div_ph(a, b);
    simd_select_bitmask(k, r, _mm_setzero_ph())
}

/// Divide packed half-precision (16-bit) floating-point elements in a by b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_div_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vdivph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_div_ph(a: __m256h, b: __m256h) -> __m256h {
    simd_div(a, b)
}

/// Divide packed half-precision (16-bit) floating-point elements in a by b, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_div_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vdivph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_mask_div_ph(src: __m256h, k: __mmask16, a: __m256h, b: __m256h) -> __m256h {
    let r = _mm256_div_ph(a, b);
    simd_select_bitmask(k, r, src)
}

/// Divide packed half-precision (16-bit) floating-point elements in a by b, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_div_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[cfg_attr(test, assert_instr(vdivph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_maskz_div_ph(k: __mmask16, a: __m256h, b: __m256h) -> __m256h {
    let r = _mm256_div_ph(a, b);
    simd_select_bitmask(k, r, _mm256_setzero_ph())
}

/// Divide packed half-precision (16-bit) floating-point elements in a by b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_div_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vdivph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_div_ph(a: __m512h, b: __m512h) -> __m512h {
    simd_div(a, b)
}

/// Divide packed half-precision (16-bit) floating-point elements in a by b, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_div_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vdivph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_mask_div_ph(src: __m512h, k: __mmask32, a: __m512h, b: __m512h) -> __m512h {
    let r = _mm512_div_ph(a, b);
    simd_select_bitmask(k, r, src)
}

/// Divide packed half-precision (16-bit) floating-point elements in a by b, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_div_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vdivph))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_maskz_div_ph(k: __mmask32, a: __m512h, b: __m512h) -> __m512h {
    let r = _mm512_div_ph(a, b);
    simd_select_bitmask(k, r, _mm512_setzero_ph())
}

/// Divide packed half-precision (16-bit) floating-point elements in a by b, and store the results in dst.
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_div_round_ph)

#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vdivph, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_div_round_ph<const ROUNDING: i32>(a: __m512h, b: __m512h) -> __m512h {
    static_assert_rounding!(ROUNDING);
    vdivph(a, b, ROUNDING)
}

/// Divide packed half-precision (16-bit) floating-point elements in a by b, and store the results in dst using
/// writemask k (elements are copied from src when the corresponding mask bit is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_div_round_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vdivph, ROUNDING = 8))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_mask_div_round_ph<const ROUNDING: i32>(
    src: __m512h,
    k: __mmask32,
    a: __m512h,
    b: __m512h,
) -> __m512h {
    static_assert_rounding!(ROUNDING);
    let r = _mm512_div_round_ph::<ROUNDING>(a, b);
    simd_select_bitmask(k, r, src)
}

/// Divide packed half-precision (16-bit) floating-point elements in a by b, and store the results in dst using
/// zeromask k (elements are zeroed out when the corresponding mask bit is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_div_round_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vdivph, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_maskz_div_round_ph<const ROUNDING: i32>(
    k: __mmask32,
    a: __m512h,
    b: __m512h,
) -> __m512h {
    static_assert_rounding!(ROUNDING);
    let r = _mm512_div_round_ph::<ROUNDING>(a, b);
    simd_select_bitmask(k, r, _mm512_setzero_ph())
}

/// Divide the lower half-precision (16-bit) floating-point elements in a by b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst.
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_div_round_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vdivsh, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_div_round_sh<const ROUNDING: i32>(a: __m128h, b: __m128h) -> __m128h {
    static_assert_rounding!(ROUNDING);
    _mm_mask_div_round_sh::<ROUNDING>(_mm_undefined_ph(), 0xff, a, b)
}

/// Divide the lower half-precision (16-bit) floating-point elements in a by b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// writemask k (the element is copied from src when mask bit 0 is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_div_round_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vdivsh, ROUNDING = 8))]
#[rustc_legacy_const_generics(4)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_div_round_sh<const ROUNDING: i32>(
    src: __m128h,
    k: __mmask8,
    a: __m128h,
    b: __m128h,
) -> __m128h {
    static_assert_rounding!(ROUNDING);
    vdivsh(a, b, src, k, ROUNDING)
}

/// Divide the lower half-precision (16-bit) floating-point elements in a by b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// zeromask k (the element is zeroed out when mask bit 0 is not set).
/// Rounding is done according to the rounding parameter, which can be one of:
///
///     (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
///     (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
///     (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
///     (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
///     _MM_FROUND_CUR_DIRECTION
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_div_round_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vdivsh, ROUNDING = 8))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_maskz_div_round_sh<const ROUNDING: i32>(
    k: __mmask8,
    a: __m128h,
    b: __m128h,
) -> __m128h {
    static_assert_rounding!(ROUNDING);
    _mm_mask_div_round_sh::<ROUNDING>(_mm_setzero_ph(), k, a, b)
}

/// Divide the lower half-precision (16-bit) floating-point elements in a by b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_div_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vdivsh))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_div_sh(a: __m128h, b: __m128h) -> __m128h {
    _mm_div_round_sh::<_MM_FROUND_CUR_DIRECTION>(a, b)
}

/// Divide the lower half-precision (16-bit) floating-point elements in a by b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// writemask k (the element is copied from src when mask bit 0 is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_div_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vdivsh))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_mask_div_sh(src: __m128h, k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    _mm_mask_div_round_sh::<_MM_FROUND_CUR_DIRECTION>(src, k, a, b)
}

/// Divide the lower half-precision (16-bit) floating-point elements in a by b, store the result in the
/// lower element of dst, and copy the upper 7 packed elements from a to the upper elements of dst using
/// zeromask k (the element is zeroed out when mask bit 0 is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_div_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vdivsh))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_maskz_div_sh(k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    _mm_maskz_div_round_sh::<_MM_FROUND_CUR_DIRECTION>(k, a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512fp16.vcomi.sh"]
    fn vcomish(a: __m128h, b: __m128h, imm8: i32, sae: i32) -> i32;

    #[link_name = "llvm.x86.avx512fp16.add.ph.512"]
    fn vaddph(a: __m512h, b: __m512h, rounding: i32) -> __m512h;
    #[link_name = "llvm.x86.avx512fp16.sub.ph.512"]
    fn vsubph(a: __m512h, b: __m512h, rounding: i32) -> __m512h;
    #[link_name = "llvm.x86.avx512fp16.mul.ph.512"]
    fn vmulph(a: __m512h, b: __m512h, rounding: i32) -> __m512h;
    #[link_name = "llvm.x86.avx512fp16.div.ph.512"]
    fn vdivph(a: __m512h, b: __m512h, rounding: i32) -> __m512h;

    #[link_name = "llvm.x86.avx512fp16.mask.add.sh.round"]
    fn vaddsh(a: __m128h, b: __m128h, src: __m128h, k: __mmask8, rounding: i32) -> __m128h;
    #[link_name = "llvm.x86.avx512fp16.mask.sub.sh.round"]
    fn vsubsh(a: __m128h, b: __m128h, src: __m128h, k: __mmask8, rounding: i32) -> __m128h;
    #[link_name = "llvm.x86.avx512fp16.mask.mul.sh.round"]
    fn vmulsh(a: __m128h, b: __m128h, src: __m128h, k: __mmask8, rounding: i32) -> __m128h;
    #[link_name = "llvm.x86.avx512fp16.mask.div.sh.round"]
    fn vdivsh(a: __m128h, b: __m128h, src: __m128h, k: __mmask8, rounding: i32) -> __m128h;

}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;
    use crate::mem::transmute;
    use crate::ptr::{addr_of, addr_of_mut};
    use stdarch_test::simd_test;

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_set_ph() {
        let r = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let e = _mm_setr_ph(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm256_set_ph() {
        let r = _mm256_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let e = _mm256_setr_ph(
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_set_ph() {
        let r = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let e = _mm512_setr_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_set_sh() {
        let r = _mm_set_sh(1.0);
        let e = _mm_set_ph(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_set1_ph() {
        let r = _mm_set1_ph(1.0);
        let e = _mm_set_ph(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm256_set1_ph() {
        let r = _mm256_set1_ph(1.0);
        let e = _mm256_set_ph(
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_set1_ph() {
        let r = _mm512_set1_ph(1.0);
        let e = _mm512_set_ph(
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_setr_ph() {
        let r = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let e = _mm_set_ph(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm256_setr_ph() {
        let r = _mm256_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let e = _mm256_set_ph(
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_setr_ph() {
        let r = _mm512_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let e = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_setzero_ph() {
        let r = _mm_setzero_ph();
        let e = _mm_set1_ph(0.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_setzero_ph() {
        let r = _mm256_setzero_ph();
        let e = _mm256_set1_ph(0.0);
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_setzero_ph() {
        let r = _mm512_setzero_ph();
        let e = _mm512_set1_ph(0.0);
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_castsi128_ph() {
        let a = _mm_set1_epi16(0x3c00);
        let r = _mm_castsi128_ph(a);
        let e = _mm_set1_ph(1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm256_castsi256_ph() {
        let a = _mm256_set1_epi16(0x3c00);
        let r = _mm256_castsi256_ph(a);
        let e = _mm256_set1_ph(1.0);
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_castsi512_ph() {
        let a = _mm512_set1_epi16(0x3c00);
        let r = _mm512_castsi512_ph(a);
        let e = _mm512_set1_ph(1.0);
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_castph_si128() {
        let a = _mm_set1_ph(1.0);
        let r = _mm_castph_si128(a);
        let e = _mm_set1_epi16(0x3c00);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm256_castph_si256() {
        let a = _mm256_set1_ph(1.0);
        let r = _mm256_castph_si256(a);
        let e = _mm256_set1_epi16(0x3c00);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_castph_si512() {
        let a = _mm512_set1_ph(1.0);
        let r = _mm512_castph_si512(a);
        let e = _mm512_set1_epi16(0x3c00);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_castps_ph() {
        let a = _mm_castsi128_ps(_mm_set1_epi16(0x3c00));
        let r = _mm_castps_ph(a);
        let e = _mm_set1_ph(1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm256_castps_ph() {
        let a = _mm256_castsi256_ps(_mm256_set1_epi16(0x3c00));
        let r = _mm256_castps_ph(a);
        let e = _mm256_set1_ph(1.0);
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_castps_ph() {
        let a = _mm512_castsi512_ps(_mm512_set1_epi16(0x3c00));
        let r = _mm512_castps_ph(a);
        let e = _mm512_set1_ph(1.0);
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_castph_ps() {
        let a = _mm_castsi128_ph(_mm_set1_epi32(0x3f800000));
        let r = _mm_castph_ps(a);
        let e = _mm_set1_ps(1.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm256_castph_ps() {
        let a = _mm256_castsi256_ph(_mm256_set1_epi32(0x3f800000));
        let r = _mm256_castph_ps(a);
        let e = _mm256_set1_ps(1.0);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_castph_ps() {
        let a = _mm512_castsi512_ph(_mm512_set1_epi32(0x3f800000));
        let r = _mm512_castph_ps(a);
        let e = _mm512_set1_ps(1.0);
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_castpd_ph() {
        let a = _mm_castsi128_pd(_mm_set1_epi16(0x3c00));
        let r = _mm_castpd_ph(a);
        let e = _mm_set1_ph(1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm256_castpd_ph() {
        let a = _mm256_castsi256_pd(_mm256_set1_epi16(0x3c00));
        let r = _mm256_castpd_ph(a);
        let e = _mm256_set1_ph(1.0);
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_castpd_ph() {
        let a = _mm512_castsi512_pd(_mm512_set1_epi16(0x3c00));
        let r = _mm512_castpd_ph(a);
        let e = _mm512_set1_ph(1.0);
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_castph_pd() {
        let a = _mm_castsi128_ph(_mm_set1_epi64x(0x3ff0000000000000));
        let r = _mm_castph_pd(a);
        let e = _mm_set1_pd(1.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm256_castph_pd() {
        let a = _mm256_castsi256_ph(_mm256_set1_epi64x(0x3ff0000000000000));
        let r = _mm256_castph_pd(a);
        let e = _mm256_set1_pd(1.0);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_castph_pd() {
        let a = _mm512_castsi512_ph(_mm512_set1_epi64(0x3ff0000000000000));
        let r = _mm512_castph_pd(a);
        let e = _mm512_set1_pd(1.0);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm256_castph256_ph128() {
        let a = _mm256_setr_ph(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let r = _mm256_castph256_ph128(a);
        let e = _mm_setr_ph(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_castph512_ph128() {
        let a = _mm512_setr_ph(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.,
        );
        let r = _mm512_castph512_ph128(a);
        let e = _mm_setr_ph(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_castph512_ph256() {
        let a = _mm512_setr_ph(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.,
        );
        let r = _mm512_castph512_ph256(a);
        let e = _mm256_setr_ph(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm256_castph128_ph256() {
        let a = _mm_setr_ph(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_castph128_ph256(a);
        assert_eq_m128h(_mm256_castph256_ph128(r), a);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_castph128_ph512() {
        let a = _mm_setr_ph(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_castph128_ph512(a);
        assert_eq_m128h(_mm512_castph512_ph128(r), a);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_castph256_ph512() {
        let a = _mm256_setr_ph(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let r = _mm512_castph256_ph512(a);
        assert_eq_m256h(_mm512_castph512_ph256(r), a);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm256_zextph128_ph256() {
        let a = _mm_setr_ph(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_zextph128_ph256(a);
        let e = _mm256_setr_ph(
            1., 2., 3., 4., 5., 6., 7., 8., 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_zextph128_ph512() {
        let a = _mm_setr_ph(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm512_zextph128_ph512(a);
        let e = _mm512_setr_ph(
            1., 2., 3., 4., 5., 6., 7., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_zextph256_ph512() {
        let a = _mm256_setr_ph(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let r = _mm512_zextph256_ph512(a);
        let e = _mm512_setr_ph(
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_comi_round_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(1.0);
        let r = _mm_comi_round_sh::<_CMP_EQ_OQ, _MM_FROUND_NO_EXC>(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_comi_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(1.0);
        let r = _mm_comi_sh::<_CMP_EQ_OQ>(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_comieq_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(1.0);
        let r = _mm_comieq_sh(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_comige_sh() {
        let a = _mm_set_sh(2.0);
        let b = _mm_set_sh(1.0);
        let r = _mm_comige_sh(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_comigt_sh() {
        let a = _mm_set_sh(2.0);
        let b = _mm_set_sh(1.0);
        let r = _mm_comigt_sh(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_comile_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_comile_sh(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_comilt_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_comilt_sh(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_comineq_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_comineq_sh(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_ucomieq_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(1.0);
        let r = _mm_ucomieq_sh(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_ucomige_sh() {
        let a = _mm_set_sh(2.0);
        let b = _mm_set_sh(1.0);
        let r = _mm_ucomige_sh(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_ucomigt_sh() {
        let a = _mm_set_sh(2.0);
        let b = _mm_set_sh(1.0);
        let r = _mm_ucomigt_sh(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_ucomile_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_ucomile_sh(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_ucomilt_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_ucomilt_sh(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_ucomineq_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_ucomineq_sh(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_load_ph() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_load_ph(addr_of!(a).cast());
        assert_eq_m128h(a, b);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_load_ph() {
        let a = _mm256_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm256_load_ph(addr_of!(a).cast());
        assert_eq_m256h(a, b);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_load_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_load_ph(addr_of!(a).cast());
        assert_eq_m512h(a, b);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_load_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_load_sh(addr_of!(a).cast());
        assert_eq_m128h(a, b);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_mask_load_sh() {
        let a = _mm_set_sh(1.0);
        let src = _mm_set_sh(2.);
        let b = _mm_mask_load_sh(src, 1, addr_of!(a).cast());
        assert_eq_m128h(a, b);
        let b = _mm_mask_load_sh(src, 0, addr_of!(a).cast());
        assert_eq_m128h(src, b);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_maskz_load_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_maskz_load_sh(1, addr_of!(a).cast());
        assert_eq_m128h(a, b);
        let b = _mm_maskz_load_sh(0, addr_of!(a).cast());
        assert_eq_m128h(_mm_setzero_ph(), b);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_loadu_ph() {
        let array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let r = _mm_loadu_ph(array.as_ptr());
        let e = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_loadu_ph() {
        let array = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let r = _mm256_loadu_ph(array.as_ptr());
        let e = _mm256_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_loadu_ph() {
        let array = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        ];
        let r = _mm512_loadu_ph(array.as_ptr());
        let e = _mm512_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_move_sh() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_set_sh(9.0);
        let r = _mm_move_sh(a, b);
        let e = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_mask_move_sh() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_set_sh(9.0);
        let src = _mm_set_sh(10.0);
        let r = _mm_mask_move_sh(src, 0, a, b);
        let e = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_maskz_move_sh() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_set_sh(9.0);
        let r = _mm_maskz_move_sh(0, a, b);
        let e = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_store_ph() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let mut b = _mm_setzero_ph();
        _mm_store_ph(addr_of_mut!(b).cast(), a);
        assert_eq_m128h(a, b);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_store_ph() {
        let a = _mm256_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let mut b = _mm256_setzero_ph();
        _mm256_store_ph(addr_of_mut!(b).cast(), a);
        assert_eq_m256h(a, b);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_store_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let mut b = _mm512_setzero_ph();
        _mm512_store_ph(addr_of_mut!(b).cast(), a);
        assert_eq_m512h(a, b);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_store_sh() {
        let a = _mm_set_sh(1.0);
        let mut b = _mm_setzero_ph();
        _mm_store_sh(addr_of_mut!(b).cast(), a);
        assert_eq_m128h(a, b);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_mask_store_sh() {
        let a = _mm_set_sh(1.0);
        let mut b = _mm_setzero_ph();
        _mm_mask_store_sh(addr_of_mut!(b).cast(), 0, a);
        assert_eq_m128h(_mm_setzero_ph(), b);
        _mm_mask_store_sh(addr_of_mut!(b).cast(), 1, a);
        assert_eq_m128h(a, b);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_storeu_ph() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let mut array = [0.0; 8];
        _mm_storeu_ph(array.as_mut_ptr(), a);
        assert_eq_m128h(a, _mm_loadu_ph(array.as_ptr()));
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_storeu_ph() {
        let a = _mm256_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let mut array = [0.0; 16];
        _mm256_storeu_ph(array.as_mut_ptr(), a);
        assert_eq_m256h(a, _mm256_loadu_ph(array.as_ptr()));
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_storeu_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let mut array = [0.0; 32];
        _mm512_storeu_ph(array.as_mut_ptr(), a);
        assert_eq_m512h(a, _mm512_loadu_ph(array.as_ptr()));
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_add_ph() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_set_ph(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        let r = _mm_add_ph(a, b);
        let e = _mm_set1_ph(9.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_mask_add_ph() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_set_ph(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        let src = _mm_set_ph(10., 11., 12., 13., 14., 15., 16., 17.);
        let r = _mm_mask_add_ph(src, 0b01010101, a, b);
        let e = _mm_set_ph(10., 9., 12., 9., 14., 9., 16., 9.);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_maskz_add_ph() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_set_ph(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        let r = _mm_maskz_add_ph(0b01010101, a, b);
        let e = _mm_set_ph(0., 9., 0., 9., 0., 9., 0., 9.);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_add_ph() {
        let a = _mm256_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm256_set_ph(
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        );
        let r = _mm256_add_ph(a, b);
        let e = _mm256_set1_ph(17.0);
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_mask_add_ph() {
        let a = _mm256_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm256_set_ph(
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        );
        let src = _mm256_set_ph(
            18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33.,
        );
        let r = _mm256_mask_add_ph(src, 0b0101010101010101, a, b);
        let e = _mm256_set_ph(
            18., 17., 20., 17., 22., 17., 24., 17., 26., 17., 28., 17., 30., 17., 32., 17.,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_maskz_add_ph() {
        let a = _mm256_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm256_set_ph(
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        );
        let r = _mm256_maskz_add_ph(0b0101010101010101, a, b);
        let e = _mm256_set_ph(
            0., 17., 0., 17., 0., 17., 0., 17., 0., 17., 0., 17., 0., 17., 0., 17.,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_add_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let r = _mm512_add_ph(a, b);
        let e = _mm512_set1_ph(33.0);
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_mask_add_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let src = _mm512_set_ph(
            34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50.,
            51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65.,
        );
        let r = _mm512_mask_add_ph(src, 0b01010101010101010101010101010101, a, b);
        let e = _mm512_set_ph(
            34., 33., 36., 33., 38., 33., 40., 33., 42., 33., 44., 33., 46., 33., 48., 33., 50.,
            33., 52., 33., 54., 33., 56., 33., 58., 33., 60., 33., 62., 33., 64., 33.,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_maskz_add_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let r = _mm512_maskz_add_ph(0b01010101010101010101010101010101, a, b);
        let e = _mm512_set_ph(
            0., 33., 0., 33., 0., 33., 0., 33., 0., 33., 0., 33., 0., 33., 0., 33., 0., 33., 0.,
            33., 0., 33., 0., 33., 0., 33., 0., 33., 0., 33., 0., 33.,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_add_round_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let r = _mm512_add_round_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a, b);
        let e = _mm512_set1_ph(33.0);
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_mask_add_round_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let src = _mm512_set_ph(
            34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50.,
            51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65.,
        );
        let r = _mm512_mask_add_round_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            src,
            0b01010101010101010101010101010101,
            a,
            b,
        );
        let e = _mm512_set_ph(
            34., 33., 36., 33., 38., 33., 40., 33., 42., 33., 44., 33., 46., 33., 48., 33., 50.,
            33., 52., 33., 54., 33., 56., 33., 58., 33., 60., 33., 62., 33., 64., 33.,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_maskz_add_round_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let r = _mm512_maskz_add_round_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            0b01010101010101010101010101010101,
            a,
            b,
        );
        let e = _mm512_set_ph(
            0., 33., 0., 33., 0., 33., 0., 33., 0., 33., 0., 33., 0., 33., 0., 33., 0., 33., 0.,
            33., 0., 33., 0., 33., 0., 33., 0., 33., 0., 33., 0., 33.,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_add_round_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_add_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a, b);
        let e = _mm_set_sh(3.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_mask_add_round_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let src = _mm_set_sh(4.0);
        let r = _mm_mask_add_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            src, 0, a, b,
        );
        let e = _mm_set_sh(4.0);
        assert_eq_m128h(r, e);
        let r = _mm_mask_add_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            src, 1, a, b,
        );
        let e = _mm_set_sh(3.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_maskz_add_round_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r =
            _mm_maskz_add_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(0, a, b);
        let e = _mm_set_sh(0.0);
        assert_eq_m128h(r, e);
        let r =
            _mm_maskz_add_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(1, a, b);
        let e = _mm_set_sh(3.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_add_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_add_sh(a, b);
        let e = _mm_set_sh(3.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_mask_add_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let src = _mm_set_sh(4.0);
        let r = _mm_mask_add_sh(src, 0, a, b);
        let e = _mm_set_sh(4.0);
        assert_eq_m128h(r, e);
        let r = _mm_mask_add_sh(src, 1, a, b);
        let e = _mm_set_sh(3.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_maskz_add_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_maskz_add_sh(0, a, b);
        let e = _mm_set_sh(0.0);
        assert_eq_m128h(r, e);
        let r = _mm_maskz_add_sh(1, a, b);
        let e = _mm_set_sh(3.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_sub_ph() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_set_ph(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        let r = _mm_sub_ph(a, b);
        let e = _mm_set_ph(-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_mask_sub_ph() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_set_ph(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        let src = _mm_set_ph(10., 11., 12., 13., 14., 15., 16., 17.);
        let r = _mm_mask_sub_ph(src, 0b01010101, a, b);
        let e = _mm_set_ph(10., -5., 12., -1., 14., 3., 16., 7.);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_maskz_sub_ph() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_set_ph(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        let r = _mm_maskz_sub_ph(0b01010101, a, b);
        let e = _mm_set_ph(0., -5., 0., -1., 0., 3., 0., 7.);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_sub_ph() {
        let a = _mm256_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm256_set_ph(
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        );
        let r = _mm256_sub_ph(a, b);
        let e = _mm256_set_ph(
            -15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0,
            15.0,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_mask_sub_ph() {
        let a = _mm256_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm256_set_ph(
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        );
        let src = _mm256_set_ph(
            18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33.,
        );
        let r = _mm256_mask_sub_ph(src, 0b0101010101010101, a, b);
        let e = _mm256_set_ph(
            18., -13., 20., -9., 22., -5., 24., -1., 26., 3., 28., 7., 30., 11., 32., 15.,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_maskz_sub_ph() {
        let a = _mm256_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm256_set_ph(
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        );
        let r = _mm256_maskz_sub_ph(0b0101010101010101, a, b);
        let e = _mm256_set_ph(
            0., -13., 0., -9., 0., -5., 0., -1., 0., 3., 0., 7., 0., 11., 0., 15.,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_sub_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let r = _mm512_sub_ph(a, b);
        let e = _mm512_set_ph(
            -31.0, -29.0, -27.0, -25.0, -23.0, -21.0, -19.0, -17.0, -15.0, -13.0, -11.0, -9.0,
            -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0,
            23.0, 25.0, 27.0, 29.0, 31.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_mask_sub_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let src = _mm512_set_ph(
            34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50.,
            51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65.,
        );
        let r = _mm512_mask_sub_ph(src, 0b01010101010101010101010101010101, a, b);
        let e = _mm512_set_ph(
            34., -29., 36., -25., 38., -21., 40., -17., 42., -13., 44., -9., 46., -5., 48., -1.,
            50., 3., 52., 7., 54., 11., 56., 15., 58., 19., 60., 23., 62., 27., 64., 31.,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_maskz_sub_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let r = _mm512_maskz_sub_ph(0b01010101010101010101010101010101, a, b);
        let e = _mm512_set_ph(
            0., -29., 0., -25., 0., -21., 0., -17., 0., -13., 0., -9., 0., -5., 0., -1., 0., 3.,
            0., 7., 0., 11., 0., 15., 0., 19., 0., 23., 0., 27., 0., 31.,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_sub_round_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let r = _mm512_sub_round_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a, b);
        let e = _mm512_set_ph(
            -31.0, -29.0, -27.0, -25.0, -23.0, -21.0, -19.0, -17.0, -15.0, -13.0, -11.0, -9.0,
            -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0,
            23.0, 25.0, 27.0, 29.0, 31.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_mask_sub_round_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let src = _mm512_set_ph(
            34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50.,
            51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65.,
        );
        let r = _mm512_mask_sub_round_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            src,
            0b01010101010101010101010101010101,
            a,
            b,
        );
        let e = _mm512_set_ph(
            34., -29., 36., -25., 38., -21., 40., -17., 42., -13., 44., -9., 46., -5., 48., -1.,
            50., 3., 52., 7., 54., 11., 56., 15., 58., 19., 60., 23., 62., 27., 64., 31.,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_maskz_sub_round_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let r = _mm512_maskz_sub_round_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            0b01010101010101010101010101010101,
            a,
            b,
        );
        let e = _mm512_set_ph(
            0., -29., 0., -25., 0., -21., 0., -17., 0., -13., 0., -9., 0., -5., 0., -1., 0., 3.,
            0., 7., 0., 11., 0., 15., 0., 19., 0., 23., 0., 27., 0., 31.,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_sub_round_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_sub_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a, b);
        let e = _mm_set_sh(-1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_mask_sub_round_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let src = _mm_set_sh(4.0);
        let r = _mm_mask_sub_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            src, 0, a, b,
        );
        let e = _mm_set_sh(4.0);
        assert_eq_m128h(r, e);
        let r = _mm_mask_sub_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            src, 1, a, b,
        );
        let e = _mm_set_sh(-1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_maskz_sub_round_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r =
            _mm_maskz_sub_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(0, a, b);
        let e = _mm_set_sh(0.0);
        assert_eq_m128h(r, e);
        let r =
            _mm_maskz_sub_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(1, a, b);
        let e = _mm_set_sh(-1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_sub_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_sub_sh(a, b);
        let e = _mm_set_sh(-1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_mask_sub_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let src = _mm_set_sh(4.0);
        let r = _mm_mask_sub_sh(src, 0, a, b);
        let e = _mm_set_sh(4.0);
        assert_eq_m128h(r, e);
        let r = _mm_mask_sub_sh(src, 1, a, b);
        let e = _mm_set_sh(-1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_maskz_sub_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_maskz_sub_sh(0, a, b);
        let e = _mm_set_sh(0.0);
        assert_eq_m128h(r, e);
        let r = _mm_maskz_sub_sh(1, a, b);
        let e = _mm_set_sh(-1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_mul_ph() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_set_ph(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        let r = _mm_mul_ph(a, b);
        let e = _mm_set_ph(8.0, 14.0, 18.0, 20.0, 20.0, 18.0, 14.0, 8.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_mask_mul_ph() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_set_ph(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        let src = _mm_set_ph(10., 11., 12., 13., 14., 15., 16., 17.);
        let r = _mm_mask_mul_ph(src, 0b01010101, a, b);
        let e = _mm_set_ph(10., 14., 12., 20., 14., 18., 16., 8.);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_maskz_mul_ph() {
        let a = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_set_ph(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        let r = _mm_maskz_mul_ph(0b01010101, a, b);
        let e = _mm_set_ph(0., 14., 0., 20., 0., 18., 0., 8.);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_mul_ph() {
        let a = _mm256_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm256_set_ph(
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        );
        let r = _mm256_mul_ph(a, b);
        let e = _mm256_set_ph(
            16.0, 30.0, 42.0, 52.0, 60.0, 66.0, 70.0, 72.0, 72.0, 70.0, 66.0, 60.0, 52.0, 42.0,
            30.0, 16.0,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_mask_mul_ph() {
        let a = _mm256_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm256_set_ph(
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        );
        let src = _mm256_set_ph(
            18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33.,
        );
        let r = _mm256_mask_mul_ph(src, 0b0101010101010101, a, b);
        let e = _mm256_set_ph(
            18., 30., 20., 52., 22., 66., 24., 72., 26., 70., 28., 60., 30., 42., 32., 16.,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_maskz_mul_ph() {
        let a = _mm256_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm256_set_ph(
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        );
        let r = _mm256_maskz_mul_ph(0b0101010101010101, a, b);
        let e = _mm256_set_ph(
            0., 30., 0., 52., 0., 66., 0., 72., 0., 70., 0., 60., 0., 42., 0., 16.,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_mul_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let r = _mm512_mul_ph(a, b);
        let e = _mm512_set_ph(
            32.0, 62.0, 90.0, 116.0, 140.0, 162.0, 182.0, 200.0, 216.0, 230.0, 242.0, 252.0, 260.0,
            266.0, 270.0, 272.0, 272.0, 270.0, 266.0, 260.0, 252.0, 242.0, 230.0, 216.0, 200.0,
            182.0, 162.0, 140.0, 116.0, 90.0, 62.0, 32.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_mask_mul_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let src = _mm512_set_ph(
            34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50.,
            51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65.,
        );
        let r = _mm512_mask_mul_ph(src, 0b01010101010101010101010101010101, a, b);
        let e = _mm512_set_ph(
            34., 62., 36., 116., 38., 162., 40., 200., 42., 230., 44., 252., 46., 266., 48., 272.,
            50., 270., 52., 260., 54., 242., 56., 216., 58., 182., 60., 140., 62., 90., 64., 32.,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_maskz_mul_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let r = _mm512_maskz_mul_ph(0b01010101010101010101010101010101, a, b);
        let e = _mm512_set_ph(
            0., 62., 0., 116., 0., 162., 0., 200., 0., 230., 0., 252., 0., 266., 0., 272., 0.,
            270., 0., 260., 0., 242., 0., 216., 0., 182., 0., 140., 0., 90., 0., 32.,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_mul_round_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let r = _mm512_mul_round_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a, b);
        let e = _mm512_set_ph(
            32.0, 62.0, 90.0, 116.0, 140.0, 162.0, 182.0, 200.0, 216.0, 230.0, 242.0, 252.0, 260.0,
            266.0, 270.0, 272.0, 272.0, 270.0, 266.0, 260.0, 252.0, 242.0, 230.0, 216.0, 200.0,
            182.0, 162.0, 140.0, 116.0, 90.0, 62.0, 32.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_mask_mul_round_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let src = _mm512_set_ph(
            34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50.,
            51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65.,
        );
        let r = _mm512_mask_mul_round_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            src,
            0b01010101010101010101010101010101,
            a,
            b,
        );
        let e = _mm512_set_ph(
            34., 62., 36., 116., 38., 162., 40., 200., 42., 230., 44., 252., 46., 266., 48., 272.,
            50., 270., 52., 260., 54., 242., 56., 216., 58., 182., 60., 140., 62., 90., 64., 32.,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_maskz_mul_round_ph() {
        let a = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        let r = _mm512_maskz_mul_round_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            0b01010101010101010101010101010101,
            a,
            b,
        );
        let e = _mm512_set_ph(
            0., 62., 0., 116., 0., 162., 0., 200., 0., 230., 0., 252., 0., 266., 0., 272., 0.,
            270., 0., 260., 0., 242., 0., 216., 0., 182., 0., 140., 0., 90., 0., 32.,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_mul_round_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_mul_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a, b);
        let e = _mm_set_sh(2.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_mask_mul_round_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let src = _mm_set_sh(4.0);
        let r = _mm_mask_mul_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            src, 0, a, b,
        );
        let e = _mm_set_sh(4.0);
        assert_eq_m128h(r, e);
        let r = _mm_mask_mul_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            src, 1, a, b,
        );
        let e = _mm_set_sh(2.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_maskz_mul_round_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r =
            _mm_maskz_mul_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(0, a, b);
        let e = _mm_set_sh(0.0);
        assert_eq_m128h(r, e);
        let r =
            _mm_maskz_mul_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(1, a, b);
        let e = _mm_set_sh(2.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_mul_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_mul_sh(a, b);
        let e = _mm_set_sh(2.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_mask_mul_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let src = _mm_set_sh(4.0);
        let r = _mm_mask_mul_sh(src, 0, a, b);
        let e = _mm_set_sh(4.0);
        assert_eq_m128h(r, e);
        let r = _mm_mask_mul_sh(src, 1, a, b);
        let e = _mm_set_sh(2.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_maskz_mul_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_maskz_mul_sh(0, a, b);
        let e = _mm_set_sh(0.0);
        assert_eq_m128h(r, e);
        let r = _mm_maskz_mul_sh(1, a, b);
        let e = _mm_set_sh(2.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_div_ph() {
        let a = _mm_set1_ph(1.0);
        let b = _mm_set1_ph(2.0);
        let r = _mm_div_ph(a, b);
        let e = _mm_set1_ph(0.5);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_mask_div_ph() {
        let a = _mm_set1_ph(1.0);
        let b = _mm_set1_ph(2.0);
        let src = _mm_set_ph(4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0);
        let r = _mm_mask_div_ph(src, 0b01010101, a, b);
        let e = _mm_set_ph(4.0, 0.5, 6.0, 0.5, 8.0, 0.5, 10.0, 0.5);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_maskz_div_ph() {
        let a = _mm_set1_ph(1.0);
        let b = _mm_set1_ph(2.0);
        let r = _mm_maskz_div_ph(0b01010101, a, b);
        let e = _mm_set_ph(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_div_ph() {
        let a = _mm256_set1_ph(1.0);
        let b = _mm256_set1_ph(2.0);
        let r = _mm256_div_ph(a, b);
        let e = _mm256_set1_ph(0.5);
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_mask_div_ph() {
        let a = _mm256_set1_ph(1.0);
        let b = _mm256_set1_ph(2.0);
        let src = _mm256_set_ph(
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0,
        );
        let r = _mm256_mask_div_ph(src, 0b0101010101010101, a, b);
        let e = _mm256_set_ph(
            4.0, 0.5, 6.0, 0.5, 8.0, 0.5, 10.0, 0.5, 12.0, 0.5, 14.0, 0.5, 16.0, 0.5, 18.0, 0.5,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_maskz_div_ph() {
        let a = _mm256_set1_ph(1.0);
        let b = _mm256_set1_ph(2.0);
        let r = _mm256_maskz_div_ph(0b0101010101010101, a, b);
        let e = _mm256_set_ph(
            0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_div_ph() {
        let a = _mm512_set1_ph(1.0);
        let b = _mm512_set1_ph(2.0);
        let r = _mm512_div_ph(a, b);
        let e = _mm512_set1_ph(0.5);
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_mask_div_ph() {
        let a = _mm512_set1_ph(1.0);
        let b = _mm512_set1_ph(2.0);
        let src = _mm512_set_ph(
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
            33.0, 34.0, 35.0,
        );
        let r = _mm512_mask_div_ph(src, 0b01010101010101010101010101010101, a, b);
        let e = _mm512_set_ph(
            4.0, 0.5, 6.0, 0.5, 8.0, 0.5, 10.0, 0.5, 12.0, 0.5, 14.0, 0.5, 16.0, 0.5, 18.0, 0.5,
            20.0, 0.5, 22.0, 0.5, 24.0, 0.5, 26.0, 0.5, 28.0, 0.5, 30.0, 0.5, 32.0, 0.5, 34.0, 0.5,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_maskz_div_ph() {
        let a = _mm512_set1_ph(1.0);
        let b = _mm512_set1_ph(2.0);
        let r = _mm512_maskz_div_ph(0b01010101010101010101010101010101, a, b);
        let e = _mm512_set_ph(
            0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0,
            0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_div_round_ph() {
        let a = _mm512_set1_ph(1.0);
        let b = _mm512_set1_ph(2.0);
        let r = _mm512_div_round_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a, b);
        let e = _mm512_set1_ph(0.5);
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_mask_div_round_ph() {
        let a = _mm512_set1_ph(1.0);
        let b = _mm512_set1_ph(2.0);
        let src = _mm512_set_ph(
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
            33.0, 34.0, 35.0,
        );
        let r = _mm512_mask_div_round_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            src,
            0b01010101010101010101010101010101,
            a,
            b,
        );
        let e = _mm512_set_ph(
            4.0, 0.5, 6.0, 0.5, 8.0, 0.5, 10.0, 0.5, 12.0, 0.5, 14.0, 0.5, 16.0, 0.5, 18.0, 0.5,
            20.0, 0.5, 22.0, 0.5, 24.0, 0.5, 26.0, 0.5, 28.0, 0.5, 30.0, 0.5, 32.0, 0.5, 34.0, 0.5,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_maskz_div_round_ph() {
        let a = _mm512_set1_ph(1.0);
        let b = _mm512_set1_ph(2.0);
        let r = _mm512_maskz_div_round_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            0b01010101010101010101010101010101,
            a,
            b,
        );
        let e = _mm512_set_ph(
            0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0,
            0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_div_round_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_div_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a, b);
        let e = _mm_set_sh(0.5);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_mask_div_round_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let src = _mm_set_sh(4.0);
        let r = _mm_mask_div_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            src, 0, a, b,
        );
        let e = _mm_set_sh(4.0);
        assert_eq_m128h(r, e);
        let r = _mm_mask_div_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            src, 1, a, b,
        );
        let e = _mm_set_sh(0.5);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_maskz_div_round_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r =
            _mm_maskz_div_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(0, a, b);
        let e = _mm_set_sh(0.0);
        assert_eq_m128h(r, e);
        let r =
            _mm_maskz_div_round_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(1, a, b);
        let e = _mm_set_sh(0.5);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_div_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_div_sh(a, b);
        let e = _mm_set_sh(0.5);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_mask_div_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let src = _mm_set_sh(4.0);
        let r = _mm_mask_div_sh(src, 0, a, b);
        let e = _mm_set_sh(4.0);
        assert_eq_m128h(r, e);
        let r = _mm_mask_div_sh(src, 1, a, b);
        let e = _mm_set_sh(0.5);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_maskz_div_sh() {
        let a = _mm_set_sh(1.0);
        let b = _mm_set_sh(2.0);
        let r = _mm_maskz_div_sh(0, a, b);
        let e = _mm_set_sh(0.0);
        assert_eq_m128h(r, e);
        let r = _mm_maskz_div_sh(1, a, b);
        let e = _mm_set_sh(0.5);
        assert_eq_m128h(r, e);
    }
}
