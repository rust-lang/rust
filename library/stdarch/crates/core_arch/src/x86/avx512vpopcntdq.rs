//! Vectorized Population Count Instructions for Double- and Quadwords (VPOPCNTDQ)
//!
//! The intrinsics here correspond to those in the `immintrin.h` C header.
//!
//! The reference is [Intel 64 and IA-32 Architectures Software Developer's
//! Manual Volume 2: Instruction Set Reference, A-Z][intel64_ref].
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf

use crate::core_arch::simd::*;
use crate::core_arch::x86::__m128i;
use crate::core_arch::x86::__m256i;
use crate::core_arch::x86::__m512i;
use crate::core_arch::x86::__mmask8;
use crate::core_arch::x86::__mmask16;
use crate::intrinsics::simd::{simd_ctpop, simd_select_bitmask};
use crate::mem::transmute;

#[cfg(test)]
use stdarch_test::assert_instr;

/// For each packed 32-bit integer maps the value to the number of logical 1 bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_popcnt_epi32)
#[inline]
#[target_feature(enable = "avx512vpopcntdq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntd))]
pub fn _mm512_popcnt_epi32(a: __m512i) -> __m512i {
    unsafe { transmute(simd_ctpop(a.as_i32x16())) }
}

/// For each packed 32-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_popcnt_epi32)
#[inline]
#[target_feature(enable = "avx512vpopcntdq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntd))]
pub fn _mm512_maskz_popcnt_epi32(k: __mmask16, a: __m512i) -> __m512i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i32x16()),
            i32x16::ZERO,
        ))
    }
}

/// For each packed 32-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_popcnt_epi32)
#[inline]
#[target_feature(enable = "avx512vpopcntdq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntd))]
pub fn _mm512_mask_popcnt_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i32x16()),
            src.as_i32x16(),
        ))
    }
}

/// For each packed 32-bit integer maps the value to the number of logical 1 bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_popcnt_epi32)
#[inline]
#[target_feature(enable = "avx512vpopcntdq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntd))]
pub fn _mm256_popcnt_epi32(a: __m256i) -> __m256i {
    unsafe { transmute(simd_ctpop(a.as_i32x8())) }
}

/// For each packed 32-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_popcnt_epi32)
#[inline]
#[target_feature(enable = "avx512vpopcntdq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntd))]
pub fn _mm256_maskz_popcnt_epi32(k: __mmask8, a: __m256i) -> __m256i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i32x8()),
            i32x8::ZERO,
        ))
    }
}

/// For each packed 32-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_popcnt_epi32)
#[inline]
#[target_feature(enable = "avx512vpopcntdq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntd))]
pub fn _mm256_mask_popcnt_epi32(src: __m256i, k: __mmask8, a: __m256i) -> __m256i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i32x8()),
            src.as_i32x8(),
        ))
    }
}

/// For each packed 32-bit integer maps the value to the number of logical 1 bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_popcnt_epi32)
#[inline]
#[target_feature(enable = "avx512vpopcntdq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntd))]
pub fn _mm_popcnt_epi32(a: __m128i) -> __m128i {
    unsafe { transmute(simd_ctpop(a.as_i32x4())) }
}

/// For each packed 32-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_popcnt_epi32)
#[inline]
#[target_feature(enable = "avx512vpopcntdq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntd))]
pub fn _mm_maskz_popcnt_epi32(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i32x4()),
            i32x4::ZERO,
        ))
    }
}

/// For each packed 32-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_popcnt_epi32)
#[inline]
#[target_feature(enable = "avx512vpopcntdq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntd))]
pub fn _mm_mask_popcnt_epi32(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i32x4()),
            src.as_i32x4(),
        ))
    }
}

/// For each packed 64-bit integer maps the value to the number of logical 1 bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_popcnt_epi64)
#[inline]
#[target_feature(enable = "avx512vpopcntdq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntq))]
pub fn _mm512_popcnt_epi64(a: __m512i) -> __m512i {
    unsafe { transmute(simd_ctpop(a.as_i64x8())) }
}

/// For each packed 64-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_popcnt_epi64)
#[inline]
#[target_feature(enable = "avx512vpopcntdq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntq))]
pub fn _mm512_maskz_popcnt_epi64(k: __mmask8, a: __m512i) -> __m512i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i64x8()),
            i64x8::ZERO,
        ))
    }
}

/// For each packed 64-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_popcnt_epi64)
#[inline]
#[target_feature(enable = "avx512vpopcntdq")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntq))]
pub fn _mm512_mask_popcnt_epi64(src: __m512i, k: __mmask8, a: __m512i) -> __m512i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i64x8()),
            src.as_i64x8(),
        ))
    }
}

/// For each packed 64-bit integer maps the value to the number of logical 1 bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_popcnt_epi64)
#[inline]
#[target_feature(enable = "avx512vpopcntdq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntq))]
pub fn _mm256_popcnt_epi64(a: __m256i) -> __m256i {
    unsafe { transmute(simd_ctpop(a.as_i64x4())) }
}

/// For each packed 64-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_popcnt_epi64)
#[inline]
#[target_feature(enable = "avx512vpopcntdq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntq))]
pub fn _mm256_maskz_popcnt_epi64(k: __mmask8, a: __m256i) -> __m256i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i64x4()),
            i64x4::ZERO,
        ))
    }
}

/// For each packed 64-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_popcnt_epi64)
#[inline]
#[target_feature(enable = "avx512vpopcntdq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntq))]
pub fn _mm256_mask_popcnt_epi64(src: __m256i, k: __mmask8, a: __m256i) -> __m256i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i64x4()),
            src.as_i64x4(),
        ))
    }
}

/// For each packed 64-bit integer maps the value to the number of logical 1 bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_popcnt_epi64)
#[inline]
#[target_feature(enable = "avx512vpopcntdq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntq))]
pub fn _mm_popcnt_epi64(a: __m128i) -> __m128i {
    unsafe { transmute(simd_ctpop(a.as_i64x2())) }
}

/// For each packed 64-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_popcnt_epi64)
#[inline]
#[target_feature(enable = "avx512vpopcntdq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntq))]
pub fn _mm_maskz_popcnt_epi64(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i64x2()),
            i64x2::ZERO,
        ))
    }
}

/// For each packed 64-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_popcnt_epi64)
#[inline]
#[target_feature(enable = "avx512vpopcntdq,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntq))]
pub fn _mm_mask_popcnt_epi64(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i64x2()),
            src.as_i64x2(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "avx512vpopcntdq,avx512f")]
    unsafe fn test_mm512_popcnt_epi32() {
        let test_data = _mm512_set_epi32(
            0,
            1,
            -1,
            2,
            7,
            0xFF_FE,
            0x7F_FF_FF_FF,
            -100,
            0x40_00_00_00,
            103,
            371,
            552,
            432_948,
            818_826_998,
            255,
            256,
        );
        let actual_result = _mm512_popcnt_epi32(test_data);
        let reference_result =
            _mm512_set_epi32(0, 1, 32, 1, 3, 15, 31, 28, 1, 5, 6, 3, 10, 17, 8, 1);
        assert_eq_m512i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512f")]
    unsafe fn test_mm512_mask_popcnt_epi32() {
        let test_data = _mm512_set_epi32(
            0,
            1,
            -1,
            2,
            7,
            0xFF_FE,
            0x7F_FF_FF_FF,
            -100,
            0x40_00_00_00,
            103,
            371,
            552,
            432_948,
            818_826_998,
            255,
            256,
        );
        let mask = 0xFF_00;
        let actual_result = _mm512_mask_popcnt_epi32(test_data, mask, test_data);
        let reference_result = _mm512_set_epi32(
            0,
            1,
            32,
            1,
            3,
            15,
            31,
            28,
            0x40_00_00_00,
            103,
            371,
            552,
            432_948,
            818_826_998,
            255,
            256,
        );
        assert_eq_m512i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512f")]
    unsafe fn test_mm512_maskz_popcnt_epi32() {
        let test_data = _mm512_set_epi32(
            0,
            1,
            -1,
            2,
            7,
            0xFF_FE,
            0x7F_FF_FF_FF,
            -100,
            0x40_00_00_00,
            103,
            371,
            552,
            432_948,
            818_826_998,
            255,
            256,
        );
        let mask = 0xFF_00;
        let actual_result = _mm512_maskz_popcnt_epi32(mask, test_data);
        let reference_result = _mm512_set_epi32(0, 1, 32, 1, 3, 15, 31, 28, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512f,avx512vl")]
    unsafe fn test_mm256_popcnt_epi32() {
        let test_data = _mm256_set_epi32(0, 1, -1, 2, 7, 0xFF_FE, 0x7F_FF_FF_FF, -100);
        let actual_result = _mm256_popcnt_epi32(test_data);
        let reference_result = _mm256_set_epi32(0, 1, 32, 1, 3, 15, 31, 28);
        assert_eq_m256i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512f,avx512vl")]
    unsafe fn test_mm256_mask_popcnt_epi32() {
        let test_data = _mm256_set_epi32(0, 1, -1, 2, 7, 0xFF_FE, 0x7F_FF_FF_FF, -100);
        let mask = 0xF0;
        let actual_result = _mm256_mask_popcnt_epi32(test_data, mask, test_data);
        let reference_result = _mm256_set_epi32(0, 1, 32, 1, 7, 0xFF_FE, 0x7F_FF_FF_FF, -100);
        assert_eq_m256i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512f,avx512vl")]
    unsafe fn test_mm256_maskz_popcnt_epi32() {
        let test_data = _mm256_set_epi32(0, 1, -1, 2, 7, 0xFF_FE, 0x7F_FF_FF_FF, -100);
        let mask = 0xF0;
        let actual_result = _mm256_maskz_popcnt_epi32(mask, test_data);
        let reference_result = _mm256_set_epi32(0, 1, 32, 1, 0, 0, 0, 0);
        assert_eq_m256i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512f,avx512vl")]
    unsafe fn test_mm_popcnt_epi32() {
        let test_data = _mm_set_epi32(0, 1, -1, -100);
        let actual_result = _mm_popcnt_epi32(test_data);
        let reference_result = _mm_set_epi32(0, 1, 32, 28);
        assert_eq_m128i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512f,avx512vl")]
    unsafe fn test_mm_mask_popcnt_epi32() {
        let test_data = _mm_set_epi32(0, 1, -1, -100);
        let mask = 0xE;
        let actual_result = _mm_mask_popcnt_epi32(test_data, mask, test_data);
        let reference_result = _mm_set_epi32(0, 1, 32, -100);
        assert_eq_m128i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512f,avx512vl")]
    unsafe fn test_mm_maskz_popcnt_epi32() {
        let test_data = _mm_set_epi32(0, 1, -1, -100);
        let mask = 0xE;
        let actual_result = _mm_maskz_popcnt_epi32(mask, test_data);
        let reference_result = _mm_set_epi32(0, 1, 32, 0);
        assert_eq_m128i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512f")]
    unsafe fn test_mm512_popcnt_epi64() {
        let test_data = _mm512_set_epi64(0, 1, -1, 2, 7, 0xFF_FE, 0x7F_FF_FF_FF_FF_FF_FF_FF, -100);
        let actual_result = _mm512_popcnt_epi64(test_data);
        let reference_result = _mm512_set_epi64(0, 1, 64, 1, 3, 15, 63, 60);
        assert_eq_m512i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512f")]
    unsafe fn test_mm512_mask_popcnt_epi64() {
        let test_data = _mm512_set_epi64(0, 1, -1, 2, 7, 0xFF_FE, 0x7F_FF_FF_FF_FF_FF_FF_FF, -100);
        let mask = 0xF0;
        let actual_result = _mm512_mask_popcnt_epi64(test_data, mask, test_data);
        let reference_result =
            _mm512_set_epi64(0, 1, 64, 1, 7, 0xFF_FE, 0x7F_FF_FF_FF_FF_FF_FF_FF, -100);
        assert_eq_m512i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512f")]
    unsafe fn test_mm512_maskz_popcnt_epi64() {
        let test_data = _mm512_set_epi64(0, 1, -1, 2, 7, 0xFF_FE, 0x7F_FF_FF_FF_FF_FF_FF_FF, -100);
        let mask = 0xF0;
        let actual_result = _mm512_maskz_popcnt_epi64(mask, test_data);
        let reference_result = _mm512_set_epi64(0, 1, 64, 1, 0, 0, 0, 0);
        assert_eq_m512i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512vl")]
    unsafe fn test_mm256_popcnt_epi64() {
        let test_data = _mm256_set_epi64x(0, 1, -1, -100);
        let actual_result = _mm256_popcnt_epi64(test_data);
        let reference_result = _mm256_set_epi64x(0, 1, 64, 60);
        assert_eq_m256i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512vl")]
    unsafe fn test_mm256_mask_popcnt_epi64() {
        let test_data = _mm256_set_epi64x(0, 1, -1, -100);
        let mask = 0xE;
        let actual_result = _mm256_mask_popcnt_epi64(test_data, mask, test_data);
        let reference_result = _mm256_set_epi64x(0, 1, 64, -100);
        assert_eq_m256i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512vl")]
    unsafe fn test_mm256_maskz_popcnt_epi64() {
        let test_data = _mm256_set_epi64x(0, 1, -1, -100);
        let mask = 0xE;
        let actual_result = _mm256_maskz_popcnt_epi64(mask, test_data);
        let reference_result = _mm256_set_epi64x(0, 1, 64, 0);
        assert_eq_m256i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512vl")]
    unsafe fn test_mm_popcnt_epi64() {
        let test_data = _mm_set_epi64x(0, 1);
        let actual_result = _mm_popcnt_epi64(test_data);
        let reference_result = _mm_set_epi64x(0, 1);
        assert_eq_m128i(actual_result, reference_result);
        let test_data = _mm_set_epi64x(-1, -100);
        let actual_result = _mm_popcnt_epi64(test_data);
        let reference_result = _mm_set_epi64x(64, 60);
        assert_eq_m128i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512vl")]
    unsafe fn test_mm_mask_popcnt_epi64() {
        let test_data = _mm_set_epi64x(0, -100);
        let mask = 0x2;
        let actual_result = _mm_mask_popcnt_epi64(test_data, mask, test_data);
        let reference_result = _mm_set_epi64x(0, -100);
        assert_eq_m128i(actual_result, reference_result);
        let test_data = _mm_set_epi64x(-1, 1);
        let mask = 0x2;
        let actual_result = _mm_mask_popcnt_epi64(test_data, mask, test_data);
        let reference_result = _mm_set_epi64x(64, 1);
        assert_eq_m128i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512vpopcntdq,avx512vl")]
    unsafe fn test_mm_maskz_popcnt_epi64() {
        let test_data = _mm_set_epi64x(0, 1);
        let mask = 0x2;
        let actual_result = _mm_maskz_popcnt_epi64(mask, test_data);
        let reference_result = _mm_set_epi64x(0, 0);
        assert_eq_m128i(actual_result, reference_result);
        let test_data = _mm_set_epi64x(-1, -100);
        let mask = 0x2;
        let actual_result = _mm_maskz_popcnt_epi64(mask, test_data);
        let reference_result = _mm_set_epi64x(64, 0);
        assert_eq_m128i(actual_result, reference_result);
    }
}
