use crate::{
    core_arch::{simd::*, x86::*},
    intrinsics::simd::*,
    ptr,
};

use core::hint::unreachable_unchecked;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_abs_epi16&expand=30)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub fn _mm512_abs_epi16(a: __m512i) -> __m512i {
    unsafe {
        let a = a.as_i16x32();
        let cmp: i16x32 = simd_gt(a, i16x32::ZERO);
        transmute(simd_select(cmp, a, simd_neg(a)))
    }
}

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_abs_epi16&expand=31)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub fn _mm512_mask_abs_epi16(src: __m512i, k: __mmask32, a: __m512i) -> __m512i {
    unsafe {
        let abs = _mm512_abs_epi16(a).as_i16x32();
        transmute(simd_select_bitmask(k, abs, src.as_i16x32()))
    }
}

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_abs_epi16&expand=32)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub fn _mm512_maskz_abs_epi16(k: __mmask32, a: __m512i) -> __m512i {
    unsafe {
        let abs = _mm512_abs_epi16(a).as_i16x32();
        transmute(simd_select_bitmask(k, abs, i16x32::ZERO))
    }
}

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_abs_epi16&expand=28)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub fn _mm256_mask_abs_epi16(src: __m256i, k: __mmask16, a: __m256i) -> __m256i {
    unsafe {
        let abs = _mm256_abs_epi16(a).as_i16x16();
        transmute(simd_select_bitmask(k, abs, src.as_i16x16()))
    }
}

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_abs_epi16&expand=29)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub fn _mm256_maskz_abs_epi16(k: __mmask16, a: __m256i) -> __m256i {
    unsafe {
        let abs = _mm256_abs_epi16(a).as_i16x16();
        transmute(simd_select_bitmask(k, abs, i16x16::ZERO))
    }
}

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_abs_epi16&expand=25)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub fn _mm_mask_abs_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let abs = _mm_abs_epi16(a).as_i16x8();
        transmute(simd_select_bitmask(k, abs, src.as_i16x8()))
    }
}

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_abs_epi16&expand=26)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub fn _mm_maskz_abs_epi16(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let abs = _mm_abs_epi16(a).as_i16x8();
        transmute(simd_select_bitmask(k, abs, i16x8::ZERO))
    }
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_abs_epi8&expand=57)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub fn _mm512_abs_epi8(a: __m512i) -> __m512i {
    unsafe {
        let a = a.as_i8x64();
        let cmp: i8x64 = simd_gt(a, i8x64::ZERO);
        transmute(simd_select(cmp, a, simd_neg(a)))
    }
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_abs_epi8&expand=58)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub fn _mm512_mask_abs_epi8(src: __m512i, k: __mmask64, a: __m512i) -> __m512i {
    unsafe {
        let abs = _mm512_abs_epi8(a).as_i8x64();
        transmute(simd_select_bitmask(k, abs, src.as_i8x64()))
    }
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_abs_epi8&expand=59)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub fn _mm512_maskz_abs_epi8(k: __mmask64, a: __m512i) -> __m512i {
    unsafe {
        let abs = _mm512_abs_epi8(a).as_i8x64();
        transmute(simd_select_bitmask(k, abs, i8x64::ZERO))
    }
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_abs_epi8&expand=55)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub fn _mm256_mask_abs_epi8(src: __m256i, k: __mmask32, a: __m256i) -> __m256i {
    unsafe {
        let abs = _mm256_abs_epi8(a).as_i8x32();
        transmute(simd_select_bitmask(k, abs, src.as_i8x32()))
    }
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_abs_epi8&expand=56)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub fn _mm256_maskz_abs_epi8(k: __mmask32, a: __m256i) -> __m256i {
    unsafe {
        let abs = _mm256_abs_epi8(a).as_i8x32();
        transmute(simd_select_bitmask(k, abs, i8x32::ZERO))
    }
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set)
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_abs_epi8&expand=52)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub fn _mm_mask_abs_epi8(src: __m128i, k: __mmask16, a: __m128i) -> __m128i {
    unsafe {
        let abs = _mm_abs_epi8(a).as_i8x16();
        transmute(simd_select_bitmask(k, abs, src.as_i8x16()))
    }
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_abs_epi8&expand=53)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub fn _mm_maskz_abs_epi8(k: __mmask16, a: __m128i) -> __m128i {
    unsafe {
        let abs = _mm_abs_epi8(a).as_i8x16();
        transmute(simd_select_bitmask(k, abs, i8x16::ZERO))
    }
}

/// Add packed 16-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_add_epi16&expand=91)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub fn _mm512_add_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_add(a.as_i16x32(), b.as_i16x32())) }
}

/// Add packed 16-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_add_epi16&expand=92)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub fn _mm512_mask_add_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let add = _mm512_add_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, add, src.as_i16x32()))
    }
}

/// Add packed 16-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_add_epi16&expand=93)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub fn _mm512_maskz_add_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let add = _mm512_add_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, add, i16x32::ZERO))
    }
}

/// Add packed 16-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_add_epi16&expand=89)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub fn _mm256_mask_add_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let add = _mm256_add_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, add, src.as_i16x16()))
    }
}

/// Add packed 16-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_add_epi16&expand=90)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub fn _mm256_maskz_add_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let add = _mm256_add_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, add, i16x16::ZERO))
    }
}

/// Add packed 16-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_add_epi16&expand=86)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub fn _mm_mask_add_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let add = _mm_add_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, add, src.as_i16x8()))
    }
}

/// Add packed 16-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_add_epi16&expand=87)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub fn _mm_maskz_add_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let add = _mm_add_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, add, i16x8::ZERO))
    }
}

/// Add packed 8-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_add_epi8&expand=118)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub fn _mm512_add_epi8(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_add(a.as_i8x64(), b.as_i8x64())) }
}

/// Add packed 8-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_add_epi8&expand=119)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub fn _mm512_mask_add_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let add = _mm512_add_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, add, src.as_i8x64()))
    }
}

/// Add packed 8-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_add_epi8&expand=120)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub fn _mm512_maskz_add_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let add = _mm512_add_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, add, i8x64::ZERO))
    }
}

/// Add packed 8-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_add_epi8&expand=116)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub fn _mm256_mask_add_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let add = _mm256_add_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, add, src.as_i8x32()))
    }
}

/// Add packed 8-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_add_epi8&expand=117)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub fn _mm256_maskz_add_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let add = _mm256_add_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, add, i8x32::ZERO))
    }
}

/// Add packed 8-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_add_epi8&expand=113)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub fn _mm_mask_add_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let add = _mm_add_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, add, src.as_i8x16()))
    }
}

/// Add packed 8-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_add_epi8&expand=114)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub fn _mm_maskz_add_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let add = _mm_add_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, add, i8x16::ZERO))
    }
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_adds_epu16&expand=197)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub fn _mm512_adds_epu16(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_saturating_add(a.as_u16x32(), b.as_u16x32())) }
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_adds_epu16&expand=198)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub fn _mm512_mask_adds_epu16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let add = _mm512_adds_epu16(a, b).as_u16x32();
        transmute(simd_select_bitmask(k, add, src.as_u16x32()))
    }
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_adds_epu16&expand=199)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub fn _mm512_maskz_adds_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let add = _mm512_adds_epu16(a, b).as_u16x32();
        transmute(simd_select_bitmask(k, add, u16x32::ZERO))
    }
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_adds_epu16&expand=195)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub fn _mm256_mask_adds_epu16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let add = _mm256_adds_epu16(a, b).as_u16x16();
        transmute(simd_select_bitmask(k, add, src.as_u16x16()))
    }
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_adds_epu16&expand=196)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub fn _mm256_maskz_adds_epu16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let add = _mm256_adds_epu16(a, b).as_u16x16();
        transmute(simd_select_bitmask(k, add, u16x16::ZERO))
    }
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_adds_epu16&expand=192)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub fn _mm_mask_adds_epu16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let add = _mm_adds_epu16(a, b).as_u16x8();
        transmute(simd_select_bitmask(k, add, src.as_u16x8()))
    }
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_adds_epu16&expand=193)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub fn _mm_maskz_adds_epu16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let add = _mm_adds_epu16(a, b).as_u16x8();
        transmute(simd_select_bitmask(k, add, u16x8::ZERO))
    }
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_adds_epu8&expand=206)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub fn _mm512_adds_epu8(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_saturating_add(a.as_u8x64(), b.as_u8x64())) }
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_adds_epu8&expand=207)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub fn _mm512_mask_adds_epu8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let add = _mm512_adds_epu8(a, b).as_u8x64();
        transmute(simd_select_bitmask(k, add, src.as_u8x64()))
    }
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_adds_epu8&expand=208)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub fn _mm512_maskz_adds_epu8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let add = _mm512_adds_epu8(a, b).as_u8x64();
        transmute(simd_select_bitmask(k, add, u8x64::ZERO))
    }
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_adds_epu8&expand=204)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub fn _mm256_mask_adds_epu8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let add = _mm256_adds_epu8(a, b).as_u8x32();
        transmute(simd_select_bitmask(k, add, src.as_u8x32()))
    }
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_adds_epu8&expand=205)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub fn _mm256_maskz_adds_epu8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let add = _mm256_adds_epu8(a, b).as_u8x32();
        transmute(simd_select_bitmask(k, add, u8x32::ZERO))
    }
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_adds_epu8&expand=201)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub fn _mm_mask_adds_epu8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let add = _mm_adds_epu8(a, b).as_u8x16();
        transmute(simd_select_bitmask(k, add, src.as_u8x16()))
    }
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_adds_epu8&expand=202)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub fn _mm_maskz_adds_epu8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let add = _mm_adds_epu8(a, b).as_u8x16();
        transmute(simd_select_bitmask(k, add, u8x16::ZERO))
    }
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_adds_epi16&expand=179)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub fn _mm512_adds_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_saturating_add(a.as_i16x32(), b.as_i16x32())) }
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_adds_epi16&expand=180)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub fn _mm512_mask_adds_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let add = _mm512_adds_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, add, src.as_i16x32()))
    }
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_adds_epi16&expand=181)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub fn _mm512_maskz_adds_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let add = _mm512_adds_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, add, i16x32::ZERO))
    }
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_adds_epi16&expand=177)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub fn _mm256_mask_adds_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let add = _mm256_adds_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, add, src.as_i16x16()))
    }
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_adds_epi16&expand=178)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub fn _mm256_maskz_adds_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let add = _mm256_adds_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, add, i16x16::ZERO))
    }
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_adds_epi16&expand=174)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub fn _mm_mask_adds_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let add = _mm_adds_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, add, src.as_i16x8()))
    }
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_adds_epi16&expand=175)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub fn _mm_maskz_adds_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let add = _mm_adds_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, add, i16x8::ZERO))
    }
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_adds_epi8&expand=188)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub fn _mm512_adds_epi8(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_saturating_add(a.as_i8x64(), b.as_i8x64())) }
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_adds_epi8&expand=189)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub fn _mm512_mask_adds_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let add = _mm512_adds_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, add, src.as_i8x64()))
    }
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_adds_epi8&expand=190)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub fn _mm512_maskz_adds_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let add = _mm512_adds_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, add, i8x64::ZERO))
    }
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_adds_epi8&expand=186)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub fn _mm256_mask_adds_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let add = _mm256_adds_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, add, src.as_i8x32()))
    }
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_adds_epi8&expand=187)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub fn _mm256_maskz_adds_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let add = _mm256_adds_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, add, i8x32::ZERO))
    }
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_adds_epi8&expand=183)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub fn _mm_mask_adds_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let add = _mm_adds_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, add, src.as_i8x16()))
    }
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_adds_epi8&expand=184)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub fn _mm_maskz_adds_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let add = _mm_adds_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, add, i8x16::ZERO))
    }
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_sub_epi16&expand=5685)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub fn _mm512_sub_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_sub(a.as_i16x32(), b.as_i16x32())) }
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_sub_epi16&expand=5683)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub fn _mm512_mask_sub_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let sub = _mm512_sub_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, sub, src.as_i16x32()))
    }
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_sub_epi16&expand=5684)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub fn _mm512_maskz_sub_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let sub = _mm512_sub_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, sub, i16x32::ZERO))
    }
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_sub_epi16&expand=5680)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub fn _mm256_mask_sub_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let sub = _mm256_sub_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, sub, src.as_i16x16()))
    }
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_sub_epi16&expand=5681)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub fn _mm256_maskz_sub_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let sub = _mm256_sub_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, sub, i16x16::ZERO))
    }
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_sub_epi16&expand=5677)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub fn _mm_mask_sub_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let sub = _mm_sub_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, sub, src.as_i16x8()))
    }
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_sub_epi16&expand=5678)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub fn _mm_maskz_sub_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let sub = _mm_sub_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, sub, i16x8::ZERO))
    }
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_sub_epi8&expand=5712)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub fn _mm512_sub_epi8(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_sub(a.as_i8x64(), b.as_i8x64())) }
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_sub_epi8&expand=5710)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub fn _mm512_mask_sub_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let sub = _mm512_sub_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, sub, src.as_i8x64()))
    }
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_sub_epi8&expand=5711)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub fn _mm512_maskz_sub_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let sub = _mm512_sub_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, sub, i8x64::ZERO))
    }
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_sub_epi8&expand=5707)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub fn _mm256_mask_sub_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let sub = _mm256_sub_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, sub, src.as_i8x32()))
    }
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_sub_epi8&expand=5708)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub fn _mm256_maskz_sub_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let sub = _mm256_sub_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, sub, i8x32::ZERO))
    }
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_sub_epi8&expand=5704)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub fn _mm_mask_sub_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let sub = _mm_sub_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, sub, src.as_i8x16()))
    }
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_sub_epi8&expand=5705)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub fn _mm_maskz_sub_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let sub = _mm_sub_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, sub, i8x16::ZERO))
    }
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_subs_epu16&expand=5793)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub fn _mm512_subs_epu16(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_saturating_sub(a.as_u16x32(), b.as_u16x32())) }
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_subs_epu16&expand=5791)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub fn _mm512_mask_subs_epu16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let sub = _mm512_subs_epu16(a, b).as_u16x32();
        transmute(simd_select_bitmask(k, sub, src.as_u16x32()))
    }
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_subs_epu16&expand=5792)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub fn _mm512_maskz_subs_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let sub = _mm512_subs_epu16(a, b).as_u16x32();
        transmute(simd_select_bitmask(k, sub, u16x32::ZERO))
    }
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_subs_epu16&expand=5788)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub fn _mm256_mask_subs_epu16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let sub = _mm256_subs_epu16(a, b).as_u16x16();
        transmute(simd_select_bitmask(k, sub, src.as_u16x16()))
    }
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_subs_epu16&expand=5789)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub fn _mm256_maskz_subs_epu16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let sub = _mm256_subs_epu16(a, b).as_u16x16();
        transmute(simd_select_bitmask(k, sub, u16x16::ZERO))
    }
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_subs_epu16&expand=5785)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub fn _mm_mask_subs_epu16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let sub = _mm_subs_epu16(a, b).as_u16x8();
        transmute(simd_select_bitmask(k, sub, src.as_u16x8()))
    }
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_subs_epu16&expand=5786)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub fn _mm_maskz_subs_epu16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let sub = _mm_subs_epu16(a, b).as_u16x8();
        transmute(simd_select_bitmask(k, sub, u16x8::ZERO))
    }
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_subs_epu8&expand=5802)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub fn _mm512_subs_epu8(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_saturating_sub(a.as_u8x64(), b.as_u8x64())) }
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_subs_epu8&expand=5800)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub fn _mm512_mask_subs_epu8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let sub = _mm512_subs_epu8(a, b).as_u8x64();
        transmute(simd_select_bitmask(k, sub, src.as_u8x64()))
    }
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_subs_epu8&expand=5801)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub fn _mm512_maskz_subs_epu8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let sub = _mm512_subs_epu8(a, b).as_u8x64();
        transmute(simd_select_bitmask(k, sub, u8x64::ZERO))
    }
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_subs_epu8&expand=5797)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub fn _mm256_mask_subs_epu8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let sub = _mm256_subs_epu8(a, b).as_u8x32();
        transmute(simd_select_bitmask(k, sub, src.as_u8x32()))
    }
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_subs_epu8&expand=5798)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub fn _mm256_maskz_subs_epu8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let sub = _mm256_subs_epu8(a, b).as_u8x32();
        transmute(simd_select_bitmask(k, sub, u8x32::ZERO))
    }
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_subs_epu8&expand=5794)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub fn _mm_mask_subs_epu8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let sub = _mm_subs_epu8(a, b).as_u8x16();
        transmute(simd_select_bitmask(k, sub, src.as_u8x16()))
    }
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_subs_epu8&expand=5795)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub fn _mm_maskz_subs_epu8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let sub = _mm_subs_epu8(a, b).as_u8x16();
        transmute(simd_select_bitmask(k, sub, u8x16::ZERO))
    }
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_subs_epi16&expand=5775)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub fn _mm512_subs_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_saturating_sub(a.as_i16x32(), b.as_i16x32())) }
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_subs_epi16&expand=5773)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub fn _mm512_mask_subs_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let sub = _mm512_subs_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, sub, src.as_i16x32()))
    }
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_subs_epi16&expand=5774)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub fn _mm512_maskz_subs_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let sub = _mm512_subs_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, sub, i16x32::ZERO))
    }
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_subs_epi16&expand=5770)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub fn _mm256_mask_subs_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let sub = _mm256_subs_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, sub, src.as_i16x16()))
    }
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_subs_epi16&expand=5771)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub fn _mm256_maskz_subs_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let sub = _mm256_subs_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, sub, i16x16::ZERO))
    }
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_subs_epi16&expand=5767)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub fn _mm_mask_subs_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let sub = _mm_subs_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, sub, src.as_i16x8()))
    }
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_subs_epi16&expand=5768)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub fn _mm_maskz_subs_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let sub = _mm_subs_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, sub, i16x8::ZERO))
    }
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_subs_epi8&expand=5784)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub fn _mm512_subs_epi8(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_saturating_sub(a.as_i8x64(), b.as_i8x64())) }
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_subs_epi8&expand=5782)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub fn _mm512_mask_subs_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let sub = _mm512_subs_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, sub, src.as_i8x64()))
    }
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_subs_epi8&expand=5783)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub fn _mm512_maskz_subs_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let sub = _mm512_subs_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, sub, i8x64::ZERO))
    }
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_subs_epi8&expand=5779)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub fn _mm256_mask_subs_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let sub = _mm256_subs_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, sub, src.as_i8x32()))
    }
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_subs_epi8&expand=5780)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub fn _mm256_maskz_subs_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let sub = _mm256_subs_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, sub, i8x32::ZERO))
    }
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_subs_epi8&expand=5776)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub fn _mm_mask_subs_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let sub = _mm_subs_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, sub, src.as_i8x16()))
    }
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_subs_epi8&expand=5777)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub fn _mm_maskz_subs_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let sub = _mm_subs_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, sub, i8x16::ZERO))
    }
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mulhi_epu16&expand=3973)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub fn _mm512_mulhi_epu16(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = simd_cast::<_, u32x32>(a.as_u16x32());
        let b = simd_cast::<_, u32x32>(b.as_u16x32());
        let r = simd_shr(simd_mul(a, b), u32x32::splat(16));
        transmute(simd_cast::<u32x32, u16x32>(r))
    }
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_mulhi_epu16&expand=3971)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub fn _mm512_mask_mulhi_epu16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let mul = _mm512_mulhi_epu16(a, b).as_u16x32();
        transmute(simd_select_bitmask(k, mul, src.as_u16x32()))
    }
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_mulhi_epu16&expand=3972)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub fn _mm512_maskz_mulhi_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let mul = _mm512_mulhi_epu16(a, b).as_u16x32();
        transmute(simd_select_bitmask(k, mul, u16x32::ZERO))
    }
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_mulhi_epu16&expand=3968)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub fn _mm256_mask_mulhi_epu16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let mul = _mm256_mulhi_epu16(a, b).as_u16x16();
        transmute(simd_select_bitmask(k, mul, src.as_u16x16()))
    }
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_mulhi_epu16&expand=3969)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub fn _mm256_maskz_mulhi_epu16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let mul = _mm256_mulhi_epu16(a, b).as_u16x16();
        transmute(simd_select_bitmask(k, mul, u16x16::ZERO))
    }
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_mulhi_epu16&expand=3965)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub fn _mm_mask_mulhi_epu16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let mul = _mm_mulhi_epu16(a, b).as_u16x8();
        transmute(simd_select_bitmask(k, mul, src.as_u16x8()))
    }
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_mulhi_epu16&expand=3966)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub fn _mm_maskz_mulhi_epu16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let mul = _mm_mulhi_epu16(a, b).as_u16x8();
        transmute(simd_select_bitmask(k, mul, u16x8::ZERO))
    }
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mulhi_epi16&expand=3962)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub fn _mm512_mulhi_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = simd_cast::<_, i32x32>(a.as_i16x32());
        let b = simd_cast::<_, i32x32>(b.as_i16x32());
        let r = simd_shr(simd_mul(a, b), i32x32::splat(16));
        transmute(simd_cast::<i32x32, i16x32>(r))
    }
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_mulhi_epi16&expand=3960)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub fn _mm512_mask_mulhi_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let mul = _mm512_mulhi_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, mul, src.as_i16x32()))
    }
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_mulhi_epi16&expand=3961)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub fn _mm512_maskz_mulhi_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let mul = _mm512_mulhi_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, mul, i16x32::ZERO))
    }
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_mulhi_epi16&expand=3957)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub fn _mm256_mask_mulhi_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let mul = _mm256_mulhi_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, mul, src.as_i16x16()))
    }
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_mulhi_epi16&expand=3958)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub fn _mm256_maskz_mulhi_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let mul = _mm256_mulhi_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, mul, i16x16::ZERO))
    }
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_mulhi_epi16&expand=3954)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub fn _mm_mask_mulhi_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let mul = _mm_mulhi_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, mul, src.as_i16x8()))
    }
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_mulhi_epi16&expand=3955)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub fn _mm_maskz_mulhi_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let mul = _mm_mulhi_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, mul, i16x8::ZERO))
    }
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mulhrs_epi16&expand=3986)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub fn _mm512_mulhrs_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(vpmulhrsw(a.as_i16x32(), b.as_i16x32())) }
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_mulhrs_epi16&expand=3984)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub fn _mm512_mask_mulhrs_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let mul = _mm512_mulhrs_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, mul, src.as_i16x32()))
    }
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_mulhrs_epi16&expand=3985)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub fn _mm512_maskz_mulhrs_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let mul = _mm512_mulhrs_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, mul, i16x32::ZERO))
    }
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_mulhrs_epi16&expand=3981)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub fn _mm256_mask_mulhrs_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let mul = _mm256_mulhrs_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, mul, src.as_i16x16()))
    }
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_mulhrs_epi16&expand=3982)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub fn _mm256_maskz_mulhrs_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let mul = _mm256_mulhrs_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, mul, i16x16::ZERO))
    }
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_mulhrs_epi16&expand=3978)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub fn _mm_mask_mulhrs_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let mul = _mm_mulhrs_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, mul, src.as_i16x8()))
    }
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_mulhrs_epi16&expand=3979)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub fn _mm_maskz_mulhrs_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let mul = _mm_mulhrs_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, mul, i16x8::ZERO))
    }
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mullo_epi16&expand=3996)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub fn _mm512_mullo_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_mul(a.as_i16x32(), b.as_i16x32())) }
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_mullo_epi16&expand=3994)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub fn _mm512_mask_mullo_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let mul = _mm512_mullo_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, mul, src.as_i16x32()))
    }
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_mullo_epi16&expand=3995)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub fn _mm512_maskz_mullo_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let mul = _mm512_mullo_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, mul, i16x32::ZERO))
    }
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_mullo_epi16&expand=3991)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub fn _mm256_mask_mullo_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let mul = _mm256_mullo_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, mul, src.as_i16x16()))
    }
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_mullo_epi16&expand=3992)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub fn _mm256_maskz_mullo_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let mul = _mm256_mullo_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, mul, i16x16::ZERO))
    }
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_mullo_epi16&expand=3988)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub fn _mm_mask_mullo_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let mul = _mm_mullo_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, mul, src.as_i16x8()))
    }
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_mullo_epi16&expand=3989)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub fn _mm_maskz_mullo_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let mul = _mm_mullo_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, mul, i16x8::ZERO))
    }
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_max_epu16&expand=3609)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub fn _mm512_max_epu16(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = a.as_u16x32();
        let b = b.as_u16x32();
        transmute(simd_select::<i16x32, _>(simd_gt(a, b), a, b))
    }
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_max_epu16&expand=3607)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub fn _mm512_mask_max_epu16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let max = _mm512_max_epu16(a, b).as_u16x32();
        transmute(simd_select_bitmask(k, max, src.as_u16x32()))
    }
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_max_epu16&expand=3608)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub fn _mm512_maskz_max_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let max = _mm512_max_epu16(a, b).as_u16x32();
        transmute(simd_select_bitmask(k, max, u16x32::ZERO))
    }
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_max_epu16&expand=3604)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub fn _mm256_mask_max_epu16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let max = _mm256_max_epu16(a, b).as_u16x16();
        transmute(simd_select_bitmask(k, max, src.as_u16x16()))
    }
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_max_epu16&expand=3605)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub fn _mm256_maskz_max_epu16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let max = _mm256_max_epu16(a, b).as_u16x16();
        transmute(simd_select_bitmask(k, max, u16x16::ZERO))
    }
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_max_epu16&expand=3601)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub fn _mm_mask_max_epu16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let max = _mm_max_epu16(a, b).as_u16x8();
        transmute(simd_select_bitmask(k, max, src.as_u16x8()))
    }
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_max_epu16&expand=3602)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub fn _mm_maskz_max_epu16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let max = _mm_max_epu16(a, b).as_u16x8();
        transmute(simd_select_bitmask(k, max, u16x8::ZERO))
    }
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_max_epu8&expand=3636)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub fn _mm512_max_epu8(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = a.as_u8x64();
        let b = b.as_u8x64();
        transmute(simd_select::<i8x64, _>(simd_gt(a, b), a, b))
    }
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_max_epu8&expand=3634)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub fn _mm512_mask_max_epu8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let max = _mm512_max_epu8(a, b).as_u8x64();
        transmute(simd_select_bitmask(k, max, src.as_u8x64()))
    }
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_max_epu8&expand=3635)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub fn _mm512_maskz_max_epu8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let max = _mm512_max_epu8(a, b).as_u8x64();
        transmute(simd_select_bitmask(k, max, u8x64::ZERO))
    }
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_max_epu8&expand=3631)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub fn _mm256_mask_max_epu8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let max = _mm256_max_epu8(a, b).as_u8x32();
        transmute(simd_select_bitmask(k, max, src.as_u8x32()))
    }
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_max_epu8&expand=3632)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub fn _mm256_maskz_max_epu8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let max = _mm256_max_epu8(a, b).as_u8x32();
        transmute(simd_select_bitmask(k, max, u8x32::ZERO))
    }
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_max_epu8&expand=3628)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub fn _mm_mask_max_epu8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let max = _mm_max_epu8(a, b).as_u8x16();
        transmute(simd_select_bitmask(k, max, src.as_u8x16()))
    }
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_max_epu8&expand=3629)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub fn _mm_maskz_max_epu8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let max = _mm_max_epu8(a, b).as_u8x16();
        transmute(simd_select_bitmask(k, max, u8x16::ZERO))
    }
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_max_epi16&expand=3573)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub fn _mm512_max_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = a.as_i16x32();
        let b = b.as_i16x32();
        transmute(simd_select::<i16x32, _>(simd_gt(a, b), a, b))
    }
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_max_epi16&expand=3571)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub fn _mm512_mask_max_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let max = _mm512_max_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, max, src.as_i16x32()))
    }
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_max_epi16&expand=3572)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub fn _mm512_maskz_max_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let max = _mm512_max_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, max, i16x32::ZERO))
    }
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_max_epi16&expand=3568)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub fn _mm256_mask_max_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let max = _mm256_max_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, max, src.as_i16x16()))
    }
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_max_epi16&expand=3569)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub fn _mm256_maskz_max_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let max = _mm256_max_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, max, i16x16::ZERO))
    }
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_max_epi16&expand=3565)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub fn _mm_mask_max_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let max = _mm_max_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, max, src.as_i16x8()))
    }
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_max_epi16&expand=3566)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub fn _mm_maskz_max_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let max = _mm_max_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, max, i16x8::ZERO))
    }
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_max_epi8&expand=3600)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub fn _mm512_max_epi8(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = a.as_i8x64();
        let b = b.as_i8x64();
        transmute(simd_select::<i8x64, _>(simd_gt(a, b), a, b))
    }
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_max_epi8&expand=3598)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub fn _mm512_mask_max_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let max = _mm512_max_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, max, src.as_i8x64()))
    }
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_max_epi8&expand=3599)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub fn _mm512_maskz_max_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let max = _mm512_max_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, max, i8x64::ZERO))
    }
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_max_epi8&expand=3595)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub fn _mm256_mask_max_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let max = _mm256_max_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, max, src.as_i8x32()))
    }
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_max_epi8&expand=3596)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub fn _mm256_maskz_max_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let max = _mm256_max_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, max, i8x32::ZERO))
    }
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_max_epi8&expand=3592)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub fn _mm_mask_max_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let max = _mm_max_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, max, src.as_i8x16()))
    }
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_max_epi8&expand=3593)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub fn _mm_maskz_max_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let max = _mm_max_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, max, i8x16::ZERO))
    }
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_min_epu16&expand=3723)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminuw))]
pub fn _mm512_min_epu16(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = a.as_u16x32();
        let b = b.as_u16x32();
        transmute(simd_select::<i16x32, _>(simd_lt(a, b), a, b))
    }
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_min_epu16&expand=3721)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminuw))]
pub fn _mm512_mask_min_epu16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let min = _mm512_min_epu16(a, b).as_u16x32();
        transmute(simd_select_bitmask(k, min, src.as_u16x32()))
    }
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_min_epu16&expand=3722)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminuw))]
pub fn _mm512_maskz_min_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let min = _mm512_min_epu16(a, b).as_u16x32();
        transmute(simd_select_bitmask(k, min, u16x32::ZERO))
    }
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_min_epu16&expand=3718)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminuw))]
pub fn _mm256_mask_min_epu16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let min = _mm256_min_epu16(a, b).as_u16x16();
        transmute(simd_select_bitmask(k, min, src.as_u16x16()))
    }
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_min_epu16&expand=3719)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminuw))]
pub fn _mm256_maskz_min_epu16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let min = _mm256_min_epu16(a, b).as_u16x16();
        transmute(simd_select_bitmask(k, min, u16x16::ZERO))
    }
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_min_epu16&expand=3715)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminuw))]
pub fn _mm_mask_min_epu16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let min = _mm_min_epu16(a, b).as_u16x8();
        transmute(simd_select_bitmask(k, min, src.as_u16x8()))
    }
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_min_epu16&expand=3716)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminuw))]
pub fn _mm_maskz_min_epu16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let min = _mm_min_epu16(a, b).as_u16x8();
        transmute(simd_select_bitmask(k, min, u16x8::ZERO))
    }
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_min_epu8&expand=3750)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminub))]
pub fn _mm512_min_epu8(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = a.as_u8x64();
        let b = b.as_u8x64();
        transmute(simd_select::<i8x64, _>(simd_lt(a, b), a, b))
    }
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_min_epu8&expand=3748)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminub))]
pub fn _mm512_mask_min_epu8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let min = _mm512_min_epu8(a, b).as_u8x64();
        transmute(simd_select_bitmask(k, min, src.as_u8x64()))
    }
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_min_epu8&expand=3749)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminub))]
pub fn _mm512_maskz_min_epu8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let min = _mm512_min_epu8(a, b).as_u8x64();
        transmute(simd_select_bitmask(k, min, u8x64::ZERO))
    }
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_min_epu8&expand=3745)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminub))]
pub fn _mm256_mask_min_epu8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let min = _mm256_min_epu8(a, b).as_u8x32();
        transmute(simd_select_bitmask(k, min, src.as_u8x32()))
    }
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_min_epu8&expand=3746)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminub))]
pub fn _mm256_maskz_min_epu8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let min = _mm256_min_epu8(a, b).as_u8x32();
        transmute(simd_select_bitmask(k, min, u8x32::ZERO))
    }
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_min_epu8&expand=3742)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminub))]
pub fn _mm_mask_min_epu8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let min = _mm_min_epu8(a, b).as_u8x16();
        transmute(simd_select_bitmask(k, min, src.as_u8x16()))
    }
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_min_epu8&expand=3743)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminub))]
pub fn _mm_maskz_min_epu8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let min = _mm_min_epu8(a, b).as_u8x16();
        transmute(simd_select_bitmask(k, min, u8x16::ZERO))
    }
}

/// Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_min_epi16&expand=3687)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminsw))]
pub fn _mm512_min_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = a.as_i16x32();
        let b = b.as_i16x32();
        transmute(simd_select::<i16x32, _>(simd_lt(a, b), a, b))
    }
}

/// Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_min_epi16&expand=3685)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminsw))]
pub fn _mm512_mask_min_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let min = _mm512_min_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, min, src.as_i16x32()))
    }
}

/// Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_min_epi16&expand=3686)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminsw))]
pub fn _mm512_maskz_min_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let min = _mm512_min_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, min, i16x32::ZERO))
    }
}

/// Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_min_epi16&expand=3682)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminsw))]
pub fn _mm256_mask_min_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let min = _mm256_min_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, min, src.as_i16x16()))
    }
}

/// Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_min_epi16&expand=3683)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminsw))]
pub fn _mm256_maskz_min_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let min = _mm256_min_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, min, i16x16::ZERO))
    }
}

/// Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_min_epi16&expand=3679)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminsw))]
pub fn _mm_mask_min_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let min = _mm_min_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, min, src.as_i16x8()))
    }
}

/// Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_min_epi16&expand=3680)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminsw))]
pub fn _mm_maskz_min_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let min = _mm_min_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, min, i16x8::ZERO))
    }
}

/// Compare packed signed 8-bit integers in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_min_epi8&expand=3714)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminsb))]
pub fn _mm512_min_epi8(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = a.as_i8x64();
        let b = b.as_i8x64();
        transmute(simd_select::<i8x64, _>(simd_lt(a, b), a, b))
    }
}

/// Compare packed signed 8-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_min_epi8&expand=3712)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminsb))]
pub fn _mm512_mask_min_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let min = _mm512_min_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, min, src.as_i8x64()))
    }
}

/// Compare packed signed 8-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_min_epi8&expand=3713)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminsb))]
pub fn _mm512_maskz_min_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let min = _mm512_min_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, min, i8x64::ZERO))
    }
}

/// Compare packed signed 8-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_min_epi8&expand=3709)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminsb))]
pub fn _mm256_mask_min_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let min = _mm256_min_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, min, src.as_i8x32()))
    }
}

/// Compare packed signed 8-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_min_epi8&expand=3710)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminsb))]
pub fn _mm256_maskz_min_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let min = _mm256_min_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, min, i8x32::ZERO))
    }
}

/// Compare packed signed 8-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_min_epi8&expand=3706)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminsb))]
pub fn _mm_mask_min_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let min = _mm_min_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, min, src.as_i8x16()))
    }
}

/// Compare packed signed 8-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_min_epi8&expand=3707)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpminsb))]
pub fn _mm_maskz_min_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let min = _mm_min_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, min, i8x16::ZERO))
    }
}

/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmplt_epu16_mask&expand=1050)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmplt_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    unsafe { simd_bitmask::<u16x32, _>(simd_lt(a.as_u16x32(), b.as_u16x32())) }
}

/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmplt_epu16_mask&expand=1051)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmplt_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_mask_cmp_epu16_mask::<_MM_CMPINT_LT>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmplt_epu16_mask&expand=1050)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmplt_epu16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    unsafe { simd_bitmask::<u16x16, _>(simd_lt(a.as_u16x16(), b.as_u16x16())) }
}

/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmplt_epu16_mask&expand=1049)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmplt_epu16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_mask_cmp_epu16_mask::<_MM_CMPINT_LT>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmplt_epu16_mask&expand=1018)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmplt_epu16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    unsafe { simd_bitmask::<u16x8, _>(simd_lt(a.as_u16x8(), b.as_u16x8())) }
}

/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmplt_epu16_mask&expand=1019)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmplt_epu16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_mask_cmp_epu16_mask::<_MM_CMPINT_LT>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=mm512_cmplt_epu8_mask&expand=1068)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmplt_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    unsafe { simd_bitmask::<u8x64, _>(simd_lt(a.as_u8x64(), b.as_u8x64())) }
}

/// Compare packed unsigned 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmplt_epu8_mask&expand=1069)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmplt_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_mask_cmp_epu8_mask::<_MM_CMPINT_LT>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmplt_epu8_mask&expand=1066)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmplt_epu8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    unsafe { simd_bitmask::<u8x32, _>(simd_lt(a.as_u8x32(), b.as_u8x32())) }
}

/// Compare packed unsigned 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmplt_epu8_mask&expand=1067)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmplt_epu8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_mask_cmp_epu8_mask::<_MM_CMPINT_LT>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmplt_epu8_mask&expand=1064)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmplt_epu8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    unsafe { simd_bitmask::<u8x16, _>(simd_lt(a.as_u8x16(), b.as_u8x16())) }
}

/// Compare packed unsigned 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmplt_epu8_mask&expand=1065)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmplt_epu8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_mask_cmp_epu8_mask::<_MM_CMPINT_LT>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmplt_epi16_mask&expand=1022)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmplt_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    unsafe { simd_bitmask::<i16x32, _>(simd_lt(a.as_i16x32(), b.as_i16x32())) }
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmplt_epi16_mask&expand=1023)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmplt_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_mask_cmp_epi16_mask::<_MM_CMPINT_LT>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmplt_epi16_mask&expand=1020)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmplt_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    unsafe { simd_bitmask::<i16x16, _>(simd_lt(a.as_i16x16(), b.as_i16x16())) }
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmplt_epi16_mask&expand=1021)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmplt_epi16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_mask_cmp_epi16_mask::<_MM_CMPINT_LT>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmplt_epi16_mask&expand=1018)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmplt_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    unsafe { simd_bitmask::<i16x8, _>(simd_lt(a.as_i16x8(), b.as_i16x8())) }
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmplt_epi16_mask&expand=1019)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmplt_epi16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_mask_cmp_epi16_mask::<_MM_CMPINT_LT>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmplt_epi8_mask&expand=1044)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmplt_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    unsafe { simd_bitmask::<i8x64, _>(simd_lt(a.as_i8x64(), b.as_i8x64())) }
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmplt_epi8_mask&expand=1045)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmplt_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_mask_cmp_epi8_mask::<_MM_CMPINT_LT>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmplt_epi8_mask&expand=1042)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmplt_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    unsafe { simd_bitmask::<i8x32, _>(simd_lt(a.as_i8x32(), b.as_i8x32())) }
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmplt_epi8_mask&expand=1043)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmplt_epi8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_mask_cmp_epi8_mask::<_MM_CMPINT_LT>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmplt_epi8_mask&expand=1040)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmplt_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    unsafe { simd_bitmask::<i8x16, _>(simd_lt(a.as_i8x16(), b.as_i8x16())) }
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmplt_epi8_mask&expand=1041)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmplt_epi8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_mask_cmp_epi8_mask::<_MM_CMPINT_LT>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpgt_epu16_mask&expand=927)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpgt_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    unsafe { simd_bitmask::<u16x32, _>(simd_gt(a.as_u16x32(), b.as_u16x32())) }
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpgt_epu16_mask&expand=928)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpgt_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_mask_cmp_epu16_mask::<_MM_CMPINT_NLE>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpgt_epu16_mask&expand=925)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpgt_epu16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    unsafe { simd_bitmask::<u16x16, _>(simd_gt(a.as_u16x16(), b.as_u16x16())) }
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpgt_epu16_mask&expand=926)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpgt_epu16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_mask_cmp_epu16_mask::<_MM_CMPINT_NLE>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpgt_epu16_mask&expand=923)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpgt_epu16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    unsafe { simd_bitmask::<u16x8, _>(simd_gt(a.as_u16x8(), b.as_u16x8())) }
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpgt_epu16_mask&expand=924)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpgt_epu16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_mask_cmp_epu16_mask::<_MM_CMPINT_NLE>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpgt_epu8_mask&expand=945)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpgt_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    unsafe { simd_bitmask::<u8x64, _>(simd_gt(a.as_u8x64(), b.as_u8x64())) }
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpgt_epu8_mask&expand=946)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpgt_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_mask_cmp_epu8_mask::<_MM_CMPINT_NLE>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpgt_epu8_mask&expand=943)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpgt_epu8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    unsafe { simd_bitmask::<u8x32, _>(simd_gt(a.as_u8x32(), b.as_u8x32())) }
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpgt_epu8_mask&expand=944)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpgt_epu8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_mask_cmp_epu8_mask::<_MM_CMPINT_NLE>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpgt_epu8_mask&expand=941)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpgt_epu8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    unsafe { simd_bitmask::<u8x16, _>(simd_gt(a.as_u8x16(), b.as_u8x16())) }
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpgt_epu8_mask&expand=942)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpgt_epu8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_mask_cmp_epu8_mask::<_MM_CMPINT_NLE>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpgt_epi16_mask&expand=897)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpgt_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    unsafe { simd_bitmask::<i16x32, _>(simd_gt(a.as_i16x32(), b.as_i16x32())) }
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpgt_epi16_mask&expand=898)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpgt_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_mask_cmp_epi16_mask::<_MM_CMPINT_NLE>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpgt_epi16_mask&expand=895)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpgt_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    unsafe { simd_bitmask::<i16x16, _>(simd_gt(a.as_i16x16(), b.as_i16x16())) }
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpgt_epi16_mask&expand=896)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpgt_epi16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_mask_cmp_epi16_mask::<_MM_CMPINT_NLE>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpgt_epi16_mask&expand=893)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpgt_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    unsafe { simd_bitmask::<i16x8, _>(simd_gt(a.as_i16x8(), b.as_i16x8())) }
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpgt_epi16_mask&expand=894)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpgt_epi16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_mask_cmp_epi16_mask::<_MM_CMPINT_NLE>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpgt_epi8_mask&expand=921)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpgt_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    unsafe { simd_bitmask::<i8x64, _>(simd_gt(a.as_i8x64(), b.as_i8x64())) }
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpgt_epi8_mask&expand=922)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpgt_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_mask_cmp_epi8_mask::<_MM_CMPINT_NLE>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpgt_epi8_mask&expand=919)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpgt_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    unsafe { simd_bitmask::<i8x32, _>(simd_gt(a.as_i8x32(), b.as_i8x32())) }
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpgt_epi8_mask&expand=920)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpgt_epi8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_mask_cmp_epi8_mask::<_MM_CMPINT_NLE>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpgt_epi8_mask&expand=917)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpgt_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    unsafe { simd_bitmask::<i8x16, _>(simd_gt(a.as_i8x16(), b.as_i8x16())) }
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpgt_epi8_mask&expand=918)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpgt_epi8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_mask_cmp_epi8_mask::<_MM_CMPINT_NLE>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmple_epu16_mask&expand=989)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmple_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    unsafe { simd_bitmask::<u16x32, _>(simd_le(a.as_u16x32(), b.as_u16x32())) }
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmple_epu16_mask&expand=990)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmple_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_mask_cmp_epu16_mask::<_MM_CMPINT_LE>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmple_epu16_mask&expand=987)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmple_epu16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    unsafe { simd_bitmask::<u16x16, _>(simd_le(a.as_u16x16(), b.as_u16x16())) }
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmple_epu16_mask&expand=988)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmple_epu16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_mask_cmp_epu16_mask::<_MM_CMPINT_LE>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmple_epu16_mask&expand=985)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmple_epu16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    unsafe { simd_bitmask::<u16x8, _>(simd_le(a.as_u16x8(), b.as_u16x8())) }
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmple_epu16_mask&expand=986)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmple_epu16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_mask_cmp_epu16_mask::<_MM_CMPINT_LE>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.   
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmple_epu8_mask&expand=1007)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmple_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    unsafe { simd_bitmask::<u8x64, _>(simd_le(a.as_u8x64(), b.as_u8x64())) }
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmple_epu8_mask&expand=1008)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmple_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_mask_cmp_epu8_mask::<_MM_CMPINT_LE>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.   
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmple_epu8_mask&expand=1005)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmple_epu8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    unsafe { simd_bitmask::<u8x32, _>(simd_le(a.as_u8x32(), b.as_u8x32())) }
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmple_epu8_mask&expand=1006)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmple_epu8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_mask_cmp_epu8_mask::<_MM_CMPINT_LE>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.   
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmple_epu8_mask&expand=1003)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmple_epu8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    unsafe { simd_bitmask::<u8x16, _>(simd_le(a.as_u8x16(), b.as_u8x16())) }
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmple_epu8_mask&expand=1004)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmple_epu8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_mask_cmp_epu8_mask::<_MM_CMPINT_LE>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmple_epi16_mask&expand=965)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmple_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    unsafe { simd_bitmask::<i16x32, _>(simd_le(a.as_i16x32(), b.as_i16x32())) }
}

/// Compare packed signed 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmple_epi16_mask&expand=966)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmple_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_mask_cmp_epi16_mask::<_MM_CMPINT_LE>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmple_epi16_mask&expand=963)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmple_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    unsafe { simd_bitmask::<i16x16, _>(simd_le(a.as_i16x16(), b.as_i16x16())) }
}

/// Compare packed signed 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmple_epi16_mask&expand=964)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmple_epi16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_mask_cmp_epi16_mask::<_MM_CMPINT_LE>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmple_epi16_mask&expand=961)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmple_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    unsafe { simd_bitmask::<i16x8, _>(simd_le(a.as_i16x8(), b.as_i16x8())) }
}

/// Compare packed signed 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmple_epi16_mask&expand=962)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmple_epi16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_mask_cmp_epi16_mask::<_MM_CMPINT_LE>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmple_epi8_mask&expand=983)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmple_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    unsafe { simd_bitmask::<i8x64, _>(simd_le(a.as_i8x64(), b.as_i8x64())) }
}

/// Compare packed signed 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmple_epi8_mask&expand=984)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmple_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_mask_cmp_epi8_mask::<_MM_CMPINT_LE>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmple_epi8_mask&expand=981)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmple_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    unsafe { simd_bitmask::<i8x32, _>(simd_le(a.as_i8x32(), b.as_i8x32())) }
}

/// Compare packed signed 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmple_epi8_mask&expand=982)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmple_epi8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_mask_cmp_epi8_mask::<_MM_CMPINT_LE>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmple_epi8_mask&expand=979)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmple_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    unsafe { simd_bitmask::<i8x16, _>(simd_le(a.as_i8x16(), b.as_i8x16())) }
}

/// Compare packed signed 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmple_epi8_mask&expand=980)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmple_epi8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_mask_cmp_epi8_mask::<_MM_CMPINT_LE>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpge_epu16_mask&expand=867)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpge_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    unsafe { simd_bitmask::<u16x32, _>(simd_ge(a.as_u16x32(), b.as_u16x32())) }
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpge_epu16_mask&expand=868)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpge_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_mask_cmp_epu16_mask::<_MM_CMPINT_NLT>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpge_epu16_mask&expand=865)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpge_epu16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    unsafe { simd_bitmask::<u16x16, _>(simd_ge(a.as_u16x16(), b.as_u16x16())) }
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpge_epu16_mask&expand=866)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpge_epu16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_mask_cmp_epu16_mask::<_MM_CMPINT_NLT>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpge_epu16_mask&expand=863)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpge_epu16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    unsafe { simd_bitmask::<u16x8, _>(simd_ge(a.as_u16x8(), b.as_u16x8())) }
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpge_epu16_mask&expand=864)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpge_epu16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_mask_cmp_epu16_mask::<_MM_CMPINT_NLT>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpge_epu8_mask&expand=885)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpge_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    unsafe { simd_bitmask::<u8x64, _>(simd_ge(a.as_u8x64(), b.as_u8x64())) }
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpge_epu8_mask&expand=886)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpge_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_mask_cmp_epu8_mask::<_MM_CMPINT_NLT>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpge_epu8_mask&expand=883)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpge_epu8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    unsafe { simd_bitmask::<u8x32, _>(simd_ge(a.as_u8x32(), b.as_u8x32())) }
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpge_epu8_mask&expand=884)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpge_epu8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_mask_cmp_epu8_mask::<_MM_CMPINT_NLT>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpge_epu8_mask&expand=881)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpge_epu8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    unsafe { simd_bitmask::<u8x16, _>(simd_ge(a.as_u8x16(), b.as_u8x16())) }
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpge_epu8_mask&expand=882)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpge_epu8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_mask_cmp_epu8_mask::<_MM_CMPINT_NLT>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpge_epi16_mask&expand=843)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpge_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    unsafe { simd_bitmask::<i16x32, _>(simd_ge(a.as_i16x32(), b.as_i16x32())) }
}

/// Compare packed signed 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpge_epi16_mask&expand=844)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpge_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_mask_cmp_epi16_mask::<_MM_CMPINT_NLT>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpge_epi16_mask&expand=841)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpge_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    unsafe { simd_bitmask::<i16x16, _>(simd_ge(a.as_i16x16(), b.as_i16x16())) }
}

/// Compare packed signed 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpge_epi16_mask&expand=842)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpge_epi16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_mask_cmp_epi16_mask::<_MM_CMPINT_NLT>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpge_epi16_mask&expand=839)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpge_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    unsafe { simd_bitmask::<i16x8, _>(simd_ge(a.as_i16x8(), b.as_i16x8())) }
}

/// Compare packed signed 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpge_epi16_mask&expand=840)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpge_epi16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_mask_cmp_epi16_mask::<_MM_CMPINT_NLT>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpge_epi8_mask&expand=861)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpge_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    unsafe { simd_bitmask::<i8x64, _>(simd_ge(a.as_i8x64(), b.as_i8x64())) }
}

/// Compare packed signed 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpge_epi8_mask&expand=862)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpge_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_mask_cmp_epi8_mask::<_MM_CMPINT_NLT>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpge_epi8_mask&expand=859)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpge_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    unsafe { simd_bitmask::<i8x32, _>(simd_ge(a.as_i8x32(), b.as_i8x32())) }
}

/// Compare packed signed 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpge_epi8_mask&expand=860)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpge_epi8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_mask_cmp_epi8_mask::<_MM_CMPINT_NLT>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpge_epi8_mask&expand=857)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpge_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    unsafe { simd_bitmask::<i8x16, _>(simd_ge(a.as_i8x16(), b.as_i8x16())) }
}

/// Compare packed signed 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpge_epi8_mask&expand=858)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpge_epi8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_mask_cmp_epi8_mask::<_MM_CMPINT_NLT>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpeq_epu16_mask&expand=801)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpeq_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    unsafe { simd_bitmask::<u16x32, _>(simd_eq(a.as_u16x32(), b.as_u16x32())) }
}

/// Compare packed unsigned 16-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpeq_epu16_mask&expand=802)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpeq_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_mask_cmp_epu16_mask::<_MM_CMPINT_EQ>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpeq_epu16_mask&expand=799)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpeq_epu16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    unsafe { simd_bitmask::<u16x16, _>(simd_eq(a.as_u16x16(), b.as_u16x16())) }
}

/// Compare packed unsigned 16-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpeq_epu16_mask&expand=800)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpeq_epu16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_mask_cmp_epu16_mask::<_MM_CMPINT_EQ>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpeq_epu16_mask&expand=797)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpeq_epu16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    unsafe { simd_bitmask::<u16x8, _>(simd_eq(a.as_u16x8(), b.as_u16x8())) }
}

/// Compare packed unsigned 16-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpeq_epu16_mask&expand=798)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpeq_epu16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_mask_cmp_epu16_mask::<_MM_CMPINT_EQ>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpeq_epu8_mask&expand=819)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpeq_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    unsafe { simd_bitmask::<u8x64, _>(simd_eq(a.as_u8x64(), b.as_u8x64())) }
}

/// Compare packed unsigned 8-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpeq_epu8_mask&expand=820)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpeq_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_mask_cmp_epu8_mask::<_MM_CMPINT_EQ>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpeq_epu8_mask&expand=817)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpeq_epu8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    unsafe { simd_bitmask::<u8x32, _>(simd_eq(a.as_u8x32(), b.as_u8x32())) }
}

/// Compare packed unsigned 8-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpeq_epu8_mask&expand=818)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpeq_epu8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_mask_cmp_epu8_mask::<_MM_CMPINT_EQ>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpeq_epu8_mask&expand=815)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpeq_epu8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    unsafe { simd_bitmask::<u8x16, _>(simd_eq(a.as_u8x16(), b.as_u8x16())) }
}

/// Compare packed unsigned 8-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpeq_epu8_mask&expand=816)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpeq_epu8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_mask_cmp_epu8_mask::<_MM_CMPINT_EQ>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpeq_epi16_mask&expand=771)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpeq_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    unsafe { simd_bitmask::<i16x32, _>(simd_eq(a.as_i16x32(), b.as_i16x32())) }
}

/// Compare packed signed 16-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpeq_epi16_mask&expand=772)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpeq_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_mask_cmp_epi16_mask::<_MM_CMPINT_EQ>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpeq_epi16_mask&expand=769)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpeq_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    unsafe { simd_bitmask::<i16x16, _>(simd_eq(a.as_i16x16(), b.as_i16x16())) }
}

/// Compare packed signed 16-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpeq_epi16_mask&expand=770)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpeq_epi16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_mask_cmp_epi16_mask::<_MM_CMPINT_EQ>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpeq_epi16_mask&expand=767)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpeq_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    unsafe { simd_bitmask::<i16x8, _>(simd_eq(a.as_i16x8(), b.as_i16x8())) }
}

/// Compare packed signed 16-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpeq_epi16_mask&expand=768)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpeq_epi16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_mask_cmp_epi16_mask::<_MM_CMPINT_EQ>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpeq_epi8_mask&expand=795)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpeq_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    unsafe { simd_bitmask::<i8x64, _>(simd_eq(a.as_i8x64(), b.as_i8x64())) }
}

/// Compare packed signed 8-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpeq_epi8_mask&expand=796)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpeq_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_mask_cmp_epi8_mask::<_MM_CMPINT_EQ>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpeq_epi8_mask&expand=793)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpeq_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    unsafe { simd_bitmask::<i8x32, _>(simd_eq(a.as_i8x32(), b.as_i8x32())) }
}

/// Compare packed signed 8-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpeq_epi8_mask&expand=794)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpeq_epi8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_mask_cmp_epi8_mask::<_MM_CMPINT_EQ>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpeq_epi8_mask&expand=791)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpeq_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    unsafe { simd_bitmask::<i8x16, _>(simd_eq(a.as_i8x16(), b.as_i8x16())) }
}

/// Compare packed signed 8-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpeq_epi8_mask&expand=792)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpeq_epi8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_mask_cmp_epi8_mask::<_MM_CMPINT_EQ>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpneq_epu16_mask&expand=1106)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpneq_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    unsafe { simd_bitmask::<u16x32, _>(simd_ne(a.as_u16x32(), b.as_u16x32())) }
}

/// Compare packed unsigned 16-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpneq_epu16_mask&expand=1107)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpneq_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_mask_cmp_epu16_mask::<_MM_CMPINT_NE>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpneq_epu16_mask&expand=1104)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpneq_epu16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    unsafe { simd_bitmask::<u16x16, _>(simd_ne(a.as_u16x16(), b.as_u16x16())) }
}

/// Compare packed unsigned 16-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpneq_epu16_mask&expand=1105)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpneq_epu16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_mask_cmp_epu16_mask::<_MM_CMPINT_NE>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpneq_epu16_mask&expand=1102)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpneq_epu16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    unsafe { simd_bitmask::<u16x8, _>(simd_ne(a.as_u16x8(), b.as_u16x8())) }
}

/// Compare packed unsigned 16-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpneq_epu16_mask&expand=1103)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpneq_epu16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_mask_cmp_epu16_mask::<_MM_CMPINT_NE>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpneq_epu8_mask&expand=1124)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpneq_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    unsafe { simd_bitmask::<u8x64, _>(simd_ne(a.as_u8x64(), b.as_u8x64())) }
}

/// Compare packed unsigned 8-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpneq_epu8_mask&expand=1125)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpneq_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_mask_cmp_epu8_mask::<_MM_CMPINT_NE>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpneq_epu8_mask&expand=1122)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpneq_epu8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    unsafe { simd_bitmask::<u8x32, _>(simd_ne(a.as_u8x32(), b.as_u8x32())) }
}

/// Compare packed unsigned 8-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpneq_epu8_mask&expand=1123)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpneq_epu8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_mask_cmp_epu8_mask::<_MM_CMPINT_NE>(k1, a, b)
}

/// Compare packed unsigned 8-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpneq_epu8_mask&expand=1120)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpneq_epu8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    unsafe { simd_bitmask::<u8x16, _>(simd_ne(a.as_u8x16(), b.as_u8x16())) }
}

/// Compare packed unsigned 8-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpneq_epu8_mask&expand=1121)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpneq_epu8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_mask_cmp_epu8_mask::<_MM_CMPINT_NE>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpneq_epi16_mask&expand=1082)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpneq_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    unsafe { simd_bitmask::<i16x32, _>(simd_ne(a.as_i16x32(), b.as_i16x32())) }
}

/// Compare packed signed 16-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpneq_epi16_mask&expand=1083)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpneq_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_mask_cmp_epi16_mask::<_MM_CMPINT_NE>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpneq_epi16_mask&expand=1080)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpneq_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    unsafe { simd_bitmask::<i16x16, _>(simd_ne(a.as_i16x16(), b.as_i16x16())) }
}

/// Compare packed signed 16-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpneq_epi16_mask&expand=1081)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpneq_epi16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_mask_cmp_epi16_mask::<_MM_CMPINT_NE>(k1, a, b)
}

/// Compare packed signed 16-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpneq_epi16_mask&expand=1078)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpneq_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    unsafe { simd_bitmask::<i16x8, _>(simd_ne(a.as_i16x8(), b.as_i16x8())) }
}

/// Compare packed signed 16-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpneq_epi16_mask&expand=1079)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpneq_epi16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_mask_cmp_epi16_mask::<_MM_CMPINT_NE>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmpneq_epi8_mask&expand=1100)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_cmpneq_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    unsafe { simd_bitmask::<i8x64, _>(simd_ne(a.as_i8x64(), b.as_i8x64())) }
}

/// Compare packed signed 8-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmpneq_epi8_mask&expand=1101)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm512_mask_cmpneq_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_mask_cmp_epi8_mask::<_MM_CMPINT_NE>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpneq_epi8_mask&expand=1098)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_cmpneq_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    unsafe { simd_bitmask::<i8x32, _>(simd_ne(a.as_i8x32(), b.as_i8x32())) }
}

/// Compare packed signed 8-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmpneq_epi8_mask&expand=1099)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm256_mask_cmpneq_epi8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_mask_cmp_epi8_mask::<_MM_CMPINT_NE>(k1, a, b)
}

/// Compare packed signed 8-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpneq_epi8_mask&expand=1096)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_cmpneq_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    unsafe { simd_bitmask::<i8x16, _>(simd_ne(a.as_i8x16(), b.as_i8x16())) }
}

/// Compare packed signed 8-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmpneq_epi8_mask&expand=1097)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub fn _mm_mask_cmpneq_epi8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_mask_cmp_epi8_mask::<_MM_CMPINT_NE>(k1, a, b)
}

/// Compare packed unsigned 16-bit integers in a and b based on the comparison operand specified by `IMM8`, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmp_epu16_mask&expand=715)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm512_cmp_epu16_mask<const IMM8: i32>(a: __m512i, b: __m512i) -> __mmask32 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_u16x32();
        let b = b.as_u16x32();
        let r = match IMM8 {
            0 => simd_eq(a, b),
            1 => simd_lt(a, b),
            2 => simd_le(a, b),
            3 => i16x32::ZERO,
            4 => simd_ne(a, b),
            5 => simd_ge(a, b),
            6 => simd_gt(a, b),
            _ => i16x32::splat(-1),
        };
        simd_bitmask(r)
    }
}

/// Compare packed unsigned 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmp_epu16_mask&expand=716)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm512_mask_cmp_epu16_mask<const IMM8: i32>(
    k1: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __mmask32 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_u16x32();
        let b = b.as_u16x32();
        let k1 = simd_select_bitmask(k1, i16x32::splat(-1), i16x32::ZERO);
        let r = match IMM8 {
            0 => simd_and(k1, simd_eq(a, b)),
            1 => simd_and(k1, simd_lt(a, b)),
            2 => simd_and(k1, simd_le(a, b)),
            3 => i16x32::ZERO,
            4 => simd_and(k1, simd_ne(a, b)),
            5 => simd_and(k1, simd_ge(a, b)),
            6 => simd_and(k1, simd_gt(a, b)),
            _ => k1,
        };
        simd_bitmask(r)
    }
}

/// Compare packed unsigned 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmp_epu16_mask&expand=713)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm256_cmp_epu16_mask<const IMM8: i32>(a: __m256i, b: __m256i) -> __mmask16 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_u16x16();
        let b = b.as_u16x16();
        let r = match IMM8 {
            0 => simd_eq(a, b),
            1 => simd_lt(a, b),
            2 => simd_le(a, b),
            3 => i16x16::ZERO,
            4 => simd_ne(a, b),
            5 => simd_ge(a, b),
            6 => simd_gt(a, b),
            _ => i16x16::splat(-1),
        };
        simd_bitmask(r)
    }
}

/// Compare packed unsigned 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmp_epu16_mask&expand=714)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm256_mask_cmp_epu16_mask<const IMM8: i32>(
    k1: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __mmask16 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_u16x16();
        let b = b.as_u16x16();
        let k1 = simd_select_bitmask(k1, i16x16::splat(-1), i16x16::ZERO);
        let r = match IMM8 {
            0 => simd_and(k1, simd_eq(a, b)),
            1 => simd_and(k1, simd_lt(a, b)),
            2 => simd_and(k1, simd_le(a, b)),
            3 => i16x16::ZERO,
            4 => simd_and(k1, simd_ne(a, b)),
            5 => simd_and(k1, simd_ge(a, b)),
            6 => simd_and(k1, simd_gt(a, b)),
            _ => k1,
        };
        simd_bitmask(r)
    }
}

/// Compare packed unsigned 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmp_epu16_mask&expand=711)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm_cmp_epu16_mask<const IMM8: i32>(a: __m128i, b: __m128i) -> __mmask8 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_u16x8();
        let b = b.as_u16x8();
        let r = match IMM8 {
            0 => simd_eq(a, b),
            1 => simd_lt(a, b),
            2 => simd_le(a, b),
            3 => i16x8::ZERO,
            4 => simd_ne(a, b),
            5 => simd_ge(a, b),
            6 => simd_gt(a, b),
            _ => i16x8::splat(-1),
        };
        simd_bitmask(r)
    }
}

/// Compare packed unsigned 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmp_epu16_mask&expand=712)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm_mask_cmp_epu16_mask<const IMM8: i32>(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_u16x8();
        let b = b.as_u16x8();
        let k1 = simd_select_bitmask(k1, i16x8::splat(-1), i16x8::ZERO);
        let r = match IMM8 {
            0 => simd_and(k1, simd_eq(a, b)),
            1 => simd_and(k1, simd_lt(a, b)),
            2 => simd_and(k1, simd_le(a, b)),
            3 => i16x8::ZERO,
            4 => simd_and(k1, simd_ne(a, b)),
            5 => simd_and(k1, simd_ge(a, b)),
            6 => simd_and(k1, simd_gt(a, b)),
            _ => k1,
        };
        simd_bitmask(r)
    }
}

/// Compare packed unsigned 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmp_epu8_mask&expand=733)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm512_cmp_epu8_mask<const IMM8: i32>(a: __m512i, b: __m512i) -> __mmask64 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_u8x64();
        let b = b.as_u8x64();
        let r = match IMM8 {
            0 => simd_eq(a, b),
            1 => simd_lt(a, b),
            2 => simd_le(a, b),
            3 => i8x64::ZERO,
            4 => simd_ne(a, b),
            5 => simd_ge(a, b),
            6 => simd_gt(a, b),
            _ => i8x64::splat(-1),
        };
        simd_bitmask(r)
    }
}

/// Compare packed unsigned 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmp_epu8_mask&expand=734)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm512_mask_cmp_epu8_mask<const IMM8: i32>(
    k1: __mmask64,
    a: __m512i,
    b: __m512i,
) -> __mmask64 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_u8x64();
        let b = b.as_u8x64();
        let k1 = simd_select_bitmask(k1, i8x64::splat(-1), i8x64::ZERO);
        let r = match IMM8 {
            0 => simd_and(k1, simd_eq(a, b)),
            1 => simd_and(k1, simd_lt(a, b)),
            2 => simd_and(k1, simd_le(a, b)),
            3 => i8x64::ZERO,
            4 => simd_and(k1, simd_ne(a, b)),
            5 => simd_and(k1, simd_ge(a, b)),
            6 => simd_and(k1, simd_gt(a, b)),
            _ => k1,
        };
        simd_bitmask(r)
    }
}

/// Compare packed unsigned 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmp_epu8_mask&expand=731)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm256_cmp_epu8_mask<const IMM8: i32>(a: __m256i, b: __m256i) -> __mmask32 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_u8x32();
        let b = b.as_u8x32();
        let r = match IMM8 {
            0 => simd_eq(a, b),
            1 => simd_lt(a, b),
            2 => simd_le(a, b),
            3 => i8x32::ZERO,
            4 => simd_ne(a, b),
            5 => simd_ge(a, b),
            6 => simd_gt(a, b),
            _ => i8x32::splat(-1),
        };
        simd_bitmask(r)
    }
}

/// Compare packed unsigned 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmp_epu8_mask&expand=732)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm256_mask_cmp_epu8_mask<const IMM8: i32>(
    k1: __mmask32,
    a: __m256i,
    b: __m256i,
) -> __mmask32 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_u8x32();
        let b = b.as_u8x32();
        let k1 = simd_select_bitmask(k1, i8x32::splat(-1), i8x32::ZERO);
        let r = match IMM8 {
            0 => simd_and(k1, simd_eq(a, b)),
            1 => simd_and(k1, simd_lt(a, b)),
            2 => simd_and(k1, simd_le(a, b)),
            3 => i8x32::ZERO,
            4 => simd_and(k1, simd_ne(a, b)),
            5 => simd_and(k1, simd_ge(a, b)),
            6 => simd_and(k1, simd_gt(a, b)),
            _ => k1,
        };
        simd_bitmask(r)
    }
}

/// Compare packed unsigned 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmp_epu8_mask&expand=729)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm_cmp_epu8_mask<const IMM8: i32>(a: __m128i, b: __m128i) -> __mmask16 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_u8x16();
        let b = b.as_u8x16();
        let r = match IMM8 {
            0 => simd_eq(a, b),
            1 => simd_lt(a, b),
            2 => simd_le(a, b),
            3 => i8x16::ZERO,
            4 => simd_ne(a, b),
            5 => simd_ge(a, b),
            6 => simd_gt(a, b),
            _ => i8x16::splat(-1),
        };
        simd_bitmask(r)
    }
}

/// Compare packed unsigned 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmp_epu8_mask&expand=730)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm_mask_cmp_epu8_mask<const IMM8: i32>(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_u8x16();
        let b = b.as_u8x16();
        let k1 = simd_select_bitmask(k1, i8x16::splat(-1), i8x16::ZERO);
        let r = match IMM8 {
            0 => simd_and(k1, simd_eq(a, b)),
            1 => simd_and(k1, simd_lt(a, b)),
            2 => simd_and(k1, simd_le(a, b)),
            3 => i8x16::ZERO,
            4 => simd_and(k1, simd_ne(a, b)),
            5 => simd_and(k1, simd_ge(a, b)),
            6 => simd_and(k1, simd_gt(a, b)),
            _ => k1,
        };
        simd_bitmask(r)
    }
}

/// Compare packed signed 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmp_epi16_mask&expand=691)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm512_cmp_epi16_mask<const IMM8: i32>(a: __m512i, b: __m512i) -> __mmask32 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_i16x32();
        let b = b.as_i16x32();
        let r = match IMM8 {
            0 => simd_eq(a, b),
            1 => simd_lt(a, b),
            2 => simd_le(a, b),
            3 => i16x32::ZERO,
            4 => simd_ne(a, b),
            5 => simd_ge(a, b),
            6 => simd_gt(a, b),
            _ => i16x32::splat(-1),
        };
        simd_bitmask(r)
    }
}

/// Compare packed signed 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmp_epi16_mask&expand=692)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm512_mask_cmp_epi16_mask<const IMM8: i32>(
    k1: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __mmask32 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_i16x32();
        let b = b.as_i16x32();
        let k1 = simd_select_bitmask(k1, i16x32::splat(-1), i16x32::ZERO);
        let r = match IMM8 {
            0 => simd_and(k1, simd_eq(a, b)),
            1 => simd_and(k1, simd_lt(a, b)),
            2 => simd_and(k1, simd_le(a, b)),
            3 => i16x32::ZERO,
            4 => simd_and(k1, simd_ne(a, b)),
            5 => simd_and(k1, simd_ge(a, b)),
            6 => simd_and(k1, simd_gt(a, b)),
            _ => k1,
        };
        simd_bitmask(r)
    }
}

/// Compare packed signed 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmp_epi16_mask&expand=689)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm256_cmp_epi16_mask<const IMM8: i32>(a: __m256i, b: __m256i) -> __mmask16 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_i16x16();
        let b = b.as_i16x16();
        let r = match IMM8 {
            0 => simd_eq(a, b),
            1 => simd_lt(a, b),
            2 => simd_le(a, b),
            3 => i16x16::ZERO,
            4 => simd_ne(a, b),
            5 => simd_ge(a, b),
            6 => simd_gt(a, b),
            _ => i16x16::splat(-1),
        };
        simd_bitmask(r)
    }
}

/// Compare packed signed 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmp_epi16_mask&expand=690)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm256_mask_cmp_epi16_mask<const IMM8: i32>(
    k1: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __mmask16 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_i16x16();
        let b = b.as_i16x16();
        let k1 = simd_select_bitmask(k1, i16x16::splat(-1), i16x16::ZERO);
        let r = match IMM8 {
            0 => simd_and(k1, simd_eq(a, b)),
            1 => simd_and(k1, simd_lt(a, b)),
            2 => simd_and(k1, simd_le(a, b)),
            3 => i16x16::ZERO,
            4 => simd_and(k1, simd_ne(a, b)),
            5 => simd_and(k1, simd_ge(a, b)),
            6 => simd_and(k1, simd_gt(a, b)),
            _ => k1,
        };
        simd_bitmask(r)
    }
}

/// Compare packed signed 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmp_epi16_mask&expand=687)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm_cmp_epi16_mask<const IMM8: i32>(a: __m128i, b: __m128i) -> __mmask8 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_i16x8();
        let b = b.as_i16x8();
        let r = match IMM8 {
            0 => simd_eq(a, b),
            1 => simd_lt(a, b),
            2 => simd_le(a, b),
            3 => i16x8::ZERO,
            4 => simd_ne(a, b),
            5 => simd_ge(a, b),
            6 => simd_gt(a, b),
            _ => i16x8::splat(-1),
        };
        simd_bitmask(r)
    }
}

/// Compare packed signed 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmp_epi16_mask&expand=688)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm_mask_cmp_epi16_mask<const IMM8: i32>(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_i16x8();
        let b = b.as_i16x8();
        let k1 = simd_select_bitmask(k1, i16x8::splat(-1), i16x8::ZERO);
        let r = match IMM8 {
            0 => simd_and(k1, simd_eq(a, b)),
            1 => simd_and(k1, simd_lt(a, b)),
            2 => simd_and(k1, simd_le(a, b)),
            3 => i16x8::ZERO,
            4 => simd_and(k1, simd_ne(a, b)),
            5 => simd_and(k1, simd_ge(a, b)),
            6 => simd_and(k1, simd_gt(a, b)),
            _ => k1,
        };
        simd_bitmask(r)
    }
}

/// Compare packed signed 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmp_epi8_mask&expand=709)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm512_cmp_epi8_mask<const IMM8: i32>(a: __m512i, b: __m512i) -> __mmask64 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_i8x64();
        let b = b.as_i8x64();
        let r = match IMM8 {
            0 => simd_eq(a, b),
            1 => simd_lt(a, b),
            2 => simd_le(a, b),
            3 => i8x64::ZERO,
            4 => simd_ne(a, b),
            5 => simd_ge(a, b),
            6 => simd_gt(a, b),
            _ => i8x64::splat(-1),
        };
        simd_bitmask(r)
    }
}

/// Compare packed signed 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cmp_epi8_mask&expand=710)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm512_mask_cmp_epi8_mask<const IMM8: i32>(
    k1: __mmask64,
    a: __m512i,
    b: __m512i,
) -> __mmask64 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_i8x64();
        let b = b.as_i8x64();
        let k1 = simd_select_bitmask(k1, i8x64::splat(-1), i8x64::ZERO);
        let r = match IMM8 {
            0 => simd_and(k1, simd_eq(a, b)),
            1 => simd_and(k1, simd_lt(a, b)),
            2 => simd_and(k1, simd_le(a, b)),
            3 => i8x64::ZERO,
            4 => simd_and(k1, simd_ne(a, b)),
            5 => simd_and(k1, simd_ge(a, b)),
            6 => simd_and(k1, simd_gt(a, b)),
            _ => k1,
        };
        simd_bitmask(r)
    }
}

/// Compare packed signed 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmp_epi8_mask&expand=707)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm256_cmp_epi8_mask<const IMM8: i32>(a: __m256i, b: __m256i) -> __mmask32 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_i8x32();
        let b = b.as_i8x32();
        let r = match IMM8 {
            0 => simd_eq(a, b),
            1 => simd_lt(a, b),
            2 => simd_le(a, b),
            3 => i8x32::ZERO,
            4 => simd_ne(a, b),
            5 => simd_ge(a, b),
            6 => simd_gt(a, b),
            _ => i8x32::splat(-1),
        };
        simd_bitmask(r)
    }
}

/// Compare packed signed 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cmp_epi8_mask&expand=708)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm256_mask_cmp_epi8_mask<const IMM8: i32>(
    k1: __mmask32,
    a: __m256i,
    b: __m256i,
) -> __mmask32 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_i8x32();
        let b = b.as_i8x32();
        let k1 = simd_select_bitmask(k1, i8x32::splat(-1), i8x32::ZERO);
        let r = match IMM8 {
            0 => simd_and(k1, simd_eq(a, b)),
            1 => simd_and(k1, simd_lt(a, b)),
            2 => simd_and(k1, simd_le(a, b)),
            3 => i8x32::ZERO,
            4 => simd_and(k1, simd_ne(a, b)),
            5 => simd_and(k1, simd_ge(a, b)),
            6 => simd_and(k1, simd_gt(a, b)),
            _ => k1,
        };
        simd_bitmask(r)
    }
}

/// Compare packed signed 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmp_epi8_mask&expand=705)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm_cmp_epi8_mask<const IMM8: i32>(a: __m128i, b: __m128i) -> __mmask16 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_i8x16();
        let b = b.as_i8x16();
        let r = match IMM8 {
            0 => simd_eq(a, b),
            1 => simd_lt(a, b),
            2 => simd_le(a, b),
            3 => i8x16::ZERO,
            4 => simd_ne(a, b),
            5 => simd_ge(a, b),
            6 => simd_gt(a, b),
            _ => i8x16::splat(-1),
        };
        simd_bitmask(r)
    }
}

/// Compare packed signed 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cmp_epi8_mask&expand=706)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub fn _mm_mask_cmp_epi8_mask<const IMM8: i32>(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    unsafe {
        static_assert_uimm_bits!(IMM8, 3);
        let a = a.as_i8x16();
        let b = b.as_i8x16();
        let k1 = simd_select_bitmask(k1, i8x16::splat(-1), i8x16::ZERO);
        let r = match IMM8 {
            0 => simd_and(k1, simd_eq(a, b)),
            1 => simd_and(k1, simd_lt(a, b)),
            2 => simd_and(k1, simd_le(a, b)),
            3 => i8x16::ZERO,
            4 => simd_and(k1, simd_ne(a, b)),
            5 => simd_and(k1, simd_ge(a, b)),
            6 => simd_and(k1, simd_gt(a, b)),
            _ => k1,
        };
        simd_bitmask(r)
    }
}

/// Reduce the packed 16-bit integers in a by addition. Returns the sum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_add_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_add_epi16(a: __m256i) -> i16 {
    unsafe { simd_reduce_add_unordered(a.as_i16x16()) }
}

/// Reduce the packed 16-bit integers in a by addition using mask k. Returns the sum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_add_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_add_epi16(k: __mmask16, a: __m256i) -> i16 {
    unsafe { simd_reduce_add_unordered(simd_select_bitmask(k, a.as_i16x16(), i16x16::ZERO)) }
}

/// Reduce the packed 16-bit integers in a by addition. Returns the sum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_add_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_add_epi16(a: __m128i) -> i16 {
    unsafe { simd_reduce_add_unordered(a.as_i16x8()) }
}

/// Reduce the packed 16-bit integers in a by addition using mask k. Returns the sum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_add_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_add_epi16(k: __mmask8, a: __m128i) -> i16 {
    unsafe { simd_reduce_add_unordered(simd_select_bitmask(k, a.as_i16x8(), i16x8::ZERO)) }
}

/// Reduce the packed 8-bit integers in a by addition. Returns the sum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_add_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_add_epi8(a: __m256i) -> i8 {
    unsafe { simd_reduce_add_unordered(a.as_i8x32()) }
}

/// Reduce the packed 8-bit integers in a by addition using mask k. Returns the sum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_add_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_add_epi8(k: __mmask32, a: __m256i) -> i8 {
    unsafe { simd_reduce_add_unordered(simd_select_bitmask(k, a.as_i8x32(), i8x32::ZERO)) }
}

/// Reduce the packed 8-bit integers in a by addition. Returns the sum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_add_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_add_epi8(a: __m128i) -> i8 {
    unsafe { simd_reduce_add_unordered(a.as_i8x16()) }
}

/// Reduce the packed 8-bit integers in a by addition using mask k. Returns the sum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_add_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_add_epi8(k: __mmask16, a: __m128i) -> i8 {
    unsafe { simd_reduce_add_unordered(simd_select_bitmask(k, a.as_i8x16(), i8x16::ZERO)) }
}

/// Reduce the packed 16-bit integers in a by bitwise AND. Returns the bitwise AND of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_and_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_and_epi16(a: __m256i) -> i16 {
    unsafe { simd_reduce_and(a.as_i16x16()) }
}

/// Reduce the packed 16-bit integers in a by bitwise AND using mask k. Returns the bitwise AND of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_and_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_and_epi16(k: __mmask16, a: __m256i) -> i16 {
    unsafe {
        simd_reduce_and(simd_select_bitmask(
            k,
            a.as_i16x16(),
            _mm256_set1_epi64x(-1).as_i16x16(),
        ))
    }
}

/// Reduce the packed 16-bit integers in a by bitwise AND. Returns the bitwise AND of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_and_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_and_epi16(a: __m128i) -> i16 {
    unsafe { simd_reduce_and(a.as_i16x8()) }
}

/// Reduce the packed 16-bit integers in a by bitwise AND using mask k. Returns the bitwise AND of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_and_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_and_epi16(k: __mmask8, a: __m128i) -> i16 {
    unsafe {
        simd_reduce_and(simd_select_bitmask(
            k,
            a.as_i16x8(),
            _mm_set1_epi64x(-1).as_i16x8(),
        ))
    }
}

/// Reduce the packed 8-bit integers in a by bitwise AND. Returns the bitwise AND of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_and_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_and_epi8(a: __m256i) -> i8 {
    unsafe { simd_reduce_and(a.as_i8x32()) }
}

/// Reduce the packed 8-bit integers in a by bitwise AND using mask k. Returns the bitwise AND of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_and_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_and_epi8(k: __mmask32, a: __m256i) -> i8 {
    unsafe {
        simd_reduce_and(simd_select_bitmask(
            k,
            a.as_i8x32(),
            _mm256_set1_epi64x(-1).as_i8x32(),
        ))
    }
}

/// Reduce the packed 8-bit integers in a by bitwise AND. Returns the bitwise AND of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_and_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_and_epi8(a: __m128i) -> i8 {
    unsafe { simd_reduce_and(a.as_i8x16()) }
}

/// Reduce the packed 8-bit integers in a by bitwise AND using mask k. Returns the bitwise AND of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_and_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_and_epi8(k: __mmask16, a: __m128i) -> i8 {
    unsafe {
        simd_reduce_and(simd_select_bitmask(
            k,
            a.as_i8x16(),
            _mm_set1_epi64x(-1).as_i8x16(),
        ))
    }
}

/// Reduce the packed 16-bit integers in a by maximum. Returns the maximum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_max_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_max_epi16(a: __m256i) -> i16 {
    unsafe { simd_reduce_max(a.as_i16x16()) }
}

/// Reduce the packed 16-bit integers in a by maximum using mask k. Returns the maximum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_max_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_max_epi16(k: __mmask16, a: __m256i) -> i16 {
    unsafe { simd_reduce_max(simd_select_bitmask(k, a.as_i16x16(), i16x16::splat(-32768))) }
}

/// Reduce the packed 16-bit integers in a by maximum. Returns the maximum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_max_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_max_epi16(a: __m128i) -> i16 {
    unsafe { simd_reduce_max(a.as_i16x8()) }
}

/// Reduce the packed 16-bit integers in a by maximum using mask k. Returns the maximum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_max_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_max_epi16(k: __mmask8, a: __m128i) -> i16 {
    unsafe { simd_reduce_max(simd_select_bitmask(k, a.as_i16x8(), i16x8::splat(-32768))) }
}

/// Reduce the packed 8-bit integers in a by maximum. Returns the maximum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_max_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_max_epi8(a: __m256i) -> i8 {
    unsafe { simd_reduce_max(a.as_i8x32()) }
}

/// Reduce the packed 8-bit integers in a by maximum using mask k. Returns the maximum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_max_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_max_epi8(k: __mmask32, a: __m256i) -> i8 {
    unsafe { simd_reduce_max(simd_select_bitmask(k, a.as_i8x32(), i8x32::splat(-128))) }
}

/// Reduce the packed 8-bit integers in a by maximum. Returns the maximum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_max_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_max_epi8(a: __m128i) -> i8 {
    unsafe { simd_reduce_max(a.as_i8x16()) }
}

/// Reduce the packed 8-bit integers in a by maximum using mask k. Returns the maximum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_max_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_max_epi8(k: __mmask16, a: __m128i) -> i8 {
    unsafe { simd_reduce_max(simd_select_bitmask(k, a.as_i8x16(), i8x16::splat(-128))) }
}

/// Reduce the packed unsigned 16-bit integers in a by maximum. Returns the maximum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_max_epu16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_max_epu16(a: __m256i) -> u16 {
    unsafe { simd_reduce_max(a.as_u16x16()) }
}

/// Reduce the packed unsigned 16-bit integers in a by maximum using mask k. Returns the maximum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_max_epu16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_max_epu16(k: __mmask16, a: __m256i) -> u16 {
    unsafe { simd_reduce_max(simd_select_bitmask(k, a.as_u16x16(), u16x16::ZERO)) }
}

/// Reduce the packed unsigned 16-bit integers in a by maximum. Returns the maximum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_max_epu16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_max_epu16(a: __m128i) -> u16 {
    unsafe { simd_reduce_max(a.as_u16x8()) }
}

/// Reduce the packed unsigned 16-bit integers in a by maximum using mask k. Returns the maximum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_max_epu16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_max_epu16(k: __mmask8, a: __m128i) -> u16 {
    unsafe { simd_reduce_max(simd_select_bitmask(k, a.as_u16x8(), u16x8::ZERO)) }
}

/// Reduce the packed unsigned 8-bit integers in a by maximum. Returns the maximum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_max_epu8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_max_epu8(a: __m256i) -> u8 {
    unsafe { simd_reduce_max(a.as_u8x32()) }
}

/// Reduce the packed unsigned 8-bit integers in a by maximum using mask k. Returns the maximum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_max_epu8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_max_epu8(k: __mmask32, a: __m256i) -> u8 {
    unsafe { simd_reduce_max(simd_select_bitmask(k, a.as_u8x32(), u8x32::ZERO)) }
}

/// Reduce the packed unsigned 8-bit integers in a by maximum. Returns the maximum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_max_epu8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_max_epu8(a: __m128i) -> u8 {
    unsafe { simd_reduce_max(a.as_u8x16()) }
}

/// Reduce the packed unsigned 8-bit integers in a by maximum using mask k. Returns the maximum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_max_epu8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_max_epu8(k: __mmask16, a: __m128i) -> u8 {
    unsafe { simd_reduce_max(simd_select_bitmask(k, a.as_u8x16(), u8x16::ZERO)) }
}

/// Reduce the packed 16-bit integers in a by minimum. Returns the minimum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_min_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_min_epi16(a: __m256i) -> i16 {
    unsafe { simd_reduce_min(a.as_i16x16()) }
}

/// Reduce the packed 16-bit integers in a by minimum using mask k. Returns the minimum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_min_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_min_epi16(k: __mmask16, a: __m256i) -> i16 {
    unsafe { simd_reduce_min(simd_select_bitmask(k, a.as_i16x16(), i16x16::splat(0x7fff))) }
}

/// Reduce the packed 16-bit integers in a by minimum. Returns the minimum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_min_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_min_epi16(a: __m128i) -> i16 {
    unsafe { simd_reduce_min(a.as_i16x8()) }
}

/// Reduce the packed 16-bit integers in a by minimum using mask k. Returns the minimum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_min_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_min_epi16(k: __mmask8, a: __m128i) -> i16 {
    unsafe { simd_reduce_min(simd_select_bitmask(k, a.as_i16x8(), i16x8::splat(0x7fff))) }
}

/// Reduce the packed 8-bit integers in a by minimum. Returns the minimum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_min_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_min_epi8(a: __m256i) -> i8 {
    unsafe { simd_reduce_min(a.as_i8x32()) }
}

/// Reduce the packed 8-bit integers in a by minimum using mask k. Returns the minimum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_min_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_min_epi8(k: __mmask32, a: __m256i) -> i8 {
    unsafe { simd_reduce_min(simd_select_bitmask(k, a.as_i8x32(), i8x32::splat(0x7f))) }
}

/// Reduce the packed 8-bit integers in a by minimum. Returns the minimum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_min_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_min_epi8(a: __m128i) -> i8 {
    unsafe { simd_reduce_min(a.as_i8x16()) }
}

/// Reduce the packed 8-bit integers in a by minimum using mask k. Returns the minimum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_min_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_min_epi8(k: __mmask16, a: __m128i) -> i8 {
    unsafe { simd_reduce_min(simd_select_bitmask(k, a.as_i8x16(), i8x16::splat(0x7f))) }
}

/// Reduce the packed unsigned 16-bit integers in a by minimum. Returns the minimum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_min_epu16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_min_epu16(a: __m256i) -> u16 {
    unsafe { simd_reduce_min(a.as_u16x16()) }
}

/// Reduce the packed unsigned 16-bit integers in a by minimum using mask k. Returns the minimum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_min_epu16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_min_epu16(k: __mmask16, a: __m256i) -> u16 {
    unsafe { simd_reduce_min(simd_select_bitmask(k, a.as_u16x16(), u16x16::splat(0xffff))) }
}

/// Reduce the packed unsigned 16-bit integers in a by minimum. Returns the minimum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_min_epu16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_min_epu16(a: __m128i) -> u16 {
    unsafe { simd_reduce_min(a.as_u16x8()) }
}

/// Reduce the packed unsigned 16-bit integers in a by minimum using mask k. Returns the minimum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_min_epu16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_min_epu16(k: __mmask8, a: __m128i) -> u16 {
    unsafe { simd_reduce_min(simd_select_bitmask(k, a.as_u16x8(), u16x8::splat(0xffff))) }
}

/// Reduce the packed unsigned 8-bit integers in a by minimum. Returns the minimum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_min_epu8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_min_epu8(a: __m256i) -> u8 {
    unsafe { simd_reduce_min(a.as_u8x32()) }
}

/// Reduce the packed unsigned 8-bit integers in a by minimum using mask k. Returns the minimum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_min_epu8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_min_epu8(k: __mmask32, a: __m256i) -> u8 {
    unsafe { simd_reduce_min(simd_select_bitmask(k, a.as_u8x32(), u8x32::splat(0xff))) }
}

/// Reduce the packed unsigned 8-bit integers in a by minimum. Returns the minimum of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_min_epu8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_min_epu8(a: __m128i) -> u8 {
    unsafe { simd_reduce_min(a.as_u8x16()) }
}

/// Reduce the packed unsigned 8-bit integers in a by minimum using mask k. Returns the minimum of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_min_epu8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_min_epu8(k: __mmask16, a: __m128i) -> u8 {
    unsafe { simd_reduce_min(simd_select_bitmask(k, a.as_u8x16(), u8x16::splat(0xff))) }
}

/// Reduce the packed 16-bit integers in a by multiplication. Returns the product of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_mul_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_mul_epi16(a: __m256i) -> i16 {
    unsafe { simd_reduce_mul_unordered(a.as_i16x16()) }
}

/// Reduce the packed 16-bit integers in a by multiplication using mask k. Returns the product of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_mul_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_mul_epi16(k: __mmask16, a: __m256i) -> i16 {
    unsafe { simd_reduce_mul_unordered(simd_select_bitmask(k, a.as_i16x16(), i16x16::splat(1))) }
}

/// Reduce the packed 16-bit integers in a by multiplication. Returns the product of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_mul_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_mul_epi16(a: __m128i) -> i16 {
    unsafe { simd_reduce_mul_unordered(a.as_i16x8()) }
}

/// Reduce the packed 16-bit integers in a by multiplication using mask k. Returns the product of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_mul_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_mul_epi16(k: __mmask8, a: __m128i) -> i16 {
    unsafe { simd_reduce_mul_unordered(simd_select_bitmask(k, a.as_i16x8(), i16x8::splat(1))) }
}

/// Reduce the packed 8-bit integers in a by multiplication. Returns the product of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_mul_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_mul_epi8(a: __m256i) -> i8 {
    unsafe { simd_reduce_mul_unordered(a.as_i8x32()) }
}

/// Reduce the packed 8-bit integers in a by multiplication using mask k. Returns the product of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_mul_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_mul_epi8(k: __mmask32, a: __m256i) -> i8 {
    unsafe { simd_reduce_mul_unordered(simd_select_bitmask(k, a.as_i8x32(), i8x32::splat(1))) }
}

/// Reduce the packed 8-bit integers in a by multiplication. Returns the product of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_mul_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_mul_epi8(a: __m128i) -> i8 {
    unsafe { simd_reduce_mul_unordered(a.as_i8x16()) }
}

/// Reduce the packed 8-bit integers in a by multiplication using mask k. Returns the product of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_mul_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_mul_epi8(k: __mmask16, a: __m128i) -> i8 {
    unsafe { simd_reduce_mul_unordered(simd_select_bitmask(k, a.as_i8x16(), i8x16::splat(1))) }
}

/// Reduce the packed 16-bit integers in a by bitwise OR. Returns the bitwise OR of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_or_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_or_epi16(a: __m256i) -> i16 {
    unsafe { simd_reduce_or(a.as_i16x16()) }
}

/// Reduce the packed 16-bit integers in a by bitwise OR using mask k. Returns the bitwise OR of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_or_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_or_epi16(k: __mmask16, a: __m256i) -> i16 {
    unsafe { simd_reduce_or(simd_select_bitmask(k, a.as_i16x16(), i16x16::ZERO)) }
}

/// Reduce the packed 16-bit integers in a by bitwise OR. Returns the bitwise OR of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_or_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_or_epi16(a: __m128i) -> i16 {
    unsafe { simd_reduce_or(a.as_i16x8()) }
}

/// Reduce the packed 16-bit integers in a by bitwise OR using mask k. Returns the bitwise OR of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_or_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_or_epi16(k: __mmask8, a: __m128i) -> i16 {
    unsafe { simd_reduce_or(simd_select_bitmask(k, a.as_i16x8(), i16x8::ZERO)) }
}

/// Reduce the packed 8-bit integers in a by bitwise OR. Returns the bitwise OR of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_reduce_or_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_reduce_or_epi8(a: __m256i) -> i8 {
    unsafe { simd_reduce_or(a.as_i8x32()) }
}

/// Reduce the packed 8-bit integers in a by bitwise OR using mask k. Returns the bitwise OR of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_reduce_or_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm256_mask_reduce_or_epi8(k: __mmask32, a: __m256i) -> i8 {
    unsafe { simd_reduce_or(simd_select_bitmask(k, a.as_i8x32(), i8x32::ZERO)) }
}

/// Reduce the packed 8-bit integers in a by bitwise OR. Returns the bitwise OR of all elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_reduce_or_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_reduce_or_epi8(a: __m128i) -> i8 {
    unsafe { simd_reduce_or(a.as_i8x16()) }
}

/// Reduce the packed 8-bit integers in a by bitwise OR using mask k. Returns the bitwise OR of all active elements in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_reduce_or_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _mm_mask_reduce_or_epi8(k: __mmask16, a: __m128i) -> i8 {
    unsafe { simd_reduce_or(simd_select_bitmask(k, a.as_i8x16(), i8x16::ZERO)) }
}

/// Load 512-bits (composed of 32 packed 16-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_epi16&expand=3368)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu16
pub unsafe fn _mm512_loadu_epi16(mem_addr: *const i16) -> __m512i {
    ptr::read_unaligned(mem_addr as *const __m512i)
}

/// Load 256-bits (composed of 16 packed 16-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_epi16&expand=3365)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu16
pub unsafe fn _mm256_loadu_epi16(mem_addr: *const i16) -> __m256i {
    ptr::read_unaligned(mem_addr as *const __m256i)
}

/// Load 128-bits (composed of 8 packed 16-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_epi16&expand=3362)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu16
pub unsafe fn _mm_loadu_epi16(mem_addr: *const i16) -> __m128i {
    ptr::read_unaligned(mem_addr as *const __m128i)
}

/// Load 512-bits (composed of 64 packed 8-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_epi8&expand=3395)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu8
pub unsafe fn _mm512_loadu_epi8(mem_addr: *const i8) -> __m512i {
    ptr::read_unaligned(mem_addr as *const __m512i)
}

/// Load 256-bits (composed of 32 packed 8-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_epi8&expand=3392)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu8
pub unsafe fn _mm256_loadu_epi8(mem_addr: *const i8) -> __m256i {
    ptr::read_unaligned(mem_addr as *const __m256i)
}

/// Load 128-bits (composed of 16 packed 8-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_epi8&expand=3389)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu8
pub unsafe fn _mm_loadu_epi8(mem_addr: *const i8) -> __m128i {
    ptr::read_unaligned(mem_addr as *const __m128i)
}

/// Store 512-bits (composed of 32 packed 16-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_epi16&expand=5622)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu16
pub unsafe fn _mm512_storeu_epi16(mem_addr: *mut i16, a: __m512i) {
    ptr::write_unaligned(mem_addr as *mut __m512i, a);
}

/// Store 256-bits (composed of 16 packed 16-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_epi16&expand=5620)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu16
pub unsafe fn _mm256_storeu_epi16(mem_addr: *mut i16, a: __m256i) {
    ptr::write_unaligned(mem_addr as *mut __m256i, a);
}

/// Store 128-bits (composed of 8 packed 16-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_epi16&expand=5618)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu16
pub unsafe fn _mm_storeu_epi16(mem_addr: *mut i16, a: __m128i) {
    ptr::write_unaligned(mem_addr as *mut __m128i, a);
}

/// Store 512-bits (composed of 64 packed 8-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_epi8&expand=5640)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu8
pub unsafe fn _mm512_storeu_epi8(mem_addr: *mut i8, a: __m512i) {
    ptr::write_unaligned(mem_addr as *mut __m512i, a);
}

/// Store 256-bits (composed of 32 packed 8-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_epi8&expand=5638)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu8
pub unsafe fn _mm256_storeu_epi8(mem_addr: *mut i8, a: __m256i) {
    ptr::write_unaligned(mem_addr as *mut __m256i, a);
}

/// Store 128-bits (composed of 16 packed 8-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_epi8&expand=5636)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu8
pub unsafe fn _mm_storeu_epi8(mem_addr: *mut i8, a: __m128i) {
    ptr::write_unaligned(mem_addr as *mut __m128i, a);
}

/// Load packed 16-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_loadu_epi16(src: __m512i, k: __mmask32, mem_addr: *const i16) -> __m512i {
    transmute(loaddqu16_512(mem_addr, src.as_i16x32(), k))
}

/// Load packed 16-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_loadu_epi16(k: __mmask32, mem_addr: *const i16) -> __m512i {
    _mm512_mask_loadu_epi16(_mm512_setzero_si512(), k, mem_addr)
}

/// Load packed 8-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_loadu_epi8(src: __m512i, k: __mmask64, mem_addr: *const i8) -> __m512i {
    transmute(loaddqu8_512(mem_addr, src.as_i8x64(), k))
}

/// Load packed 8-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_maskz_loadu_epi8(k: __mmask64, mem_addr: *const i8) -> __m512i {
    _mm512_mask_loadu_epi8(_mm512_setzero_si512(), k, mem_addr)
}

/// Load packed 16-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_loadu_epi16(src: __m256i, k: __mmask16, mem_addr: *const i16) -> __m256i {
    transmute(loaddqu16_256(mem_addr, src.as_i16x16(), k))
}

/// Load packed 16-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_loadu_epi16(k: __mmask16, mem_addr: *const i16) -> __m256i {
    _mm256_mask_loadu_epi16(_mm256_setzero_si256(), k, mem_addr)
}

/// Load packed 8-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_loadu_epi8(src: __m256i, k: __mmask32, mem_addr: *const i8) -> __m256i {
    transmute(loaddqu8_256(mem_addr, src.as_i8x32(), k))
}

/// Load packed 8-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_maskz_loadu_epi8(k: __mmask32, mem_addr: *const i8) -> __m256i {
    _mm256_mask_loadu_epi8(_mm256_setzero_si256(), k, mem_addr)
}

/// Load packed 16-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_loadu_epi16(src: __m128i, k: __mmask8, mem_addr: *const i16) -> __m128i {
    transmute(loaddqu16_128(mem_addr, src.as_i16x8(), k))
}

/// Load packed 16-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_loadu_epi16(k: __mmask8, mem_addr: *const i16) -> __m128i {
    _mm_mask_loadu_epi16(_mm_setzero_si128(), k, mem_addr)
}

/// Load packed 8-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_loadu_epi8(src: __m128i, k: __mmask16, mem_addr: *const i8) -> __m128i {
    transmute(loaddqu8_128(mem_addr, src.as_i8x16(), k))
}

/// Load packed 8-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_maskz_loadu_epi8(k: __mmask16, mem_addr: *const i8) -> __m128i {
    _mm_mask_loadu_epi8(_mm_setzero_si128(), k, mem_addr)
}

/// Store packed 16-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_epi16)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_storeu_epi16(mem_addr: *mut i16, mask: __mmask32, a: __m512i) {
    storedqu16_512(mem_addr, a.as_i16x32(), mask)
}

/// Store packed 8-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm512_mask_storeu_epi8(mem_addr: *mut i8, mask: __mmask64, a: __m512i) {
    storedqu8_512(mem_addr, a.as_i8x64(), mask)
}

/// Store packed 16-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_storeu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_storeu_epi16(mem_addr: *mut i16, mask: __mmask16, a: __m256i) {
    storedqu16_256(mem_addr, a.as_i16x16(), mask)
}

/// Store packed 8-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm256_mask_storeu_epi8(mem_addr: *mut i8, mask: __mmask32, a: __m256i) {
    storedqu8_256(mem_addr, a.as_i8x32(), mask)
}

/// Store packed 16-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_storeu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_storeu_epi16(mem_addr: *mut i16, mask: __mmask8, a: __m128i) {
    storedqu16_128(mem_addr, a.as_i16x8(), mask)
}

/// Store packed 8-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _mm_mask_storeu_epi8(mem_addr: *mut i8, mask: __mmask16, a: __m128i) {
    storedqu8_128(mem_addr, a.as_i8x16(), mask)
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_madd_epi16&expand=3511)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub fn _mm512_madd_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(vpmaddwd(a.as_i16x32(), b.as_i16x32())) }
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_madd_epi16&expand=3512)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub fn _mm512_mask_madd_epi16(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let madd = _mm512_madd_epi16(a, b).as_i32x16();
        transmute(simd_select_bitmask(k, madd, src.as_i32x16()))
    }
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_madd_epi16&expand=3513)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub fn _mm512_maskz_madd_epi16(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let madd = _mm512_madd_epi16(a, b).as_i32x16();
        transmute(simd_select_bitmask(k, madd, i32x16::ZERO))
    }
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_madd_epi16&expand=3509)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub fn _mm256_mask_madd_epi16(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let madd = _mm256_madd_epi16(a, b).as_i32x8();
        transmute(simd_select_bitmask(k, madd, src.as_i32x8()))
    }
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_madd_epi16&expand=3510)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub fn _mm256_maskz_madd_epi16(k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let madd = _mm256_madd_epi16(a, b).as_i32x8();
        transmute(simd_select_bitmask(k, madd, i32x8::ZERO))
    }
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_madd_epi16&expand=3506)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub fn _mm_mask_madd_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let madd = _mm_madd_epi16(a, b).as_i32x4();
        transmute(simd_select_bitmask(k, madd, src.as_i32x4()))
    }
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_madd_epi16&expand=3507)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub fn _mm_maskz_madd_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let madd = _mm_madd_epi16(a, b).as_i32x4();
        transmute(simd_select_bitmask(k, madd, i32x4::ZERO))
    }
}

/// Vertically multiply each unsigned 8-bit integer from a with the corresponding signed 8-bit integer from b, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers, and pack the saturated results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maddubs_epi16&expand=3539)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub fn _mm512_maddubs_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(vpmaddubsw(a.as_i8x64(), b.as_i8x64())) }
}

/// Multiply packed unsigned 8-bit integers in a by packed signed 8-bit integers in b, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers, and pack the saturated results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_maddubs_epi16&expand=3540)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub fn _mm512_mask_maddubs_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let madd = _mm512_maddubs_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, madd, src.as_i16x32()))
    }
}

/// Multiply packed unsigned 8-bit integers in a by packed signed 8-bit integers in b, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers, and pack the saturated results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_maddubs_epi16&expand=3541)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub fn _mm512_maskz_maddubs_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let madd = _mm512_maddubs_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, madd, i16x32::ZERO))
    }
}

/// Multiply packed unsigned 8-bit integers in a by packed signed 8-bit integers in b, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers, and pack the saturated results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_maddubs_epi16&expand=3537)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub fn _mm256_mask_maddubs_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let madd = _mm256_maddubs_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, madd, src.as_i16x16()))
    }
}

/// Multiply packed unsigned 8-bit integers in a by packed signed 8-bit integers in b, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers, and pack the saturated results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_maddubs_epi16&expand=3538)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub fn _mm256_maskz_maddubs_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let madd = _mm256_maddubs_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, madd, i16x16::ZERO))
    }
}

/// Multiply packed unsigned 8-bit integers in a by packed signed 8-bit integers in b, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers, and pack the saturated results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_maddubs_epi16&expand=3534)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub fn _mm_mask_maddubs_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let madd = _mm_maddubs_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, madd, src.as_i16x8()))
    }
}

/// Multiply packed unsigned 8-bit integers in a by packed signed 8-bit integers in b, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers, and pack the saturated results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_maddubs_epi16&expand=3535)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub fn _mm_maskz_maddubs_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let madd = _mm_maddubs_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, madd, i16x8::ZERO))
    }
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_packs_epi32&expand=4091)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub fn _mm512_packs_epi32(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(vpackssdw(a.as_i32x16(), b.as_i32x16())) }
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_packs_epi32&expand=4089)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub fn _mm512_mask_packs_epi32(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let pack = _mm512_packs_epi32(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, pack, src.as_i16x32()))
    }
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_packs_epi32&expand=4090)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub fn _mm512_maskz_packs_epi32(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let pack = _mm512_packs_epi32(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, pack, i16x32::ZERO))
    }
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_packs_epi32&expand=4086)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub fn _mm256_mask_packs_epi32(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let pack = _mm256_packs_epi32(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, pack, src.as_i16x16()))
    }
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_packs_epi32&expand=4087)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub fn _mm256_maskz_packs_epi32(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let pack = _mm256_packs_epi32(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, pack, i16x16::ZERO))
    }
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_packs_epi32&expand=4083)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub fn _mm_mask_packs_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let pack = _mm_packs_epi32(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, pack, src.as_i16x8()))
    }
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_packs_epi32&expand=4084)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub fn _mm_maskz_packs_epi32(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let pack = _mm_packs_epi32(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, pack, i16x8::ZERO))
    }
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_packs_epi16&expand=4082)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub fn _mm512_packs_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(vpacksswb(a.as_i16x32(), b.as_i16x32())) }
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_packs_epi16&expand=4080)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub fn _mm512_mask_packs_epi16(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let pack = _mm512_packs_epi16(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, pack, src.as_i8x64()))
    }
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_packs_epi16&expand=4081)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub fn _mm512_maskz_packs_epi16(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let pack = _mm512_packs_epi16(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, pack, i8x64::ZERO))
    }
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_packs_epi16&expand=4077)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub fn _mm256_mask_packs_epi16(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let pack = _mm256_packs_epi16(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, pack, src.as_i8x32()))
    }
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=#text=_mm256_maskz_packs_epi16&expand=4078)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub fn _mm256_maskz_packs_epi16(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let pack = _mm256_packs_epi16(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, pack, i8x32::ZERO))
    }
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_packs_epi16&expand=4074)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub fn _mm_mask_packs_epi16(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let pack = _mm_packs_epi16(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, pack, src.as_i8x16()))
    }
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_packs_epi16&expand=4075)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub fn _mm_maskz_packs_epi16(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let pack = _mm_packs_epi16(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, pack, i8x16::ZERO))
    }
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_packus_epi32&expand=4130)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub fn _mm512_packus_epi32(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(vpackusdw(a.as_i32x16(), b.as_i32x16())) }
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_packus_epi32&expand=4128)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub fn _mm512_mask_packus_epi32(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let pack = _mm512_packus_epi32(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, pack, src.as_i16x32()))
    }
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_packus_epi32&expand=4129)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub fn _mm512_maskz_packus_epi32(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let pack = _mm512_packus_epi32(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, pack, i16x32::ZERO))
    }
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_packus_epi32&expand=4125)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub fn _mm256_mask_packus_epi32(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let pack = _mm256_packus_epi32(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, pack, src.as_i16x16()))
    }
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_packus_epi32&expand=4126)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub fn _mm256_maskz_packus_epi32(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let pack = _mm256_packus_epi32(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, pack, i16x16::ZERO))
    }
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_packus_epi32&expand=4122)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub fn _mm_mask_packus_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let pack = _mm_packus_epi32(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, pack, src.as_i16x8()))
    }
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_packus_epi32&expand=4123)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub fn _mm_maskz_packus_epi32(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let pack = _mm_packus_epi32(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, pack, i16x8::ZERO))
    }
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_packus_epi16&expand=4121)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub fn _mm512_packus_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(vpackuswb(a.as_i16x32(), b.as_i16x32())) }
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_packus_epi16&expand=4119)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub fn _mm512_mask_packus_epi16(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let pack = _mm512_packus_epi16(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, pack, src.as_i8x64()))
    }
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_packus_epi16&expand=4120)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub fn _mm512_maskz_packus_epi16(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let pack = _mm512_packus_epi16(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, pack, i8x64::ZERO))
    }
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_packus_epi16&expand=4116)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub fn _mm256_mask_packus_epi16(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let pack = _mm256_packus_epi16(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, pack, src.as_i8x32()))
    }
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_packus_epi16&expand=4117)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub fn _mm256_maskz_packus_epi16(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let pack = _mm256_packus_epi16(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, pack, i8x32::ZERO))
    }
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_packus_epi16&expand=4113)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub fn _mm_mask_packus_epi16(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let pack = _mm_packus_epi16(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, pack, src.as_i8x16()))
    }
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_packus_epi16&expand=4114)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub fn _mm_maskz_packus_epi16(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let pack = _mm_packus_epi16(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, pack, i8x16::ZERO))
    }
}

/// Average packed unsigned 16-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_avg_epu16&expand=388)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpavgw))]
pub fn _mm512_avg_epu16(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = simd_cast::<_, u32x32>(a.as_u16x32());
        let b = simd_cast::<_, u32x32>(b.as_u16x32());
        let r = simd_shr(simd_add(simd_add(a, b), u32x32::splat(1)), u32x32::splat(1));
        transmute(simd_cast::<_, u16x32>(r))
    }
}

/// Average packed unsigned 16-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_avg_epu16&expand=389)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpavgw))]
pub fn _mm512_mask_avg_epu16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let avg = _mm512_avg_epu16(a, b).as_u16x32();
        transmute(simd_select_bitmask(k, avg, src.as_u16x32()))
    }
}

/// Average packed unsigned 16-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_avg_epu16&expand=390)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpavgw))]
pub fn _mm512_maskz_avg_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let avg = _mm512_avg_epu16(a, b).as_u16x32();
        transmute(simd_select_bitmask(k, avg, u16x32::ZERO))
    }
}

/// Average packed unsigned 16-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_avg_epu16&expand=386)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpavgw))]
pub fn _mm256_mask_avg_epu16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let avg = _mm256_avg_epu16(a, b).as_u16x16();
        transmute(simd_select_bitmask(k, avg, src.as_u16x16()))
    }
}

/// Average packed unsigned 16-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_avg_epu16&expand=387)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpavgw))]
pub fn _mm256_maskz_avg_epu16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let avg = _mm256_avg_epu16(a, b).as_u16x16();
        transmute(simd_select_bitmask(k, avg, u16x16::ZERO))
    }
}

/// Average packed unsigned 16-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_avg_epu16&expand=383)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpavgw))]
pub fn _mm_mask_avg_epu16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let avg = _mm_avg_epu16(a, b).as_u16x8();
        transmute(simd_select_bitmask(k, avg, src.as_u16x8()))
    }
}

/// Average packed unsigned 16-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_avg_epu16&expand=384)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpavgw))]
pub fn _mm_maskz_avg_epu16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let avg = _mm_avg_epu16(a, b).as_u16x8();
        transmute(simd_select_bitmask(k, avg, u16x8::ZERO))
    }
}

/// Average packed unsigned 8-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_avg_epu8&expand=397)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpavgb))]
pub fn _mm512_avg_epu8(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = simd_cast::<_, u16x64>(a.as_u8x64());
        let b = simd_cast::<_, u16x64>(b.as_u8x64());
        let r = simd_shr(simd_add(simd_add(a, b), u16x64::splat(1)), u16x64::splat(1));
        transmute(simd_cast::<_, u8x64>(r))
    }
}

/// Average packed unsigned 8-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_avg_epu8&expand=398)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpavgb))]
pub fn _mm512_mask_avg_epu8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let avg = _mm512_avg_epu8(a, b).as_u8x64();
        transmute(simd_select_bitmask(k, avg, src.as_u8x64()))
    }
}

/// Average packed unsigned 8-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_avg_epu8&expand=399)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpavgb))]
pub fn _mm512_maskz_avg_epu8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let avg = _mm512_avg_epu8(a, b).as_u8x64();
        transmute(simd_select_bitmask(k, avg, u8x64::ZERO))
    }
}

/// Average packed unsigned 8-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_avg_epu8&expand=395)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpavgb))]
pub fn _mm256_mask_avg_epu8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let avg = _mm256_avg_epu8(a, b).as_u8x32();
        transmute(simd_select_bitmask(k, avg, src.as_u8x32()))
    }
}

/// Average packed unsigned 8-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_avg_epu8&expand=396)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpavgb))]
pub fn _mm256_maskz_avg_epu8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let avg = _mm256_avg_epu8(a, b).as_u8x32();
        transmute(simd_select_bitmask(k, avg, u8x32::ZERO))
    }
}

/// Average packed unsigned 8-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_avg_epu8&expand=392)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpavgb))]
pub fn _mm_mask_avg_epu8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let avg = _mm_avg_epu8(a, b).as_u8x16();
        transmute(simd_select_bitmask(k, avg, src.as_u8x16()))
    }
}

/// Average packed unsigned 8-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_avg_epu8&expand=393)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpavgb))]
pub fn _mm_maskz_avg_epu8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let avg = _mm_avg_epu8(a, b).as_u8x16();
        transmute(simd_select_bitmask(k, avg, u8x16::ZERO))
    }
}

/// Shift packed 16-bit integers in a left by count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_sll_epi16&expand=5271)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllw))]
pub fn _mm512_sll_epi16(a: __m512i, count: __m128i) -> __m512i {
    unsafe { transmute(vpsllw(a.as_i16x32(), count.as_i16x8())) }
}

/// Shift packed 16-bit integers in a left by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_sll_epi16&expand=5269)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllw))]
pub fn _mm512_mask_sll_epi16(src: __m512i, k: __mmask32, a: __m512i, count: __m128i) -> __m512i {
    unsafe {
        let shf = _mm512_sll_epi16(a, count).as_i16x32();
        transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
    }
}

/// Shift packed 16-bit integers in a left by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_sll_epi16&expand=5270)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllw))]
pub fn _mm512_maskz_sll_epi16(k: __mmask32, a: __m512i, count: __m128i) -> __m512i {
    unsafe {
        let shf = _mm512_sll_epi16(a, count).as_i16x32();
        transmute(simd_select_bitmask(k, shf, i16x32::ZERO))
    }
}

/// Shift packed 16-bit integers in a left by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_sll_epi16&expand=5266)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllw))]
pub fn _mm256_mask_sll_epi16(src: __m256i, k: __mmask16, a: __m256i, count: __m128i) -> __m256i {
    unsafe {
        let shf = _mm256_sll_epi16(a, count).as_i16x16();
        transmute(simd_select_bitmask(k, shf, src.as_i16x16()))
    }
}

/// Shift packed 16-bit integers in a left by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_sll_epi16&expand=5267)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllw))]
pub fn _mm256_maskz_sll_epi16(k: __mmask16, a: __m256i, count: __m128i) -> __m256i {
    unsafe {
        let shf = _mm256_sll_epi16(a, count).as_i16x16();
        transmute(simd_select_bitmask(k, shf, i16x16::ZERO))
    }
}

/// Shift packed 16-bit integers in a left by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_sll_epi16&expand=5263)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllw))]
pub fn _mm_mask_sll_epi16(src: __m128i, k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_sll_epi16(a, count).as_i16x8();
        transmute(simd_select_bitmask(k, shf, src.as_i16x8()))
    }
}

/// Shift packed 16-bit integers in a left by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_sll_epi16&expand=5264)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllw))]
pub fn _mm_maskz_sll_epi16(k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_sll_epi16(a, count).as_i16x8();
        transmute(simd_select_bitmask(k, shf, i16x8::ZERO))
    }
}

/// Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_slli_epi16&expand=5301)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllw, IMM8 = 5))]
#[rustc_legacy_const_generics(1)]
pub fn _mm512_slli_epi16<const IMM8: u32>(a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        if IMM8 >= 16 {
            _mm512_setzero_si512()
        } else {
            transmute(simd_shl(a.as_u16x32(), u16x32::splat(IMM8 as u16)))
        }
    }
}

/// Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_slli_epi16&expand=5299)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllw, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm512_mask_slli_epi16<const IMM8: u32>(src: __m512i, k: __mmask32, a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = if IMM8 >= 16 {
            u16x32::ZERO
        } else {
            simd_shl(a.as_u16x32(), u16x32::splat(IMM8 as u16))
        };
        transmute(simd_select_bitmask(k, shf, src.as_u16x32()))
    }
}

/// Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_slli_epi16&expand=5300)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllw, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm512_maskz_slli_epi16<const IMM8: u32>(k: __mmask32, a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        if IMM8 >= 16 {
            _mm512_setzero_si512()
        } else {
            let shf = simd_shl(a.as_u16x32(), u16x32::splat(IMM8 as u16));
            transmute(simd_select_bitmask(k, shf, u16x32::ZERO))
        }
    }
}

/// Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_slli_epi16&expand=5296)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllw, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm256_mask_slli_epi16<const IMM8: u32>(src: __m256i, k: __mmask16, a: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = if IMM8 >= 16 {
            u16x16::ZERO
        } else {
            simd_shl(a.as_u16x16(), u16x16::splat(IMM8 as u16))
        };
        transmute(simd_select_bitmask(k, shf, src.as_u16x16()))
    }
}

/// Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_slli_epi16&expand=5297)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllw, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm256_maskz_slli_epi16<const IMM8: u32>(k: __mmask16, a: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        if IMM8 >= 16 {
            _mm256_setzero_si256()
        } else {
            let shf = simd_shl(a.as_u16x16(), u16x16::splat(IMM8 as u16));
            transmute(simd_select_bitmask(k, shf, u16x16::ZERO))
        }
    }
}

/// Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_slli_epi16&expand=5293)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllw, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm_mask_slli_epi16<const IMM8: u32>(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = if IMM8 >= 16 {
            u16x8::ZERO
        } else {
            simd_shl(a.as_u16x8(), u16x8::splat(IMM8 as u16))
        };
        transmute(simd_select_bitmask(k, shf, src.as_u16x8()))
    }
}

/// Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_slli_epi16&expand=5294)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllw, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm_maskz_slli_epi16<const IMM8: u32>(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        if IMM8 >= 16 {
            _mm_setzero_si128()
        } else {
            let shf = simd_shl(a.as_u16x8(), u16x8::splat(IMM8 as u16));
            transmute(simd_select_bitmask(k, shf, u16x8::ZERO))
        }
    }
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_sllv_epi16&expand=5333)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub fn _mm512_sllv_epi16(a: __m512i, count: __m512i) -> __m512i {
    unsafe { transmute(vpsllvw(a.as_i16x32(), count.as_i16x32())) }
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_sllv_epi16&expand=5331)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub fn _mm512_mask_sllv_epi16(src: __m512i, k: __mmask32, a: __m512i, count: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_sllv_epi16(a, count).as_i16x32();
        transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
    }
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_sllv_epi16&expand=5332)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub fn _mm512_maskz_sllv_epi16(k: __mmask32, a: __m512i, count: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_sllv_epi16(a, count).as_i16x32();
        transmute(simd_select_bitmask(k, shf, i16x32::ZERO))
    }
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_sllv_epi16&expand=5330)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub fn _mm256_sllv_epi16(a: __m256i, count: __m256i) -> __m256i {
    unsafe { transmute(vpsllvw256(a.as_i16x16(), count.as_i16x16())) }
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_sllv_epi16&expand=5328)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub fn _mm256_mask_sllv_epi16(src: __m256i, k: __mmask16, a: __m256i, count: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_sllv_epi16(a, count).as_i16x16();
        transmute(simd_select_bitmask(k, shf, src.as_i16x16()))
    }
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_sllv_epi16&expand=5329)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub fn _mm256_maskz_sllv_epi16(k: __mmask16, a: __m256i, count: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_sllv_epi16(a, count).as_i16x16();
        transmute(simd_select_bitmask(k, shf, i16x16::ZERO))
    }
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sllv_epi16&expand=5327)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub fn _mm_sllv_epi16(a: __m128i, count: __m128i) -> __m128i {
    unsafe { transmute(vpsllvw128(a.as_i16x8(), count.as_i16x8())) }
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_sllv_epi16&expand=5325)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub fn _mm_mask_sllv_epi16(src: __m128i, k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_sllv_epi16(a, count).as_i16x8();
        transmute(simd_select_bitmask(k, shf, src.as_i16x8()))
    }
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_sllv_epi16&expand=5326)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub fn _mm_maskz_sllv_epi16(k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_sllv_epi16(a, count).as_i16x8();
        transmute(simd_select_bitmask(k, shf, i16x8::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_srl_epi16&expand=5483)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub fn _mm512_srl_epi16(a: __m512i, count: __m128i) -> __m512i {
    unsafe { transmute(vpsrlw(a.as_i16x32(), count.as_i16x8())) }
}

/// Shift packed 16-bit integers in a right by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_srl_epi16&expand=5481)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub fn _mm512_mask_srl_epi16(src: __m512i, k: __mmask32, a: __m512i, count: __m128i) -> __m512i {
    unsafe {
        let shf = _mm512_srl_epi16(a, count).as_i16x32();
        transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
    }
}

/// Shift packed 16-bit integers in a right by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_srl_epi16&expand=5482)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub fn _mm512_maskz_srl_epi16(k: __mmask32, a: __m512i, count: __m128i) -> __m512i {
    unsafe {
        let shf = _mm512_srl_epi16(a, count).as_i16x32();
        transmute(simd_select_bitmask(k, shf, i16x32::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_srl_epi16&expand=5478)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub fn _mm256_mask_srl_epi16(src: __m256i, k: __mmask16, a: __m256i, count: __m128i) -> __m256i {
    unsafe {
        let shf = _mm256_srl_epi16(a, count).as_i16x16();
        transmute(simd_select_bitmask(k, shf, src.as_i16x16()))
    }
}

/// Shift packed 16-bit integers in a right by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_srl_epi16&expand=5479)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub fn _mm256_maskz_srl_epi16(k: __mmask16, a: __m256i, count: __m128i) -> __m256i {
    unsafe {
        let shf = _mm256_srl_epi16(a, count).as_i16x16();
        transmute(simd_select_bitmask(k, shf, i16x16::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_srl_epi16&expand=5475)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub fn _mm_mask_srl_epi16(src: __m128i, k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_srl_epi16(a, count).as_i16x8();
        transmute(simd_select_bitmask(k, shf, src.as_i16x8()))
    }
}

/// Shift packed 16-bit integers in a right by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_srl_epi16&expand=5476)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub fn _mm_maskz_srl_epi16(k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_srl_epi16(a, count).as_i16x8();
        transmute(simd_select_bitmask(k, shf, i16x8::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_srli_epi16&expand=5513)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlw, IMM8 = 5))]
#[rustc_legacy_const_generics(1)]
pub fn _mm512_srli_epi16<const IMM8: u32>(a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        if IMM8 >= 16 {
            _mm512_setzero_si512()
        } else {
            transmute(simd_shr(a.as_u16x32(), u16x32::splat(IMM8 as u16)))
        }
    }
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_srli_epi16&expand=5511)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlw, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm512_mask_srli_epi16<const IMM8: u32>(src: __m512i, k: __mmask32, a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = if IMM8 >= 16 {
            u16x32::ZERO
        } else {
            simd_shr(a.as_u16x32(), u16x32::splat(IMM8 as u16))
        };
        transmute(simd_select_bitmask(k, shf, src.as_u16x32()))
    }
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_srli_epi16&expand=5512)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlw, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm512_maskz_srli_epi16<const IMM8: i32>(k: __mmask32, a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        //imm8 should be u32, it seems the document to verify is incorrect
        if IMM8 >= 16 {
            _mm512_setzero_si512()
        } else {
            let shf = simd_shr(a.as_u16x32(), u16x32::splat(IMM8 as u16));
            transmute(simd_select_bitmask(k, shf, u16x32::ZERO))
        }
    }
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_srli_epi16&expand=5508)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlw, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm256_mask_srli_epi16<const IMM8: i32>(src: __m256i, k: __mmask16, a: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm256_srli_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, shf.as_i16x16(), src.as_i16x16()))
    }
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_srli_epi16&expand=5509)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlw, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm256_maskz_srli_epi16<const IMM8: i32>(k: __mmask16, a: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm256_srli_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, shf.as_i16x16(), i16x16::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_srli_epi16&expand=5505)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlw, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm_mask_srli_epi16<const IMM8: i32>(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm_srli_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, shf.as_i16x8(), src.as_i16x8()))
    }
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_srli_epi16&expand=5506)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlw, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm_maskz_srli_epi16<const IMM8: i32>(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = _mm_srli_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, shf.as_i16x8(), i16x8::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_srlv_epi16&expand=5545)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub fn _mm512_srlv_epi16(a: __m512i, count: __m512i) -> __m512i {
    unsafe { transmute(vpsrlvw(a.as_i16x32(), count.as_i16x32())) }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_srlv_epi16&expand=5543)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub fn _mm512_mask_srlv_epi16(src: __m512i, k: __mmask32, a: __m512i, count: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_srlv_epi16(a, count).as_i16x32();
        transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
    }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_srlv_epi16&expand=5544)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub fn _mm512_maskz_srlv_epi16(k: __mmask32, a: __m512i, count: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_srlv_epi16(a, count).as_i16x32();
        transmute(simd_select_bitmask(k, shf, i16x32::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_srlv_epi16&expand=5542)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub fn _mm256_srlv_epi16(a: __m256i, count: __m256i) -> __m256i {
    unsafe { transmute(vpsrlvw256(a.as_i16x16(), count.as_i16x16())) }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_srlv_epi16&expand=5540)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub fn _mm256_mask_srlv_epi16(src: __m256i, k: __mmask16, a: __m256i, count: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_srlv_epi16(a, count).as_i16x16();
        transmute(simd_select_bitmask(k, shf, src.as_i16x16()))
    }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_srlv_epi16&expand=5541)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub fn _mm256_maskz_srlv_epi16(k: __mmask16, a: __m256i, count: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_srlv_epi16(a, count).as_i16x16();
        transmute(simd_select_bitmask(k, shf, i16x16::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_srlv_epi16&expand=5539)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub fn _mm_srlv_epi16(a: __m128i, count: __m128i) -> __m128i {
    unsafe { transmute(vpsrlvw128(a.as_i16x8(), count.as_i16x8())) }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_srlv_epi16&expand=5537)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub fn _mm_mask_srlv_epi16(src: __m128i, k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_srlv_epi16(a, count).as_i16x8();
        transmute(simd_select_bitmask(k, shf, src.as_i16x8()))
    }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_srlv_epi16&expand=5538)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub fn _mm_maskz_srlv_epi16(k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_srlv_epi16(a, count).as_i16x8();
        transmute(simd_select_bitmask(k, shf, i16x8::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by count while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_sra_epi16&expand=5398)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsraw))]
pub fn _mm512_sra_epi16(a: __m512i, count: __m128i) -> __m512i {
    unsafe { transmute(vpsraw(a.as_i16x32(), count.as_i16x8())) }
}

/// Shift packed 16-bit integers in a right by count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_sra_epi16&expand=5396)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsraw))]
pub fn _mm512_mask_sra_epi16(src: __m512i, k: __mmask32, a: __m512i, count: __m128i) -> __m512i {
    unsafe {
        let shf = _mm512_sra_epi16(a, count).as_i16x32();
        transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
    }
}

/// Shift packed 16-bit integers in a right by count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_sra_epi16&expand=5397)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsraw))]
pub fn _mm512_maskz_sra_epi16(k: __mmask32, a: __m512i, count: __m128i) -> __m512i {
    unsafe {
        let shf = _mm512_sra_epi16(a, count).as_i16x32();
        transmute(simd_select_bitmask(k, shf, i16x32::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_sra_epi16&expand=5393)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsraw))]
pub fn _mm256_mask_sra_epi16(src: __m256i, k: __mmask16, a: __m256i, count: __m128i) -> __m256i {
    unsafe {
        let shf = _mm256_sra_epi16(a, count).as_i16x16();
        transmute(simd_select_bitmask(k, shf, src.as_i16x16()))
    }
}

/// Shift packed 16-bit integers in a right by count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_sra_epi16&expand=5394)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsraw))]
pub fn _mm256_maskz_sra_epi16(k: __mmask16, a: __m256i, count: __m128i) -> __m256i {
    unsafe {
        let shf = _mm256_sra_epi16(a, count).as_i16x16();
        transmute(simd_select_bitmask(k, shf, i16x16::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_sra_epi16&expand=5390)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsraw))]
pub fn _mm_mask_sra_epi16(src: __m128i, k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_sra_epi16(a, count).as_i16x8();
        transmute(simd_select_bitmask(k, shf, src.as_i16x8()))
    }
}

/// Shift packed 16-bit integers in a right by count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_sra_epi16&expand=5391)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsraw))]
pub fn _mm_maskz_sra_epi16(k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_sra_epi16(a, count).as_i16x8();
        transmute(simd_select_bitmask(k, shf, i16x8::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_srai_epi16&expand=5427)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsraw, IMM8 = 1))]
#[rustc_legacy_const_generics(1)]
pub fn _mm512_srai_epi16<const IMM8: u32>(a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        transmute(simd_shr(a.as_i16x32(), i16x32::splat(IMM8.min(15) as i16)))
    }
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_srai_epi16&expand=5425)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsraw, IMM8 = 1))]
#[rustc_legacy_const_generics(3)]
pub fn _mm512_mask_srai_epi16<const IMM8: u32>(src: __m512i, k: __mmask32, a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = simd_shr(a.as_i16x32(), i16x32::splat(IMM8.min(15) as i16));
        transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
    }
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_srai_epi16&expand=5426)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsraw, IMM8 = 1))]
#[rustc_legacy_const_generics(2)]
pub fn _mm512_maskz_srai_epi16<const IMM8: u32>(k: __mmask32, a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shf = simd_shr(a.as_i16x32(), i16x32::splat(IMM8.min(15) as i16));
        transmute(simd_select_bitmask(k, shf, i16x32::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_srai_epi16&expand=5422)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsraw, IMM8 = 1))]
#[rustc_legacy_const_generics(3)]
pub fn _mm256_mask_srai_epi16<const IMM8: u32>(src: __m256i, k: __mmask16, a: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let r = simd_shr(a.as_i16x16(), i16x16::splat(IMM8.min(15) as i16));
        transmute(simd_select_bitmask(k, r, src.as_i16x16()))
    }
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_srai_epi16&expand=5423)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsraw, IMM8 = 1))]
#[rustc_legacy_const_generics(2)]
pub fn _mm256_maskz_srai_epi16<const IMM8: u32>(k: __mmask16, a: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let r = simd_shr(a.as_i16x16(), i16x16::splat(IMM8.min(15) as i16));
        transmute(simd_select_bitmask(k, r, i16x16::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_srai_epi16&expand=5419)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsraw, IMM8 = 1))]
#[rustc_legacy_const_generics(3)]
pub fn _mm_mask_srai_epi16<const IMM8: u32>(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let r = simd_shr(a.as_i16x8(), i16x8::splat(IMM8.min(15) as i16));
        transmute(simd_select_bitmask(k, r, src.as_i16x8()))
    }
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_srai_epi16&expand=5420)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsraw, IMM8 = 1))]
#[rustc_legacy_const_generics(2)]
pub fn _mm_maskz_srai_epi16<const IMM8: u32>(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let r = simd_shr(a.as_i16x8(), i16x8::splat(IMM8.min(15) as i16));
        transmute(simd_select_bitmask(k, r, i16x8::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_srav_epi16&expand=5456)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub fn _mm512_srav_epi16(a: __m512i, count: __m512i) -> __m512i {
    unsafe { transmute(vpsravw(a.as_i16x32(), count.as_i16x32())) }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_srav_epi16&expand=5454)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub fn _mm512_mask_srav_epi16(src: __m512i, k: __mmask32, a: __m512i, count: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_srav_epi16(a, count).as_i16x32();
        transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
    }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_srav_epi16&expand=5455)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub fn _mm512_maskz_srav_epi16(k: __mmask32, a: __m512i, count: __m512i) -> __m512i {
    unsafe {
        let shf = _mm512_srav_epi16(a, count).as_i16x32();
        transmute(simd_select_bitmask(k, shf, i16x32::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_srav_epi16&expand=5453)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub fn _mm256_srav_epi16(a: __m256i, count: __m256i) -> __m256i {
    unsafe { transmute(vpsravw256(a.as_i16x16(), count.as_i16x16())) }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_srav_epi16&expand=5451)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub fn _mm256_mask_srav_epi16(src: __m256i, k: __mmask16, a: __m256i, count: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_srav_epi16(a, count).as_i16x16();
        transmute(simd_select_bitmask(k, shf, src.as_i16x16()))
    }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_srav_epi16&expand=5452)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub fn _mm256_maskz_srav_epi16(k: __mmask16, a: __m256i, count: __m256i) -> __m256i {
    unsafe {
        let shf = _mm256_srav_epi16(a, count).as_i16x16();
        transmute(simd_select_bitmask(k, shf, i16x16::ZERO))
    }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_srav_epi16&expand=5450)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub fn _mm_srav_epi16(a: __m128i, count: __m128i) -> __m128i {
    unsafe { transmute(vpsravw128(a.as_i16x8(), count.as_i16x8())) }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_srav_epi16&expand=5448)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub fn _mm_mask_srav_epi16(src: __m128i, k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_srav_epi16(a, count).as_i16x8();
        transmute(simd_select_bitmask(k, shf, src.as_i16x8()))
    }
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_srav_epi16&expand=5449)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub fn _mm_maskz_srav_epi16(k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    unsafe {
        let shf = _mm_srav_epi16(a, count).as_i16x8();
        transmute(simd_select_bitmask(k, shf, i16x8::ZERO))
    }
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_permutex2var_epi16&expand=4226)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vperm))] //vpermi2w or vpermt2w
pub fn _mm512_permutex2var_epi16(a: __m512i, idx: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(vpermi2w(a.as_i16x32(), idx.as_i16x32(), b.as_i16x32())) }
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_permutex2var_epi16&expand=4223)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermt2w))]
pub fn _mm512_mask_permutex2var_epi16(
    a: __m512i,
    k: __mmask32,
    idx: __m512i,
    b: __m512i,
) -> __m512i {
    unsafe {
        let permute = _mm512_permutex2var_epi16(a, idx, b).as_i16x32();
        transmute(simd_select_bitmask(k, permute, a.as_i16x32()))
    }
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_permutex2var_epi16&expand=4225)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vperm))] //vpermi2w or vpermt2w
pub fn _mm512_maskz_permutex2var_epi16(
    k: __mmask32,
    a: __m512i,
    idx: __m512i,
    b: __m512i,
) -> __m512i {
    unsafe {
        let permute = _mm512_permutex2var_epi16(a, idx, b).as_i16x32();
        transmute(simd_select_bitmask(k, permute, i16x32::ZERO))
    }
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from idx when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask2_permutex2var_epi16&expand=4224)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermi2w))]
pub fn _mm512_mask2_permutex2var_epi16(
    a: __m512i,
    idx: __m512i,
    k: __mmask32,
    b: __m512i,
) -> __m512i {
    unsafe {
        let permute = _mm512_permutex2var_epi16(a, idx, b).as_i16x32();
        transmute(simd_select_bitmask(k, permute, idx.as_i16x32()))
    }
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_permutex2var_epi16&expand=4222)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vperm))] //vpermi2w or vpermt2w
pub fn _mm256_permutex2var_epi16(a: __m256i, idx: __m256i, b: __m256i) -> __m256i {
    unsafe { transmute(vpermi2w256(a.as_i16x16(), idx.as_i16x16(), b.as_i16x16())) }
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_permutex2var_epi16&expand=4219)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermt2w))]
pub fn _mm256_mask_permutex2var_epi16(
    a: __m256i,
    k: __mmask16,
    idx: __m256i,
    b: __m256i,
) -> __m256i {
    unsafe {
        let permute = _mm256_permutex2var_epi16(a, idx, b).as_i16x16();
        transmute(simd_select_bitmask(k, permute, a.as_i16x16()))
    }
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_permutex2var_epi16&expand=4221)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vperm))] //vpermi2w or vpermt2w
pub fn _mm256_maskz_permutex2var_epi16(
    k: __mmask16,
    a: __m256i,
    idx: __m256i,
    b: __m256i,
) -> __m256i {
    unsafe {
        let permute = _mm256_permutex2var_epi16(a, idx, b).as_i16x16();
        transmute(simd_select_bitmask(k, permute, i16x16::ZERO))
    }
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from idx when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask2_permutex2var_epi16&expand=4220)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermi2w))]
pub fn _mm256_mask2_permutex2var_epi16(
    a: __m256i,
    idx: __m256i,
    k: __mmask16,
    b: __m256i,
) -> __m256i {
    unsafe {
        let permute = _mm256_permutex2var_epi16(a, idx, b).as_i16x16();
        transmute(simd_select_bitmask(k, permute, idx.as_i16x16()))
    }
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_permutex2var_epi16&expand=4218)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vperm))] //vpermi2w or vpermt2w
pub fn _mm_permutex2var_epi16(a: __m128i, idx: __m128i, b: __m128i) -> __m128i {
    unsafe { transmute(vpermi2w128(a.as_i16x8(), idx.as_i16x8(), b.as_i16x8())) }
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_permutex2var_epi16&expand=4215)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermt2w))]
pub fn _mm_mask_permutex2var_epi16(a: __m128i, k: __mmask8, idx: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let permute = _mm_permutex2var_epi16(a, idx, b).as_i16x8();
        transmute(simd_select_bitmask(k, permute, a.as_i16x8()))
    }
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_permutex2var_epi16&expand=4217)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vperm))] //vpermi2w or vpermt2w
pub fn _mm_maskz_permutex2var_epi16(k: __mmask8, a: __m128i, idx: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let permute = _mm_permutex2var_epi16(a, idx, b).as_i16x8();
        transmute(simd_select_bitmask(k, permute, i16x8::ZERO))
    }
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from idx when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask2_permutex2var_epi16&expand=4216)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermi2w))]
pub fn _mm_mask2_permutex2var_epi16(a: __m128i, idx: __m128i, k: __mmask8, b: __m128i) -> __m128i {
    unsafe {
        let permute = _mm_permutex2var_epi16(a, idx, b).as_i16x8();
        transmute(simd_select_bitmask(k, permute, idx.as_i16x8()))
    }
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_permutexvar_epi16&expand=4295)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermw))]
pub fn _mm512_permutexvar_epi16(idx: __m512i, a: __m512i) -> __m512i {
    unsafe { transmute(vpermw(a.as_i16x32(), idx.as_i16x32())) }
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_permutexvar_epi16&expand=4293)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermw))]
pub fn _mm512_mask_permutexvar_epi16(
    src: __m512i,
    k: __mmask32,
    idx: __m512i,
    a: __m512i,
) -> __m512i {
    unsafe {
        let permute = _mm512_permutexvar_epi16(idx, a).as_i16x32();
        transmute(simd_select_bitmask(k, permute, src.as_i16x32()))
    }
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_permutexvar_epi16&expand=4294)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermw))]
pub fn _mm512_maskz_permutexvar_epi16(k: __mmask32, idx: __m512i, a: __m512i) -> __m512i {
    unsafe {
        let permute = _mm512_permutexvar_epi16(idx, a).as_i16x32();
        transmute(simd_select_bitmask(k, permute, i16x32::ZERO))
    }
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_permutexvar_epi16&expand=4292)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermw))]
pub fn _mm256_permutexvar_epi16(idx: __m256i, a: __m256i) -> __m256i {
    unsafe { transmute(vpermw256(a.as_i16x16(), idx.as_i16x16())) }
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_permutexvar_epi16&expand=4290)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermw))]
pub fn _mm256_mask_permutexvar_epi16(
    src: __m256i,
    k: __mmask16,
    idx: __m256i,
    a: __m256i,
) -> __m256i {
    unsafe {
        let permute = _mm256_permutexvar_epi16(idx, a).as_i16x16();
        transmute(simd_select_bitmask(k, permute, src.as_i16x16()))
    }
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_permutexvar_epi16&expand=4291)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermw))]
pub fn _mm256_maskz_permutexvar_epi16(k: __mmask16, idx: __m256i, a: __m256i) -> __m256i {
    unsafe {
        let permute = _mm256_permutexvar_epi16(idx, a).as_i16x16();
        transmute(simd_select_bitmask(k, permute, i16x16::ZERO))
    }
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_permutexvar_epi16&expand=4289)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermw))]
pub fn _mm_permutexvar_epi16(idx: __m128i, a: __m128i) -> __m128i {
    unsafe { transmute(vpermw128(a.as_i16x8(), idx.as_i16x8())) }
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_permutexvar_epi16&expand=4287)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermw))]
pub fn _mm_mask_permutexvar_epi16(src: __m128i, k: __mmask8, idx: __m128i, a: __m128i) -> __m128i {
    unsafe {
        let permute = _mm_permutexvar_epi16(idx, a).as_i16x8();
        transmute(simd_select_bitmask(k, permute, src.as_i16x8()))
    }
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_permutexvar_epi16&expand=4288)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpermw))]
pub fn _mm_maskz_permutexvar_epi16(k: __mmask8, idx: __m128i, a: __m128i) -> __m128i {
    unsafe {
        let permute = _mm_permutexvar_epi16(idx, a).as_i16x8();
        transmute(simd_select_bitmask(k, permute, i16x8::ZERO))
    }
}

/// Blend packed 16-bit integers from a and b using control mask k, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_blend_epi16&expand=430)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu16))] //should be vpblendmw
pub fn _mm512_mask_blend_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_select_bitmask(k, b.as_i16x32(), a.as_i16x32())) }
}

/// Blend packed 16-bit integers from a and b using control mask k, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_blend_epi16&expand=429)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu16))] //should be vpblendmw
pub fn _mm256_mask_blend_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe { transmute(simd_select_bitmask(k, b.as_i16x16(), a.as_i16x16())) }
}

/// Blend packed 16-bit integers from a and b using control mask k, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_blend_epi16&expand=427)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu16))] //should be vpblendmw
pub fn _mm_mask_blend_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe { transmute(simd_select_bitmask(k, b.as_i16x8(), a.as_i16x8())) }
}

/// Blend packed 8-bit integers from a and b using control mask k, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_blend_epi8&expand=441)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu8))] //should be vpblendmb
pub fn _mm512_mask_blend_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(simd_select_bitmask(k, b.as_i8x64(), a.as_i8x64())) }
}

/// Blend packed 8-bit integers from a and b using control mask k, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_blend_epi8&expand=440)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu8))] //should be vpblendmb
pub fn _mm256_mask_blend_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe { transmute(simd_select_bitmask(k, b.as_i8x32(), a.as_i8x32())) }
}

/// Blend packed 8-bit integers from a and b using control mask k, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_blend_epi8&expand=439)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu8))] //should be vpblendmb
pub fn _mm_mask_blend_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe { transmute(simd_select_bitmask(k, b.as_i8x16(), a.as_i8x16())) }
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcastw_epi16&expand=587)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub fn _mm512_broadcastw_epi16(a: __m128i) -> __m512i {
    unsafe {
        let a = _mm512_castsi128_si512(a).as_i16x32();
        let ret: i16x32 = simd_shuffle!(
            a,
            a,
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        );
        transmute(ret)
    }
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_broadcastw_epi16&expand=588)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub fn _mm512_mask_broadcastw_epi16(src: __m512i, k: __mmask32, a: __m128i) -> __m512i {
    unsafe {
        let broadcast = _mm512_broadcastw_epi16(a).as_i16x32();
        transmute(simd_select_bitmask(k, broadcast, src.as_i16x32()))
    }
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_broadcastw_epi16&expand=589)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub fn _mm512_maskz_broadcastw_epi16(k: __mmask32, a: __m128i) -> __m512i {
    unsafe {
        let broadcast = _mm512_broadcastw_epi16(a).as_i16x32();
        transmute(simd_select_bitmask(k, broadcast, i16x32::ZERO))
    }
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_broadcastw_epi16&expand=585)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub fn _mm256_mask_broadcastw_epi16(src: __m256i, k: __mmask16, a: __m128i) -> __m256i {
    unsafe {
        let broadcast = _mm256_broadcastw_epi16(a).as_i16x16();
        transmute(simd_select_bitmask(k, broadcast, src.as_i16x16()))
    }
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_broadcastw_epi16&expand=586)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub fn _mm256_maskz_broadcastw_epi16(k: __mmask16, a: __m128i) -> __m256i {
    unsafe {
        let broadcast = _mm256_broadcastw_epi16(a).as_i16x16();
        transmute(simd_select_bitmask(k, broadcast, i16x16::ZERO))
    }
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_broadcastw_epi16&expand=582)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub fn _mm_mask_broadcastw_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let broadcast = _mm_broadcastw_epi16(a).as_i16x8();
        transmute(simd_select_bitmask(k, broadcast, src.as_i16x8()))
    }
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_broadcastw_epi16&expand=583)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub fn _mm_maskz_broadcastw_epi16(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let broadcast = _mm_broadcastw_epi16(a).as_i16x8();
        transmute(simd_select_bitmask(k, broadcast, i16x8::ZERO))
    }
}

/// Broadcast the low packed 8-bit integer from a to all elements of dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_broadcastb_epi8&expand=536)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub fn _mm512_broadcastb_epi8(a: __m128i) -> __m512i {
    unsafe {
        let a = _mm512_castsi128_si512(a).as_i8x64();
        let ret: i8x64 = simd_shuffle!(
            a,
            a,
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
        );
        transmute(ret)
    }
}

/// Broadcast the low packed 8-bit integer from a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_broadcastb_epi8&expand=537)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub fn _mm512_mask_broadcastb_epi8(src: __m512i, k: __mmask64, a: __m128i) -> __m512i {
    unsafe {
        let broadcast = _mm512_broadcastb_epi8(a).as_i8x64();
        transmute(simd_select_bitmask(k, broadcast, src.as_i8x64()))
    }
}

/// Broadcast the low packed 8-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_broadcastb_epi8&expand=538)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub fn _mm512_maskz_broadcastb_epi8(k: __mmask64, a: __m128i) -> __m512i {
    unsafe {
        let broadcast = _mm512_broadcastb_epi8(a).as_i8x64();
        transmute(simd_select_bitmask(k, broadcast, i8x64::ZERO))
    }
}

/// Broadcast the low packed 8-bit integer from a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_broadcastb_epi8&expand=534)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub fn _mm256_mask_broadcastb_epi8(src: __m256i, k: __mmask32, a: __m128i) -> __m256i {
    unsafe {
        let broadcast = _mm256_broadcastb_epi8(a).as_i8x32();
        transmute(simd_select_bitmask(k, broadcast, src.as_i8x32()))
    }
}

/// Broadcast the low packed 8-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_broadcastb_epi8&expand=535)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub fn _mm256_maskz_broadcastb_epi8(k: __mmask32, a: __m128i) -> __m256i {
    unsafe {
        let broadcast = _mm256_broadcastb_epi8(a).as_i8x32();
        transmute(simd_select_bitmask(k, broadcast, i8x32::ZERO))
    }
}

/// Broadcast the low packed 8-bit integer from a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_broadcastb_epi8&expand=531)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub fn _mm_mask_broadcastb_epi8(src: __m128i, k: __mmask16, a: __m128i) -> __m128i {
    unsafe {
        let broadcast = _mm_broadcastb_epi8(a).as_i8x16();
        transmute(simd_select_bitmask(k, broadcast, src.as_i8x16()))
    }
}

/// Broadcast the low packed 8-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_broadcastb_epi8&expand=532)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub fn _mm_maskz_broadcastb_epi8(k: __mmask16, a: __m128i) -> __m128i {
    unsafe {
        let broadcast = _mm_broadcastb_epi8(a).as_i8x16();
        transmute(simd_select_bitmask(k, broadcast, i8x16::ZERO))
    }
}

/// Unpack and interleave 16-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_unpackhi_epi16&expand=6012)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub fn _mm512_unpackhi_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = a.as_i16x32();
        let b = b.as_i16x32();
        #[rustfmt::skip]
        let r: i16x32 = simd_shuffle!(
            a,
            b,
            [
                4, 32 + 4, 5, 32 + 5,
                6, 32 + 6, 7, 32 + 7,
                12, 32 + 12, 13, 32 + 13,
                14, 32 + 14, 15, 32 + 15,
                20, 32 + 20, 21, 32 + 21,
                22, 32 + 22, 23, 32 + 23,
                28, 32 + 28, 29, 32 + 29,
                30, 32 + 30, 31, 32 + 31,
            ],
        );
        transmute(r)
    }
}

/// Unpack and interleave 16-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_unpackhi_epi16&expand=6010)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub fn _mm512_mask_unpackhi_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let unpackhi = _mm512_unpackhi_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, unpackhi, src.as_i16x32()))
    }
}

/// Unpack and interleave 16-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_unpackhi_epi16&expand=6011)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub fn _mm512_maskz_unpackhi_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let unpackhi = _mm512_unpackhi_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, unpackhi, i16x32::ZERO))
    }
}

/// Unpack and interleave 16-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_unpackhi_epi16&expand=6007)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub fn _mm256_mask_unpackhi_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let unpackhi = _mm256_unpackhi_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, unpackhi, src.as_i16x16()))
    }
}

/// Unpack and interleave 16-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_unpackhi_epi16&expand=6008)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub fn _mm256_maskz_unpackhi_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let unpackhi = _mm256_unpackhi_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, unpackhi, i16x16::ZERO))
    }
}

/// Unpack and interleave 16-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_unpackhi_epi16&expand=6004)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub fn _mm_mask_unpackhi_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let unpackhi = _mm_unpackhi_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, unpackhi, src.as_i16x8()))
    }
}

/// Unpack and interleave 16-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_unpackhi_epi16&expand=6005)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub fn _mm_maskz_unpackhi_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let unpackhi = _mm_unpackhi_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, unpackhi, i16x8::ZERO))
    }
}

/// Unpack and interleave 8-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_unpackhi_epi8&expand=6039)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub fn _mm512_unpackhi_epi8(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = a.as_i8x64();
        let b = b.as_i8x64();
        #[rustfmt::skip]
        let r: i8x64 = simd_shuffle!(
            a,
            b,
            [
                8, 64 + 8, 9, 64 + 9,
                10, 64 + 10, 11, 64 + 11,
                12, 64 + 12, 13, 64 + 13,
                14, 64 + 14, 15, 64 + 15,
                24, 64 + 24, 25, 64 + 25,
                26, 64 + 26, 27, 64 + 27,
                28, 64 + 28, 29, 64 + 29,
                30, 64 + 30, 31, 64 + 31,
                40, 64 + 40, 41, 64 + 41,
                42, 64 + 42, 43, 64 + 43,
                44, 64 + 44, 45, 64 + 45,
                46, 64 + 46, 47, 64 + 47,
                56, 64 + 56, 57, 64 + 57,
                58, 64 + 58, 59, 64 + 59,
                60, 64 + 60, 61, 64 + 61,
                62, 64 + 62, 63, 64 + 63,
            ],
        );
        transmute(r)
    }
}

/// Unpack and interleave 8-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_unpackhi_epi8&expand=6037)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub fn _mm512_mask_unpackhi_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let unpackhi = _mm512_unpackhi_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, unpackhi, src.as_i8x64()))
    }
}

/// Unpack and interleave 8-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_unpackhi_epi8&expand=6038)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub fn _mm512_maskz_unpackhi_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let unpackhi = _mm512_unpackhi_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, unpackhi, i8x64::ZERO))
    }
}

/// Unpack and interleave 8-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_unpackhi_epi8&expand=6034)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub fn _mm256_mask_unpackhi_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let unpackhi = _mm256_unpackhi_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, unpackhi, src.as_i8x32()))
    }
}

/// Unpack and interleave 8-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_unpackhi_epi8&expand=6035)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub fn _mm256_maskz_unpackhi_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let unpackhi = _mm256_unpackhi_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, unpackhi, i8x32::ZERO))
    }
}

/// Unpack and interleave 8-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_unpackhi_epi8&expand=6031)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub fn _mm_mask_unpackhi_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let unpackhi = _mm_unpackhi_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, unpackhi, src.as_i8x16()))
    }
}

/// Unpack and interleave 8-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_unpackhi_epi8&expand=6032)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub fn _mm_maskz_unpackhi_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let unpackhi = _mm_unpackhi_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, unpackhi, i8x16::ZERO))
    }
}

/// Unpack and interleave 16-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_unpacklo_epi16&expand=6069)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub fn _mm512_unpacklo_epi16(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = a.as_i16x32();
        let b = b.as_i16x32();
        #[rustfmt::skip]
        let r: i16x32 = simd_shuffle!(
            a,
            b,
            [
               0,  32+0,   1, 32+1,
               2,  32+2,   3, 32+3,
               8,  32+8,   9, 32+9,
               10, 32+10, 11, 32+11,
               16, 32+16, 17, 32+17,
               18, 32+18, 19, 32+19,
               24, 32+24, 25, 32+25,
               26, 32+26, 27, 32+27
            ],
        );
        transmute(r)
    }
}

/// Unpack and interleave 16-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_unpacklo_epi16&expand=6067)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub fn _mm512_mask_unpacklo_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let unpacklo = _mm512_unpacklo_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, unpacklo, src.as_i16x32()))
    }
}

/// Unpack and interleave 16-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_unpacklo_epi16&expand=6068)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub fn _mm512_maskz_unpacklo_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let unpacklo = _mm512_unpacklo_epi16(a, b).as_i16x32();
        transmute(simd_select_bitmask(k, unpacklo, i16x32::ZERO))
    }
}

/// Unpack and interleave 16-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_unpacklo_epi16&expand=6064)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub fn _mm256_mask_unpacklo_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let unpacklo = _mm256_unpacklo_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, unpacklo, src.as_i16x16()))
    }
}

/// Unpack and interleave 16-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_unpacklo_epi16&expand=6065)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub fn _mm256_maskz_unpacklo_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let unpacklo = _mm256_unpacklo_epi16(a, b).as_i16x16();
        transmute(simd_select_bitmask(k, unpacklo, i16x16::ZERO))
    }
}

/// Unpack and interleave 16-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_unpacklo_epi16&expand=6061)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub fn _mm_mask_unpacklo_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let unpacklo = _mm_unpacklo_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, unpacklo, src.as_i16x8()))
    }
}

/// Unpack and interleave 16-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_unpacklo_epi16&expand=6062)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub fn _mm_maskz_unpacklo_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let unpacklo = _mm_unpacklo_epi16(a, b).as_i16x8();
        transmute(simd_select_bitmask(k, unpacklo, i16x8::ZERO))
    }
}

/// Unpack and interleave 8-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_unpacklo_epi8&expand=6096)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub fn _mm512_unpacklo_epi8(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let a = a.as_i8x64();
        let b = b.as_i8x64();
        #[rustfmt::skip]
        let r: i8x64 = simd_shuffle!(
            a,
            b,
            [
                0,  64+0,   1, 64+1,
                2,  64+2,   3, 64+3,
                4,  64+4,   5, 64+5,
                6,  64+6,   7, 64+7,
                16, 64+16, 17, 64+17,
                18, 64+18, 19, 64+19,
                20, 64+20, 21, 64+21,
                22, 64+22, 23, 64+23,
                32, 64+32, 33, 64+33,
                34, 64+34, 35, 64+35,
                36, 64+36, 37, 64+37,
                38, 64+38, 39, 64+39,
                48, 64+48, 49, 64+49,
                50, 64+50, 51, 64+51,
                52, 64+52, 53, 64+53,
                54, 64+54, 55, 64+55,
            ],
        );
        transmute(r)
    }
}

/// Unpack and interleave 8-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_unpacklo_epi8&expand=6094)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub fn _mm512_mask_unpacklo_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let unpacklo = _mm512_unpacklo_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, unpacklo, src.as_i8x64()))
    }
}

/// Unpack and interleave 8-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_unpacklo_epi8&expand=6095)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub fn _mm512_maskz_unpacklo_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let unpacklo = _mm512_unpacklo_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, unpacklo, i8x64::ZERO))
    }
}

/// Unpack and interleave 8-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_unpacklo_epi8&expand=6091)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub fn _mm256_mask_unpacklo_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let unpacklo = _mm256_unpacklo_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, unpacklo, src.as_i8x32()))
    }
}

/// Unpack and interleave 8-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_unpacklo_epi8&expand=6092)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub fn _mm256_maskz_unpacklo_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let unpacklo = _mm256_unpacklo_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, unpacklo, i8x32::ZERO))
    }
}

/// Unpack and interleave 8-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_unpacklo_epi8&expand=6088)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub fn _mm_mask_unpacklo_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let unpacklo = _mm_unpacklo_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, unpacklo, src.as_i8x16()))
    }
}

/// Unpack and interleave 8-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_unpacklo_epi8&expand=6089)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub fn _mm_maskz_unpacklo_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let unpacklo = _mm_unpacklo_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, unpacklo, i8x16::ZERO))
    }
}

/// Move packed 16-bit integers from a into dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_mov_epi16&expand=3795)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
pub fn _mm512_mask_mov_epi16(src: __m512i, k: __mmask32, a: __m512i) -> __m512i {
    unsafe {
        let mov = a.as_i16x32();
        transmute(simd_select_bitmask(k, mov, src.as_i16x32()))
    }
}

/// Move packed 16-bit integers from a into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_mov_epi16&expand=3796)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
pub fn _mm512_maskz_mov_epi16(k: __mmask32, a: __m512i) -> __m512i {
    unsafe {
        let mov = a.as_i16x32();
        transmute(simd_select_bitmask(k, mov, i16x32::ZERO))
    }
}

/// Move packed 16-bit integers from a into dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_mov_epi16&expand=3793)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
pub fn _mm256_mask_mov_epi16(src: __m256i, k: __mmask16, a: __m256i) -> __m256i {
    unsafe {
        let mov = a.as_i16x16();
        transmute(simd_select_bitmask(k, mov, src.as_i16x16()))
    }
}

/// Move packed 16-bit integers from a into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_mov_epi16&expand=3794)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
pub fn _mm256_maskz_mov_epi16(k: __mmask16, a: __m256i) -> __m256i {
    unsafe {
        let mov = a.as_i16x16();
        transmute(simd_select_bitmask(k, mov, i16x16::ZERO))
    }
}

/// Move packed 16-bit integers from a into dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_mov_epi16&expand=3791)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
pub fn _mm_mask_mov_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let mov = a.as_i16x8();
        transmute(simd_select_bitmask(k, mov, src.as_i16x8()))
    }
}

/// Move packed 16-bit integers from a into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_mov_epi16&expand=3792)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
pub fn _mm_maskz_mov_epi16(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let mov = a.as_i16x8();
        transmute(simd_select_bitmask(k, mov, i16x8::ZERO))
    }
}

/// Move packed 8-bit integers from a into dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_mov_epi8&expand=3813)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
pub fn _mm512_mask_mov_epi8(src: __m512i, k: __mmask64, a: __m512i) -> __m512i {
    unsafe {
        let mov = a.as_i8x64();
        transmute(simd_select_bitmask(k, mov, src.as_i8x64()))
    }
}

/// Move packed 8-bit integers from a into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_mov_epi8&expand=3814)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
pub fn _mm512_maskz_mov_epi8(k: __mmask64, a: __m512i) -> __m512i {
    unsafe {
        let mov = a.as_i8x64();
        transmute(simd_select_bitmask(k, mov, i8x64::ZERO))
    }
}

/// Move packed 8-bit integers from a into dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_mov_epi8&expand=3811)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
pub fn _mm256_mask_mov_epi8(src: __m256i, k: __mmask32, a: __m256i) -> __m256i {
    unsafe {
        let mov = a.as_i8x32();
        transmute(simd_select_bitmask(k, mov, src.as_i8x32()))
    }
}

/// Move packed 8-bit integers from a into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_mov_epi8&expand=3812)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
pub fn _mm256_maskz_mov_epi8(k: __mmask32, a: __m256i) -> __m256i {
    unsafe {
        let mov = a.as_i8x32();
        transmute(simd_select_bitmask(k, mov, i8x32::ZERO))
    }
}

/// Move packed 8-bit integers from a into dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_mov_epi8&expand=3809)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
pub fn _mm_mask_mov_epi8(src: __m128i, k: __mmask16, a: __m128i) -> __m128i {
    unsafe {
        let mov = a.as_i8x16();
        transmute(simd_select_bitmask(k, mov, src.as_i8x16()))
    }
}

/// Move packed 8-bit integers from a into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_mov_epi8&expand=3810)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
pub fn _mm_maskz_mov_epi8(k: __mmask16, a: __m128i) -> __m128i {
    unsafe {
        let mov = a.as_i8x16();
        transmute(simd_select_bitmask(k, mov, i8x16::ZERO))
    }
}

/// Broadcast 16-bit integer a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_set1_epi16&expand=4942)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub fn _mm512_mask_set1_epi16(src: __m512i, k: __mmask32, a: i16) -> __m512i {
    unsafe {
        let r = _mm512_set1_epi16(a).as_i16x32();
        transmute(simd_select_bitmask(k, r, src.as_i16x32()))
    }
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_set1_epi16&expand=4943)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub fn _mm512_maskz_set1_epi16(k: __mmask32, a: i16) -> __m512i {
    unsafe {
        let r = _mm512_set1_epi16(a).as_i16x32();
        transmute(simd_select_bitmask(k, r, i16x32::ZERO))
    }
}

/// Broadcast 16-bit integer a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_set1_epi16&expand=4939)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub fn _mm256_mask_set1_epi16(src: __m256i, k: __mmask16, a: i16) -> __m256i {
    unsafe {
        let r = _mm256_set1_epi16(a).as_i16x16();
        transmute(simd_select_bitmask(k, r, src.as_i16x16()))
    }
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_set1_epi16&expand=4940)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub fn _mm256_maskz_set1_epi16(k: __mmask16, a: i16) -> __m256i {
    unsafe {
        let r = _mm256_set1_epi16(a).as_i16x16();
        transmute(simd_select_bitmask(k, r, i16x16::ZERO))
    }
}

/// Broadcast 16-bit integer a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_set1_epi16&expand=4936)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub fn _mm_mask_set1_epi16(src: __m128i, k: __mmask8, a: i16) -> __m128i {
    unsafe {
        let r = _mm_set1_epi16(a).as_i16x8();
        transmute(simd_select_bitmask(k, r, src.as_i16x8()))
    }
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_set1_epi16&expand=4937)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub fn _mm_maskz_set1_epi16(k: __mmask8, a: i16) -> __m128i {
    unsafe {
        let r = _mm_set1_epi16(a).as_i16x8();
        transmute(simd_select_bitmask(k, r, i16x8::ZERO))
    }
}

/// Broadcast 8-bit integer a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_set1_epi8&expand=4970)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcast))]
pub fn _mm512_mask_set1_epi8(src: __m512i, k: __mmask64, a: i8) -> __m512i {
    unsafe {
        let r = _mm512_set1_epi8(a).as_i8x64();
        transmute(simd_select_bitmask(k, r, src.as_i8x64()))
    }
}

/// Broadcast 8-bit integer a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_set1_epi8&expand=4971)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcast))]
pub fn _mm512_maskz_set1_epi8(k: __mmask64, a: i8) -> __m512i {
    unsafe {
        let r = _mm512_set1_epi8(a).as_i8x64();
        transmute(simd_select_bitmask(k, r, i8x64::ZERO))
    }
}

/// Broadcast 8-bit integer a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_set1_epi8&expand=4967)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcast))]
pub fn _mm256_mask_set1_epi8(src: __m256i, k: __mmask32, a: i8) -> __m256i {
    unsafe {
        let r = _mm256_set1_epi8(a).as_i8x32();
        transmute(simd_select_bitmask(k, r, src.as_i8x32()))
    }
}

/// Broadcast 8-bit integer a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_set1_epi8&expand=4968)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcast))]
pub fn _mm256_maskz_set1_epi8(k: __mmask32, a: i8) -> __m256i {
    unsafe {
        let r = _mm256_set1_epi8(a).as_i8x32();
        transmute(simd_select_bitmask(k, r, i8x32::ZERO))
    }
}

/// Broadcast 8-bit integer a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_set1_epi8&expand=4964)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcast))]
pub fn _mm_mask_set1_epi8(src: __m128i, k: __mmask16, a: i8) -> __m128i {
    unsafe {
        let r = _mm_set1_epi8(a).as_i8x16();
        transmute(simd_select_bitmask(k, r, src.as_i8x16()))
    }
}

/// Broadcast 8-bit integer a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_set1_epi8&expand=4965)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpbroadcast))]
pub fn _mm_maskz_set1_epi8(k: __mmask16, a: i8) -> __m128i {
    unsafe {
        let r = _mm_set1_epi8(a).as_i8x16();
        transmute(simd_select_bitmask(k, r, i8x16::ZERO))
    }
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the low 64 bits of 128-bit lanes of dst, with the high 64 bits of 128-bit lanes being copied from a to dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shufflelo_epi16&expand=5221)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshuflw, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
pub fn _mm512_shufflelo_epi16<const IMM8: i32>(a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let a = a.as_i16x32();
        let r: i16x32 = simd_shuffle!(
            a,
            a,
            [
                IMM8 as u32 & 0b11,
                (IMM8 as u32 >> 2) & 0b11,
                (IMM8 as u32 >> 4) & 0b11,
                (IMM8 as u32 >> 6) & 0b11,
                4,
                5,
                6,
                7,
                (IMM8 as u32 & 0b11) + 8,
                ((IMM8 as u32 >> 2) & 0b11) + 8,
                ((IMM8 as u32 >> 4) & 0b11) + 8,
                ((IMM8 as u32 >> 6) & 0b11) + 8,
                12,
                13,
                14,
                15,
                (IMM8 as u32 & 0b11) + 16,
                ((IMM8 as u32 >> 2) & 0b11) + 16,
                ((IMM8 as u32 >> 4) & 0b11) + 16,
                ((IMM8 as u32 >> 6) & 0b11) + 16,
                20,
                21,
                22,
                23,
                (IMM8 as u32 & 0b11) + 24,
                ((IMM8 as u32 >> 2) & 0b11) + 24,
                ((IMM8 as u32 >> 4) & 0b11) + 24,
                ((IMM8 as u32 >> 6) & 0b11) + 24,
                28,
                29,
                30,
                31,
            ],
        );
        transmute(r)
    }
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the low 64 bits of 128-bit lanes of dst, with the high 64 bits of 128-bit lanes being copied from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shufflelo_epi16&expand=5219)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshuflw, IMM8 = 0))]
#[rustc_legacy_const_generics(3)]
pub fn _mm512_mask_shufflelo_epi16<const IMM8: i32>(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let r = _mm512_shufflelo_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, r.as_i16x32(), src.as_i16x32()))
    }
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the low 64 bits of 128-bit lanes of dst, with the high 64 bits of 128-bit lanes being copied from a to dst, using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shufflelo_epi16&expand=5220)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshuflw, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
pub fn _mm512_maskz_shufflelo_epi16<const IMM8: i32>(k: __mmask32, a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let r = _mm512_shufflelo_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, r.as_i16x32(), i16x32::ZERO))
    }
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the low 64 bits of 128-bit lanes of dst, with the high 64 bits of 128-bit lanes being copied from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shufflelo_epi16&expand=5216)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshuflw, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm256_mask_shufflelo_epi16<const IMM8: i32>(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shuffle = _mm256_shufflelo_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, shuffle.as_i16x16(), src.as_i16x16()))
    }
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the low 64 bits of 128-bit lanes of dst, with the high 64 bits of 128-bit lanes being copied from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shufflelo_epi16&expand=5217)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshuflw, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm256_maskz_shufflelo_epi16<const IMM8: i32>(k: __mmask16, a: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shuffle = _mm256_shufflelo_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, shuffle.as_i16x16(), i16x16::ZERO))
    }
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the low 64 bits of 128-bit lanes of dst, with the high 64 bits of 128-bit lanes being copied from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shufflelo_epi16&expand=5213)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshuflw, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm_mask_shufflelo_epi16<const IMM8: i32>(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shuffle = _mm_shufflelo_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, shuffle.as_i16x8(), src.as_i16x8()))
    }
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the low 64 bits of 128-bit lanes of dst, with the high 64 bits of 128-bit lanes being copied from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shufflelo_epi16&expand=5214)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshuflw, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm_maskz_shufflelo_epi16<const IMM8: i32>(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shuffle = _mm_shufflelo_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, shuffle.as_i16x8(), i16x8::ZERO))
    }
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the high 64 bits of 128-bit lanes of dst, with the low 64 bits of 128-bit lanes being copied from a to dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shufflehi_epi16&expand=5212)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufhw, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
pub fn _mm512_shufflehi_epi16<const IMM8: i32>(a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let a = a.as_i16x32();
        let r: i16x32 = simd_shuffle!(
            a,
            a,
            [
                0,
                1,
                2,
                3,
                (IMM8 as u32 & 0b11) + 4,
                ((IMM8 as u32 >> 2) & 0b11) + 4,
                ((IMM8 as u32 >> 4) & 0b11) + 4,
                ((IMM8 as u32 >> 6) & 0b11) + 4,
                8,
                9,
                10,
                11,
                (IMM8 as u32 & 0b11) + 12,
                ((IMM8 as u32 >> 2) & 0b11) + 12,
                ((IMM8 as u32 >> 4) & 0b11) + 12,
                ((IMM8 as u32 >> 6) & 0b11) + 12,
                16,
                17,
                18,
                19,
                (IMM8 as u32 & 0b11) + 20,
                ((IMM8 as u32 >> 2) & 0b11) + 20,
                ((IMM8 as u32 >> 4) & 0b11) + 20,
                ((IMM8 as u32 >> 6) & 0b11) + 20,
                24,
                25,
                26,
                27,
                (IMM8 as u32 & 0b11) + 28,
                ((IMM8 as u32 >> 2) & 0b11) + 28,
                ((IMM8 as u32 >> 4) & 0b11) + 28,
                ((IMM8 as u32 >> 6) & 0b11) + 28,
            ],
        );
        transmute(r)
    }
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the high 64 bits of 128-bit lanes of dst, with the low 64 bits of 128-bit lanes being copied from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shufflehi_epi16&expand=5210)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufhw, IMM8 = 0))]
#[rustc_legacy_const_generics(3)]
pub fn _mm512_mask_shufflehi_epi16<const IMM8: i32>(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let r = _mm512_shufflehi_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, r.as_i16x32(), src.as_i16x32()))
    }
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the high 64 bits of 128-bit lanes of dst, with the low 64 bits of 128-bit lanes being copied from a to dst, using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shufflehi_epi16&expand=5211)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufhw, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
pub fn _mm512_maskz_shufflehi_epi16<const IMM8: i32>(k: __mmask32, a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let r = _mm512_shufflehi_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, r.as_i16x32(), i16x32::ZERO))
    }
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the high 64 bits of 128-bit lanes of dst, with the low 64 bits of 128-bit lanes being copied from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shufflehi_epi16&expand=5207)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufhw, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm256_mask_shufflehi_epi16<const IMM8: i32>(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shuffle = _mm256_shufflehi_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, shuffle.as_i16x16(), src.as_i16x16()))
    }
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the high 64 bits of 128-bit lanes of dst, with the low 64 bits of 128-bit lanes being copied from a to dst, using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shufflehi_epi16&expand=5208)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufhw, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm256_maskz_shufflehi_epi16<const IMM8: i32>(k: __mmask16, a: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shuffle = _mm256_shufflehi_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, shuffle.as_i16x16(), i16x16::ZERO))
    }
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the high 64 bits of 128-bit lanes of dst, with the low 64 bits of 128-bit lanes being copied from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shufflehi_epi16&expand=5204)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufhw, IMM8 = 5))]
#[rustc_legacy_const_generics(3)]
pub fn _mm_mask_shufflehi_epi16<const IMM8: i32>(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shuffle = _mm_shufflehi_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, shuffle.as_i16x8(), src.as_i16x8()))
    }
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the high 64 bits of 128-bit lanes of dst, with the low 64 bits of 128-bit lanes being copied from a to dst, using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shufflehi_epi16&expand=5205)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufhw, IMM8 = 5))]
#[rustc_legacy_const_generics(2)]
pub fn _mm_maskz_shufflehi_epi16<const IMM8: i32>(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let shuffle = _mm_shufflehi_epi16::<IMM8>(a);
        transmute(simd_select_bitmask(k, shuffle.as_i16x8(), i16x8::ZERO))
    }
}

/// Shuffle packed 8-bit integers in a according to shuffle control mask in the corresponding 8-bit element of b, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shuffle_epi8&expand=5159)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub fn _mm512_shuffle_epi8(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(vpshufb(a.as_i8x64(), b.as_i8x64())) }
}

/// Shuffle 8-bit integers in a within 128-bit lanes using the control in the corresponding 8-bit element of b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_shuffle_epi8&expand=5157)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub fn _mm512_mask_shuffle_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let shuffle = _mm512_shuffle_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, shuffle, src.as_i8x64()))
    }
}

/// Shuffle packed 8-bit integers in a according to shuffle control mask in the corresponding 8-bit element of b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_shuffle_epi8&expand=5158)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub fn _mm512_maskz_shuffle_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let shuffle = _mm512_shuffle_epi8(a, b).as_i8x64();
        transmute(simd_select_bitmask(k, shuffle, i8x64::ZERO))
    }
}

/// Shuffle 8-bit integers in a within 128-bit lanes using the control in the corresponding 8-bit element of b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_shuffle_epi8&expand=5154)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub fn _mm256_mask_shuffle_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let shuffle = _mm256_shuffle_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, shuffle, src.as_i8x32()))
    }
}

/// Shuffle packed 8-bit integers in a according to shuffle control mask in the corresponding 8-bit element of b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_shuffle_epi8&expand=5155)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub fn _mm256_maskz_shuffle_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let shuffle = _mm256_shuffle_epi8(a, b).as_i8x32();
        transmute(simd_select_bitmask(k, shuffle, i8x32::ZERO))
    }
}

/// Shuffle 8-bit integers in a within 128-bit lanes using the control in the corresponding 8-bit element of b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_shuffle_epi8&expand=5151)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub fn _mm_mask_shuffle_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let shuffle = _mm_shuffle_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, shuffle, src.as_i8x16()))
    }
}

/// Shuffle packed 8-bit integers in a according to shuffle control mask in the corresponding 8-bit element of b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_shuffle_epi8&expand=5152)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub fn _mm_maskz_shuffle_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let shuffle = _mm_shuffle_epi8(a, b).as_i8x16();
        transmute(simd_select_bitmask(k, shuffle, i8x16::ZERO))
    }
}

/// Compute the bitwise AND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_test_epi16_mask&expand=5884)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestmw))]
pub fn _mm512_test_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_cmpneq_epi16_mask(and, zero)
}

/// Compute the bitwise AND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is non-zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_test_epi16_mask&expand=5883)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestmw))]
pub fn _mm512_mask_test_epi16_mask(k: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_mask_cmpneq_epi16_mask(k, and, zero)
}

/// Compute the bitwise AND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_test_epi16_mask&expand=5882)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestmw))]
pub fn _mm256_test_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_cmpneq_epi16_mask(and, zero)
}

/// Compute the bitwise AND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is non-zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_test_epi16_mask&expand=5881)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestmw))]
pub fn _mm256_mask_test_epi16_mask(k: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_mask_cmpneq_epi16_mask(k, and, zero)
}

/// Compute the bitwise AND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_test_epi16_mask&expand=5880)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestmw))]
pub fn _mm_test_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_cmpneq_epi16_mask(and, zero)
}

/// Compute the bitwise AND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is non-zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_test_epi16_mask&expand=5879)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestmw))]
pub fn _mm_mask_test_epi16_mask(k: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_mask_cmpneq_epi16_mask(k, and, zero)
}

/// Compute the bitwise AND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_test_epi8_mask&expand=5902)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestmb))]
pub fn _mm512_test_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_cmpneq_epi8_mask(and, zero)
}

/// Compute the bitwise AND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is non-zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_test_epi8_mask&expand=5901)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestmb))]
pub fn _mm512_mask_test_epi8_mask(k: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_mask_cmpneq_epi8_mask(k, and, zero)
}

/// Compute the bitwise AND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_test_epi8_mask&expand=5900)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestmb))]
pub fn _mm256_test_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_cmpneq_epi8_mask(and, zero)
}

/// Compute the bitwise AND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is non-zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_test_epi8_mask&expand=5899)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestmb))]
pub fn _mm256_mask_test_epi8_mask(k: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_mask_cmpneq_epi8_mask(k, and, zero)
}

/// Compute the bitwise AND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_test_epi8_mask&expand=5898)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestmb))]
pub fn _mm_test_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_cmpneq_epi8_mask(and, zero)
}

/// Compute the bitwise AND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is non-zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_test_epi8_mask&expand=5897)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestmb))]
pub fn _mm_mask_test_epi8_mask(k: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_mask_cmpneq_epi8_mask(k, and, zero)
}

/// Compute the bitwise NAND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k if the intermediate value is zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_testn_epi16_mask&expand=5915)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestnmw))]
pub fn _mm512_testn_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_cmpeq_epi16_mask(and, zero)
}

/// Compute the bitwise NAND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_testn_epi16_mask&expand=5914)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestnmw))]
pub fn _mm512_mask_testn_epi16_mask(k: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_mask_cmpeq_epi16_mask(k, and, zero)
}

/// Compute the bitwise NAND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k if the intermediate value is zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_testn_epi16_mask&expand=5913)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestnmw))]
pub fn _mm256_testn_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_cmpeq_epi16_mask(and, zero)
}

/// Compute the bitwise NAND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_testn_epi16_mask&expand=5912)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestnmw))]
pub fn _mm256_mask_testn_epi16_mask(k: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_mask_cmpeq_epi16_mask(k, and, zero)
}

/// Compute the bitwise NAND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k if the intermediate value is zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_testn_epi16_mask&expand=5911)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestnmw))]
pub fn _mm_testn_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_cmpeq_epi16_mask(and, zero)
}

/// Compute the bitwise NAND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_testn_epi16_mask&expand=5910)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestnmw))]
pub fn _mm_mask_testn_epi16_mask(k: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_mask_cmpeq_epi16_mask(k, and, zero)
}

/// Compute the bitwise NAND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k if the intermediate value is zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_testn_epi8_mask&expand=5933)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestnmb))]
pub fn _mm512_testn_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_cmpeq_epi8_mask(and, zero)
}

/// Compute the bitwise NAND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_testn_epi8_mask&expand=5932)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestnmb))]
pub fn _mm512_mask_testn_epi8_mask(k: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_mask_cmpeq_epi8_mask(k, and, zero)
}

/// Compute the bitwise NAND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k if the intermediate value is zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_testn_epi8_mask&expand=5931)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestnmb))]
pub fn _mm256_testn_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_cmpeq_epi8_mask(and, zero)
}

/// Compute the bitwise NAND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_testn_epi8_mask&expand=5930)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestnmb))]
pub fn _mm256_mask_testn_epi8_mask(k: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_mask_cmpeq_epi8_mask(k, and, zero)
}

/// Compute the bitwise NAND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k if the intermediate value is zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_testn_epi8_mask&expand=5929)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestnmb))]
pub fn _mm_testn_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_cmpeq_epi8_mask(and, zero)
}

/// Compute the bitwise NAND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is zero.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_testn_epi8_mask&expand=5928)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vptestnmb))]
pub fn _mm_mask_testn_epi8_mask(k: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_mask_cmpeq_epi8_mask(k, and, zero)
}

/// Store 64-bit mask from a into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_store_mask64&expand=5578)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(mov))] //should be kmovq
pub unsafe fn _store_mask64(mem_addr: *mut __mmask64, a: __mmask64) {
    ptr::write(mem_addr as *mut __mmask64, a);
}

/// Store 32-bit mask from a into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_store_mask32&expand=5577)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(mov))] //should be kmovd
pub unsafe fn _store_mask32(mem_addr: *mut __mmask32, a: __mmask32) {
    ptr::write(mem_addr as *mut __mmask32, a);
}

/// Load 64-bit mask from memory into k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_load_mask64&expand=3318)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(mov))] //should be kmovq
pub unsafe fn _load_mask64(mem_addr: *const __mmask64) -> __mmask64 {
    ptr::read(mem_addr as *const __mmask64)
}

/// Load 32-bit mask from memory into k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_load_mask32&expand=3317)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(mov))] //should be kmovd
pub unsafe fn _load_mask32(mem_addr: *const __mmask32) -> __mmask32 {
    ptr::read(mem_addr as *const __mmask32)
}

/// Compute the absolute differences of packed unsigned 8-bit integers in a and b, then horizontally sum each consecutive 8 differences to produce eight unsigned 16-bit integers, and pack these unsigned 16-bit integers in the low 16 bits of 64-bit elements in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_sad_epu8&expand=4855)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsadbw))]
pub fn _mm512_sad_epu8(a: __m512i, b: __m512i) -> __m512i {
    unsafe { transmute(vpsadbw(a.as_u8x64(), b.as_u8x64())) }
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst. Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_dbsad_epu8&expand=2114)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vdbpsadbw, IMM8 = 0))]
pub fn _mm512_dbsad_epu8<const IMM8: i32>(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let a = a.as_u8x64();
        let b = b.as_u8x64();
        let r = vdbpsadbw(a, b, IMM8);
        transmute(r)
    }
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_dbsad_epu8&expand=2115)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(4)]
#[cfg_attr(test, assert_instr(vdbpsadbw, IMM8 = 0))]
pub fn _mm512_mask_dbsad_epu8<const IMM8: i32>(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let a = a.as_u8x64();
        let b = b.as_u8x64();
        let r = vdbpsadbw(a, b, IMM8);
        transmute(simd_select_bitmask(k, r, src.as_u16x32()))
    }
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_dbsad_epu8&expand=2116)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vdbpsadbw, IMM8 = 0))]
pub fn _mm512_maskz_dbsad_epu8<const IMM8: i32>(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let a = a.as_u8x64();
        let b = b.as_u8x64();
        let r = vdbpsadbw(a, b, IMM8);
        transmute(simd_select_bitmask(k, r, u16x32::ZERO))
    }
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst. Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_dbsad_epu8&expand=2111)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vdbpsadbw, IMM8 = 0))]
pub fn _mm256_dbsad_epu8<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let a = a.as_u8x32();
        let b = b.as_u8x32();
        let r = vdbpsadbw256(a, b, IMM8);
        transmute(r)
    }
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_dbsad_epu8&expand=2112)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(4)]
#[cfg_attr(test, assert_instr(vdbpsadbw, IMM8 = 0))]
pub fn _mm256_mask_dbsad_epu8<const IMM8: i32>(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let a = a.as_u8x32();
        let b = b.as_u8x32();
        let r = vdbpsadbw256(a, b, IMM8);
        transmute(simd_select_bitmask(k, r, src.as_u16x16()))
    }
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_dbsad_epu8&expand=2113)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vdbpsadbw, IMM8 = 0))]
pub fn _mm256_maskz_dbsad_epu8<const IMM8: i32>(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let a = a.as_u8x32();
        let b = b.as_u8x32();
        let r = vdbpsadbw256(a, b, IMM8);
        transmute(simd_select_bitmask(k, r, u16x16::ZERO))
    }
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst. Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_dbsad_epu8&expand=2108)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vdbpsadbw, IMM8 = 0))]
pub fn _mm_dbsad_epu8<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let a = a.as_u8x16();
        let b = b.as_u8x16();
        let r = vdbpsadbw128(a, b, IMM8);
        transmute(r)
    }
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_dbsad_epu8&expand=2109)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(4)]
#[cfg_attr(test, assert_instr(vdbpsadbw, IMM8 = 0))]
pub fn _mm_mask_dbsad_epu8<const IMM8: i32>(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    b: __m128i,
) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let a = a.as_u8x16();
        let b = b.as_u8x16();
        let r = vdbpsadbw128(a, b, IMM8);
        transmute(simd_select_bitmask(k, r, src.as_u16x8()))
    }
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_dbsad_epu8&expand=2110)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vdbpsadbw, IMM8 = 0))]
pub fn _mm_maskz_dbsad_epu8<const IMM8: i32>(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let a = a.as_u8x16();
        let b = b.as_u8x16();
        let r = vdbpsadbw128(a, b, IMM8);
        transmute(simd_select_bitmask(k, r, u16x8::ZERO))
    }
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 16-bit integer in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_movepi16_mask&expand=3873)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovw2m))]
pub fn _mm512_movepi16_mask(a: __m512i) -> __mmask32 {
    let filter = _mm512_set1_epi16(1 << 15);
    let a = _mm512_and_si512(a, filter);
    _mm512_cmpeq_epi16_mask(a, filter)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 16-bit integer in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_movepi16_mask&expand=3872)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovw2m))]
pub fn _mm256_movepi16_mask(a: __m256i) -> __mmask16 {
    let filter = _mm256_set1_epi16(1 << 15);
    let a = _mm256_and_si256(a, filter);
    _mm256_cmpeq_epi16_mask(a, filter)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 16-bit integer in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movepi16_mask&expand=3871)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovw2m))]
pub fn _mm_movepi16_mask(a: __m128i) -> __mmask8 {
    let filter = _mm_set1_epi16(1 << 15);
    let a = _mm_and_si128(a, filter);
    _mm_cmpeq_epi16_mask(a, filter)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 8-bit integer in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_movepi8_mask&expand=3883)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovb2m))]
pub fn _mm512_movepi8_mask(a: __m512i) -> __mmask64 {
    let filter = _mm512_set1_epi8(1 << 7);
    let a = _mm512_and_si512(a, filter);
    _mm512_cmpeq_epi8_mask(a, filter)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 8-bit integer in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_movepi8_mask&expand=3882)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovmskb))] // should be vpmovb2m but compiled to vpmovmskb in the test shim because that takes less cycles than
// using vpmovb2m plus converting the mask register to a standard register.
pub fn _mm256_movepi8_mask(a: __m256i) -> __mmask32 {
    let filter = _mm256_set1_epi8(1 << 7);
    let a = _mm256_and_si256(a, filter);
    _mm256_cmpeq_epi8_mask(a, filter)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 8-bit integer in a.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movepi8_mask&expand=3881)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovmskb))] // should be vpmovb2m but compiled to vpmovmskb in the test shim because that takes less cycles than
// using vpmovb2m plus converting the mask register to a standard register.
pub fn _mm_movepi8_mask(a: __m128i) -> __mmask16 {
    let filter = _mm_set1_epi8(1 << 7);
    let a = _mm_and_si128(a, filter);
    _mm_cmpeq_epi8_mask(a, filter)
}

/// Set each packed 16-bit integer in dst to all ones or all zeros based on the value of the corresponding bit in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_movm_epi16&expand=3886)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovm2w))]
pub fn _mm512_movm_epi16(k: __mmask32) -> __m512i {
    unsafe {
        let one = _mm512_set1_epi16(
            1 << 15
                | 1 << 14
                | 1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
        )
        .as_i16x32();
        transmute(simd_select_bitmask(k, one, i16x32::ZERO))
    }
}

/// Set each packed 16-bit integer in dst to all ones or all zeros based on the value of the corresponding bit in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_movm_epi16&expand=3885)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovm2w))]
pub fn _mm256_movm_epi16(k: __mmask16) -> __m256i {
    unsafe {
        let one = _mm256_set1_epi16(
            1 << 15
                | 1 << 14
                | 1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
        )
        .as_i16x16();
        transmute(simd_select_bitmask(k, one, i16x16::ZERO))
    }
}

/// Set each packed 16-bit integer in dst to all ones or all zeros based on the value of the corresponding bit in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movm_epi16&expand=3884)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovm2w))]
pub fn _mm_movm_epi16(k: __mmask8) -> __m128i {
    unsafe {
        let one = _mm_set1_epi16(
            1 << 15
                | 1 << 14
                | 1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
        )
        .as_i16x8();
        transmute(simd_select_bitmask(k, one, i16x8::ZERO))
    }
}

/// Set each packed 8-bit integer in dst to all ones or all zeros based on the value of the corresponding bit in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_movm_epi8&expand=3895)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovm2b))]
pub fn _mm512_movm_epi8(k: __mmask64) -> __m512i {
    unsafe {
        let one =
            _mm512_set1_epi8(1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0)
                .as_i8x64();
        transmute(simd_select_bitmask(k, one, i8x64::ZERO))
    }
}

/// Set each packed 8-bit integer in dst to all ones or all zeros based on the value of the corresponding bit in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_movm_epi8&expand=3894)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovm2b))]
pub fn _mm256_movm_epi8(k: __mmask32) -> __m256i {
    unsafe {
        let one =
            _mm256_set1_epi8(1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0)
                .as_i8x32();
        transmute(simd_select_bitmask(k, one, i8x32::ZERO))
    }
}

/// Set each packed 8-bit integer in dst to all ones or all zeros based on the value of the corresponding bit in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movm_epi8&expand=3893)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovm2b))]
pub fn _mm_movm_epi8(k: __mmask16) -> __m128i {
    unsafe {
        let one =
            _mm_set1_epi8(1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0)
                .as_i8x16();
        transmute(simd_select_bitmask(k, one, i8x16::ZERO))
    }
}

/// Convert 32-bit mask a into an integer value, and store the result in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#_cvtmask32_u32)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _cvtmask32_u32(a: __mmask32) -> u32 {
    a
}

/// Convert integer value a into an 32-bit mask, and store the result in k.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_cvtu32_mask32)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _cvtu32_mask32(a: u32) -> __mmask32 {
    a
}

/// Add 32-bit masks in a and b, and store the result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kadd_mask32&expand=3207)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kadd_mask32(a: __mmask32, b: __mmask32) -> __mmask32 {
    a + b
}

/// Add 64-bit masks in a and b, and store the result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kadd_mask64&expand=3208)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kadd_mask64(a: __mmask64, b: __mmask64) -> __mmask64 {
    a + b
}

/// Compute the bitwise AND of 32-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kand_mask32&expand=3213)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kand_mask32(a: __mmask32, b: __mmask32) -> __mmask32 {
    a & b
}

/// Compute the bitwise AND of 64-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kand_mask64&expand=3214)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kand_mask64(a: __mmask64, b: __mmask64) -> __mmask64 {
    a & b
}

/// Compute the bitwise NOT of 32-bit mask a, and store the result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_knot_mask32&expand=3234)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _knot_mask32(a: __mmask32) -> __mmask32 {
    !a
}

/// Compute the bitwise NOT of 64-bit mask a, and store the result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_knot_mask64&expand=3235)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _knot_mask64(a: __mmask64) -> __mmask64 {
    !a
}

/// Compute the bitwise NOT of 32-bit masks a and then AND with b, and store the result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kandn_mask32&expand=3219)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kandn_mask32(a: __mmask32, b: __mmask32) -> __mmask32 {
    _knot_mask32(a) & b
}

/// Compute the bitwise NOT of 64-bit masks a and then AND with b, and store the result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kandn_mask64&expand=3220)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kandn_mask64(a: __mmask64, b: __mmask64) -> __mmask64 {
    _knot_mask64(a) & b
}

/// Compute the bitwise OR of 32-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kor_mask32&expand=3240)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kor_mask32(a: __mmask32, b: __mmask32) -> __mmask32 {
    a | b
}

/// Compute the bitwise OR of 64-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kor_mask64&expand=3241)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kor_mask64(a: __mmask64, b: __mmask64) -> __mmask64 {
    a | b
}

/// Compute the bitwise XOR of 32-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kxor_mask32&expand=3292)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kxor_mask32(a: __mmask32, b: __mmask32) -> __mmask32 {
    a ^ b
}

/// Compute the bitwise XOR of 64-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kxor_mask64&expand=3293)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kxor_mask64(a: __mmask64, b: __mmask64) -> __mmask64 {
    a ^ b
}

/// Compute the bitwise XNOR of 32-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kxnor_mask32&expand=3286)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kxnor_mask32(a: __mmask32, b: __mmask32) -> __mmask32 {
    _knot_mask32(a ^ b)
}

/// Compute the bitwise XNOR of 64-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kxnor_mask64&expand=3287)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kxnor_mask64(a: __mmask64, b: __mmask64) -> __mmask64 {
    _knot_mask64(a ^ b)
}

/// Compute the bitwise OR of 32-bit masks a and b. If the result is all zeros, store 1 in dst, otherwise
/// store 0 in dst. If the result is all ones, store 1 in all_ones, otherwise store 0 in all_ones.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kortest_mask32_u8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _kortest_mask32_u8(a: __mmask32, b: __mmask32, all_ones: *mut u8) -> u8 {
    let tmp = _kor_mask32(a, b);
    *all_ones = (tmp == 0xffffffff) as u8;
    (tmp == 0) as u8
}

/// Compute the bitwise OR of 64-bit masks a and b. If the result is all zeros, store 1 in dst, otherwise
/// store 0 in dst. If the result is all ones, store 1 in all_ones, otherwise store 0 in all_ones.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kortest_mask64_u8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _kortest_mask64_u8(a: __mmask64, b: __mmask64, all_ones: *mut u8) -> u8 {
    let tmp = _kor_mask64(a, b);
    *all_ones = (tmp == 0xffffffff_ffffffff) as u8;
    (tmp == 0) as u8
}

/// Compute the bitwise OR of 32-bit masks a and b. If the result is all ones, store 1 in dst, otherwise
/// store 0 in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kortestc_mask32_u8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kortestc_mask32_u8(a: __mmask32, b: __mmask32) -> u8 {
    (_kor_mask32(a, b) == 0xffffffff) as u8
}

/// Compute the bitwise OR of 64-bit masks a and b. If the result is all ones, store 1 in dst, otherwise
/// store 0 in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kortestc_mask64_u8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kortestc_mask64_u8(a: __mmask64, b: __mmask64) -> u8 {
    (_kor_mask64(a, b) == 0xffffffff_ffffffff) as u8
}

/// Compute the bitwise OR of 32-bit masks a and b. If the result is all zeros, store 1 in dst, otherwise
/// store 0 in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kortestz_mask32_u8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kortestz_mask32_u8(a: __mmask32, b: __mmask32) -> u8 {
    (_kor_mask32(a, b) == 0) as u8
}

/// Compute the bitwise OR of 64-bit masks a and b. If the result is all zeros, store 1 in dst, otherwise
/// store 0 in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kortestz_mask64_u8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kortestz_mask64_u8(a: __mmask64, b: __mmask64) -> u8 {
    (_kor_mask64(a, b) == 0) as u8
}

/// Shift the bits of 32-bit mask a left by count while shifting in zeros, and store the least significant 32 bits of the result in k.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kshiftli_mask32)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kshiftli_mask32<const COUNT: u32>(a: __mmask32) -> __mmask32 {
    a << COUNT
}

/// Shift the bits of 64-bit mask a left by count while shifting in zeros, and store the least significant 32 bits of the result in k.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kshiftli_mask64)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kshiftli_mask64<const COUNT: u32>(a: __mmask64) -> __mmask64 {
    a << COUNT
}

/// Shift the bits of 32-bit mask a right by count while shifting in zeros, and store the least significant 32 bits of the result in k.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kshiftri_mask32)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kshiftri_mask32<const COUNT: u32>(a: __mmask32) -> __mmask32 {
    a >> COUNT
}

/// Shift the bits of 64-bit mask a right by count while shifting in zeros, and store the least significant 32 bits of the result in k.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_kshiftri_mask64)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _kshiftri_mask64<const COUNT: u32>(a: __mmask64) -> __mmask64 {
    a >> COUNT
}

/// Compute the bitwise AND of 32-bit masks a and b, and if the result is all zeros, store 1 in dst,
/// otherwise store 0 in dst. Compute the bitwise NOT of a and then AND with b, if the result is all
/// zeros, store 1 in and_not, otherwise store 0 in and_not.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_ktest_mask32_u8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _ktest_mask32_u8(a: __mmask32, b: __mmask32, and_not: *mut u8) -> u8 {
    *and_not = (_kandn_mask32(a, b) == 0) as u8;
    (_kand_mask32(a, b) == 0) as u8
}

/// Compute the bitwise AND of 64-bit masks a and b, and if the result is all zeros, store 1 in dst,
/// otherwise store 0 in dst. Compute the bitwise NOT of a and then AND with b, if the result is all
/// zeros, store 1 in and_not, otherwise store 0 in and_not.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_ktest_mask64_u8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub unsafe fn _ktest_mask64_u8(a: __mmask64, b: __mmask64, and_not: *mut u8) -> u8 {
    *and_not = (_kandn_mask64(a, b) == 0) as u8;
    (_kand_mask64(a, b) == 0) as u8
}

/// Compute the bitwise NOT of 32-bit mask a and then AND with 16-bit mask b, if the result is all
/// zeros, store 1 in dst, otherwise store 0 in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_ktestc_mask32_u8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _ktestc_mask32_u8(a: __mmask32, b: __mmask32) -> u8 {
    (_kandn_mask32(a, b) == 0) as u8
}

/// Compute the bitwise NOT of 64-bit mask a and then AND with 8-bit mask b, if the result is all
/// zeros, store 1 in dst, otherwise store 0 in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_ktestc_mask64_u8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _ktestc_mask64_u8(a: __mmask64, b: __mmask64) -> u8 {
    (_kandn_mask64(a, b) == 0) as u8
}

/// Compute the bitwise AND of 32-bit masks a and  b, if the result is all zeros, store 1 in dst, otherwise
/// store 0 in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_ktestz_mask32_u8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _ktestz_mask32_u8(a: __mmask32, b: __mmask32) -> u8 {
    (_kand_mask32(a, b) == 0) as u8
}

/// Compute the bitwise AND of 64-bit masks a and  b, if the result is all zeros, store 1 in dst, otherwise
/// store 0 in dst.
///
/// [Intel's Documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_ktestz_mask64_u8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
pub fn _ktestz_mask64_u8(a: __mmask64, b: __mmask64) -> u8 {
    (_kand_mask64(a, b) == 0) as u8
}

/// Unpack and interleave 16 bits from masks a and b, and store the 32-bit result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_kunpackw)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(mov))] // generate normal and code instead of kunpckwd
pub fn _mm512_kunpackw(a: __mmask32, b: __mmask32) -> __mmask32 {
    ((a & 0xffff) << 16) | (b & 0xffff)
}

/// Unpack and interleave 32 bits from masks a and b, and store the 64-bit result in k.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_kunpackd)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(mov))] // generate normal and code instead of kunpckdq
pub fn _mm512_kunpackd(a: __mmask64, b: __mmask64) -> __mmask64 {
    ((a & 0xffffffff) << 32) | (b & 0xffffffff)
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtepi16_epi8&expand=1407)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub fn _mm512_cvtepi16_epi8(a: __m512i) -> __m256i {
    unsafe {
        let a = a.as_i16x32();
        transmute::<i8x32, _>(simd_cast(a))
    }
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepi16_epi8&expand=1408)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub fn _mm512_mask_cvtepi16_epi8(src: __m256i, k: __mmask32, a: __m512i) -> __m256i {
    unsafe {
        let convert = _mm512_cvtepi16_epi8(a).as_i8x32();
        transmute(simd_select_bitmask(k, convert, src.as_i8x32()))
    }
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtepi16_epi8&expand=1409)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub fn _mm512_maskz_cvtepi16_epi8(k: __mmask32, a: __m512i) -> __m256i {
    unsafe {
        let convert = _mm512_cvtepi16_epi8(a).as_i8x32();
        transmute(simd_select_bitmask(k, convert, i8x32::ZERO))
    }
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtepi16_epi8&expand=1404)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub fn _mm256_cvtepi16_epi8(a: __m256i) -> __m128i {
    unsafe {
        let a = a.as_i16x16();
        transmute::<i8x16, _>(simd_cast(a))
    }
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepi16_epi8&expand=1405)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub fn _mm256_mask_cvtepi16_epi8(src: __m128i, k: __mmask16, a: __m256i) -> __m128i {
    unsafe {
        let convert = _mm256_cvtepi16_epi8(a).as_i8x16();
        transmute(simd_select_bitmask(k, convert, src.as_i8x16()))
    }
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtepi16_epi8&expand=1406)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub fn _mm256_maskz_cvtepi16_epi8(k: __mmask16, a: __m256i) -> __m128i {
    unsafe {
        let convert = _mm256_cvtepi16_epi8(a).as_i8x16();
        transmute(simd_select_bitmask(k, convert, i8x16::ZERO))
    }
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtepi16_epi8&expand=1401)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub fn _mm_cvtepi16_epi8(a: __m128i) -> __m128i {
    unsafe {
        let a = a.as_i16x8();
        let v256: i16x16 = simd_shuffle!(
            a,
            i16x8::ZERO,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8]
        );
        transmute::<i8x16, _>(simd_cast(v256))
    }
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepi16_epi8&expand=1402)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub fn _mm_mask_cvtepi16_epi8(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let convert = _mm_cvtepi16_epi8(a).as_i8x16();
        let k: __mmask16 = 0b11111111_11111111 & k as __mmask16;
        transmute(simd_select_bitmask(k, convert, src.as_i8x16()))
    }
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtepi16_epi8&expand=1403)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub fn _mm_maskz_cvtepi16_epi8(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let convert = _mm_cvtepi16_epi8(a).as_i8x16();
        let k: __mmask16 = 0b11111111_11111111 & k as __mmask16;
        transmute(simd_select_bitmask(k, convert, i8x16::ZERO))
    }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtsepi16_epi8&expand=1807)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub fn _mm512_cvtsepi16_epi8(a: __m512i) -> __m256i {
    unsafe {
        transmute(vpmovswb(
            a.as_i16x32(),
            i8x32::ZERO,
            0b11111111_11111111_11111111_11111111,
        ))
    }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtsepi16_epi8&expand=1808)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub fn _mm512_mask_cvtsepi16_epi8(src: __m256i, k: __mmask32, a: __m512i) -> __m256i {
    unsafe { transmute(vpmovswb(a.as_i16x32(), src.as_i8x32(), k)) }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtsepi16_epi8&expand=1809)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub fn _mm512_maskz_cvtsepi16_epi8(k: __mmask32, a: __m512i) -> __m256i {
    unsafe { transmute(vpmovswb(a.as_i16x32(), i8x32::ZERO, k)) }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtsepi16_epi8&expand=1804)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub fn _mm256_cvtsepi16_epi8(a: __m256i) -> __m128i {
    unsafe { transmute(vpmovswb256(a.as_i16x16(), i8x16::ZERO, 0b11111111_11111111)) }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtsepi16_epi8&expand=1805)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub fn _mm256_mask_cvtsepi16_epi8(src: __m128i, k: __mmask16, a: __m256i) -> __m128i {
    unsafe { transmute(vpmovswb256(a.as_i16x16(), src.as_i8x16(), k)) }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtsepi16_epi8&expand=1806)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub fn _mm256_maskz_cvtsepi16_epi8(k: __mmask16, a: __m256i) -> __m128i {
    unsafe { transmute(vpmovswb256(a.as_i16x16(), i8x16::ZERO, k)) }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtsepi16_epi8&expand=1801)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub fn _mm_cvtsepi16_epi8(a: __m128i) -> __m128i {
    unsafe { transmute(vpmovswb128(a.as_i16x8(), i8x16::ZERO, 0b11111111)) }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtsepi16_epi8&expand=1802)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub fn _mm_mask_cvtsepi16_epi8(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe { transmute(vpmovswb128(a.as_i16x8(), src.as_i8x16(), k)) }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtsepi16_epi8&expand=1803)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub fn _mm_maskz_cvtsepi16_epi8(k: __mmask8, a: __m128i) -> __m128i {
    unsafe { transmute(vpmovswb128(a.as_i16x8(), i8x16::ZERO, k)) }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtusepi16_epi8&expand=2042)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub fn _mm512_cvtusepi16_epi8(a: __m512i) -> __m256i {
    unsafe {
        transmute(vpmovuswb(
            a.as_u16x32(),
            u8x32::ZERO,
            0b11111111_11111111_11111111_11111111,
        ))
    }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtusepi16_epi8&expand=2043)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub fn _mm512_mask_cvtusepi16_epi8(src: __m256i, k: __mmask32, a: __m512i) -> __m256i {
    unsafe { transmute(vpmovuswb(a.as_u16x32(), src.as_u8x32(), k)) }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtusepi16_epi8&expand=2044)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub fn _mm512_maskz_cvtusepi16_epi8(k: __mmask32, a: __m512i) -> __m256i {
    unsafe { transmute(vpmovuswb(a.as_u16x32(), u8x32::ZERO, k)) }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cvtusepi16_epi8&expand=2039)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub fn _mm256_cvtusepi16_epi8(a: __m256i) -> __m128i {
    unsafe {
        transmute(vpmovuswb256(
            a.as_u16x16(),
            u8x16::ZERO,
            0b11111111_11111111,
        ))
    }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtusepi16_epi8&expand=2040)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub fn _mm256_mask_cvtusepi16_epi8(src: __m128i, k: __mmask16, a: __m256i) -> __m128i {
    unsafe { transmute(vpmovuswb256(a.as_u16x16(), src.as_u8x16(), k)) }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtusepi16_epi8&expand=2041)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub fn _mm256_maskz_cvtusepi16_epi8(k: __mmask16, a: __m256i) -> __m128i {
    unsafe { transmute(vpmovuswb256(a.as_u16x16(), u8x16::ZERO, k)) }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cvtusepi16_epi8&expand=2036)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub fn _mm_cvtusepi16_epi8(a: __m128i) -> __m128i {
    unsafe { transmute(vpmovuswb128(a.as_u16x8(), u8x16::ZERO, 0b11111111)) }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtusepi16_epi8&expand=2037)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub fn _mm_mask_cvtusepi16_epi8(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe { transmute(vpmovuswb128(a.as_u16x8(), src.as_u8x16(), k)) }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtusepi16_epi8&expand=2038)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub fn _mm_maskz_cvtusepi16_epi8(k: __mmask8, a: __m128i) -> __m128i {
    unsafe { transmute(vpmovuswb128(a.as_u16x8(), u8x16::ZERO, k)) }
}

/// Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtepi8_epi16&expand=1526)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub fn _mm512_cvtepi8_epi16(a: __m256i) -> __m512i {
    unsafe {
        let a = a.as_i8x32();
        transmute::<i16x32, _>(simd_cast(a))
    }
}

/// Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepi8_epi16&expand=1527)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub fn _mm512_mask_cvtepi8_epi16(src: __m512i, k: __mmask32, a: __m256i) -> __m512i {
    unsafe {
        let convert = _mm512_cvtepi8_epi16(a).as_i16x32();
        transmute(simd_select_bitmask(k, convert, src.as_i16x32()))
    }
}

/// Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtepi8_epi16&expand=1528)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub fn _mm512_maskz_cvtepi8_epi16(k: __mmask32, a: __m256i) -> __m512i {
    unsafe {
        let convert = _mm512_cvtepi8_epi16(a).as_i16x32();
        transmute(simd_select_bitmask(k, convert, i16x32::ZERO))
    }
}

/// Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepi8_epi16&expand=1524)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub fn _mm256_mask_cvtepi8_epi16(src: __m256i, k: __mmask16, a: __m128i) -> __m256i {
    unsafe {
        let convert = _mm256_cvtepi8_epi16(a).as_i16x16();
        transmute(simd_select_bitmask(k, convert, src.as_i16x16()))
    }
}

/// Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtepi8_epi16&expand=1525)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub fn _mm256_maskz_cvtepi8_epi16(k: __mmask16, a: __m128i) -> __m256i {
    unsafe {
        let convert = _mm256_cvtepi8_epi16(a).as_i16x16();
        transmute(simd_select_bitmask(k, convert, i16x16::ZERO))
    }
}

/// Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepi8_epi16&expand=1521)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub fn _mm_mask_cvtepi8_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let convert = _mm_cvtepi8_epi16(a).as_i16x8();
        transmute(simd_select_bitmask(k, convert, src.as_i16x8()))
    }
}

/// Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtepi8_epi16&expand=1522)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub fn _mm_maskz_cvtepi8_epi16(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let convert = _mm_cvtepi8_epi16(a).as_i16x8();
        transmute(simd_select_bitmask(k, convert, i16x8::ZERO))
    }
}

/// Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cvtepu8_epi16&expand=1612)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub fn _mm512_cvtepu8_epi16(a: __m256i) -> __m512i {
    unsafe {
        let a = a.as_u8x32();
        transmute::<i16x32, _>(simd_cast(a))
    }
}

/// Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepu8_epi16&expand=1613)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub fn _mm512_mask_cvtepu8_epi16(src: __m512i, k: __mmask32, a: __m256i) -> __m512i {
    unsafe {
        let convert = _mm512_cvtepu8_epi16(a).as_i16x32();
        transmute(simd_select_bitmask(k, convert, src.as_i16x32()))
    }
}

/// Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_cvtepu8_epi16&expand=1614)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub fn _mm512_maskz_cvtepu8_epi16(k: __mmask32, a: __m256i) -> __m512i {
    unsafe {
        let convert = _mm512_cvtepu8_epi16(a).as_i16x32();
        transmute(simd_select_bitmask(k, convert, i16x32::ZERO))
    }
}

/// Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepu8_epi16&expand=1610)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub fn _mm256_mask_cvtepu8_epi16(src: __m256i, k: __mmask16, a: __m128i) -> __m256i {
    unsafe {
        let convert = _mm256_cvtepu8_epi16(a).as_i16x16();
        transmute(simd_select_bitmask(k, convert, src.as_i16x16()))
    }
}

/// Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_cvtepu8_epi16&expand=1611)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub fn _mm256_maskz_cvtepu8_epi16(k: __mmask16, a: __m128i) -> __m256i {
    unsafe {
        let convert = _mm256_cvtepu8_epi16(a).as_i16x16();
        transmute(simd_select_bitmask(k, convert, i16x16::ZERO))
    }
}

/// Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepu8_epi16&expand=1607)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub fn _mm_mask_cvtepu8_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let convert = _mm_cvtepu8_epi16(a).as_i16x8();
        transmute(simd_select_bitmask(k, convert, src.as_i16x8()))
    }
}

/// Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_cvtepu8_epi16&expand=1608)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub fn _mm_maskz_cvtepu8_epi16(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        let convert = _mm_cvtepu8_epi16(a).as_i16x8();
        transmute(simd_select_bitmask(k, convert, i16x8::ZERO))
    }
}

/// Shift 128-bit lanes in a left by imm8 bytes while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_bslli_epi128&expand=591)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpslldq, IMM8 = 3))]
#[rustc_legacy_const_generics(1)]
pub fn _mm512_bslli_epi128<const IMM8: i32>(a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        const fn mask(shift: i32, i: u32) -> u32 {
            let shift = shift as u32 & 0xff;
            if shift > 15 || i % 16 < shift {
                0
            } else {
                64 + (i - shift)
            }
        }
        let a = a.as_i8x64();
        let zero = i8x64::ZERO;
        let r: i8x64 = simd_shuffle!(
            zero,
            a,
            [
                mask(IMM8, 0),
                mask(IMM8, 1),
                mask(IMM8, 2),
                mask(IMM8, 3),
                mask(IMM8, 4),
                mask(IMM8, 5),
                mask(IMM8, 6),
                mask(IMM8, 7),
                mask(IMM8, 8),
                mask(IMM8, 9),
                mask(IMM8, 10),
                mask(IMM8, 11),
                mask(IMM8, 12),
                mask(IMM8, 13),
                mask(IMM8, 14),
                mask(IMM8, 15),
                mask(IMM8, 16),
                mask(IMM8, 17),
                mask(IMM8, 18),
                mask(IMM8, 19),
                mask(IMM8, 20),
                mask(IMM8, 21),
                mask(IMM8, 22),
                mask(IMM8, 23),
                mask(IMM8, 24),
                mask(IMM8, 25),
                mask(IMM8, 26),
                mask(IMM8, 27),
                mask(IMM8, 28),
                mask(IMM8, 29),
                mask(IMM8, 30),
                mask(IMM8, 31),
                mask(IMM8, 32),
                mask(IMM8, 33),
                mask(IMM8, 34),
                mask(IMM8, 35),
                mask(IMM8, 36),
                mask(IMM8, 37),
                mask(IMM8, 38),
                mask(IMM8, 39),
                mask(IMM8, 40),
                mask(IMM8, 41),
                mask(IMM8, 42),
                mask(IMM8, 43),
                mask(IMM8, 44),
                mask(IMM8, 45),
                mask(IMM8, 46),
                mask(IMM8, 47),
                mask(IMM8, 48),
                mask(IMM8, 49),
                mask(IMM8, 50),
                mask(IMM8, 51),
                mask(IMM8, 52),
                mask(IMM8, 53),
                mask(IMM8, 54),
                mask(IMM8, 55),
                mask(IMM8, 56),
                mask(IMM8, 57),
                mask(IMM8, 58),
                mask(IMM8, 59),
                mask(IMM8, 60),
                mask(IMM8, 61),
                mask(IMM8, 62),
                mask(IMM8, 63),
            ],
        );
        transmute(r)
    }
}

/// Shift 128-bit lanes in a right by imm8 bytes while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_bsrli_epi128&expand=594)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpsrldq, IMM8 = 3))]
#[rustc_legacy_const_generics(1)]
pub fn _mm512_bsrli_epi128<const IMM8: i32>(a: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let a = a.as_i8x64();
        let zero = i8x64::ZERO;
        let r: i8x64 = match IMM8 % 16 {
            0 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                        59, 60, 61, 62, 63,
                    ],
                )
            }
            1 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 17, 18, 19, 20, 21,
                        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 80, 33, 34, 35, 36, 37, 38, 39, 40,
                        41, 42, 43, 44, 45, 46, 47, 96, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                        60, 61, 62, 63, 112,
                    ],
                )
            }
            2 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 18, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 31, 80, 81, 34, 35, 36, 37, 38, 39, 40, 41,
                        42, 43, 44, 45, 46, 47, 96, 97, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                        61, 62, 63, 112, 113,
                    ],
                )
            }
            3 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 31, 80, 81, 82, 35, 36, 37, 38, 39, 40, 41,
                        42, 43, 44, 45, 46, 47, 96, 97, 98, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                        61, 62, 63, 112, 113, 114,
                    ],
                )
            }
            4 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, 31, 80, 81, 82, 83, 36, 37, 38, 39, 40, 41, 42,
                        43, 44, 45, 46, 47, 96, 97, 98, 99, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                        62, 63, 112, 113, 114, 115,
                    ],
                )
            }
            5 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 21, 22, 23, 24,
                        25, 26, 27, 28, 29, 30, 31, 80, 81, 82, 83, 84, 37, 38, 39, 40, 41, 42, 43,
                        44, 45, 46, 47, 96, 97, 98, 99, 100, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                        62, 63, 112, 113, 114, 115, 116,
                    ],
                )
            }
            6 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 22, 23, 24, 25,
                        26, 27, 28, 29, 30, 31, 80, 81, 82, 83, 84, 85, 38, 39, 40, 41, 42, 43, 44,
                        45, 46, 47, 96, 97, 98, 99, 100, 101, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                        63, 112, 113, 114, 115, 116, 117,
                    ],
                )
            }
            7 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 23, 24, 25,
                        26, 27, 28, 29, 30, 31, 80, 81, 82, 83, 84, 85, 86, 39, 40, 41, 42, 43, 44,
                        45, 46, 47, 96, 97, 98, 99, 100, 101, 102, 55, 56, 57, 58, 59, 60, 61, 62,
                        63, 112, 113, 114, 115, 116, 117, 118,
                    ],
                )
            }
            8 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 24, 25, 26,
                        27, 28, 29, 30, 31, 80, 81, 82, 83, 84, 85, 86, 87, 40, 41, 42, 43, 44, 45,
                        46, 47, 96, 97, 98, 99, 100, 101, 102, 103, 56, 57, 58, 59, 60, 61, 62, 63,
                        112, 113, 114, 115, 116, 117, 118, 119,
                    ],
                )
            }
            9 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 25, 26, 27,
                        28, 29, 30, 31, 80, 81, 82, 83, 84, 85, 86, 87, 88, 41, 42, 43, 44, 45, 46,
                        47, 96, 97, 98, 99, 100, 101, 102, 103, 104, 57, 58, 59, 60, 61, 62, 63,
                        112, 113, 114, 115, 116, 117, 118, 119, 120,
                    ],
                )
            }
            10 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 26, 27, 28,
                        29, 30, 31, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 42, 43, 44, 45, 46, 47,
                        96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 58, 59, 60, 61, 62, 63, 112,
                        113, 114, 115, 116, 117, 118, 119, 120, 121,
                    ],
                )
            }
            11 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 27, 28, 29,
                        30, 31, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 43, 44, 45, 46, 47, 96,
                        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 59, 60, 61, 62, 63, 112,
                        113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                    ],
                )
            }
            12 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 28, 29, 30,
                        31, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 44, 45, 46, 47, 96, 97,
                        98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 60, 61, 62, 63, 112, 113,
                        114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
                    ],
                )
            }
            13 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 29, 30, 31,
                        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 45, 46, 47, 96, 97, 98,
                        99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 61, 62, 63, 112, 113, 114,
                        115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
                    ],
                )
            }
            14 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 30, 31, 80,
                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 46, 47, 96, 97, 98, 99,
                        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 62, 63, 112, 113, 114,
                        115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                    ],
                )
            }
            15 => {
                simd_shuffle!(
                    a,
                    zero,
                    [
                        15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 31, 80, 81,
                        82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 47, 96, 97, 98, 99,
                        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 63, 112, 113, 114,
                        115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
                    ],
                )
            }
            _ => zero,
        };
        transmute(r)
    }
}

/// Concatenate pairs of 16-byte blocks in a and b into a 32-byte temporary result, shift the result right by imm8 bytes, and store the low 16 bytes in dst.
/// Unlike [`_mm_alignr_epi8`], [`_mm256_alignr_epi8`] functions, where the entire input vectors are concatenated to the temporary result,
/// this concatenation happens in 4 steps, where each step builds 32-byte temporary result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_alignr_epi8&expand=263)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpalignr, IMM8 = 1))]
#[rustc_legacy_const_generics(2)]
pub fn _mm512_alignr_epi8<const IMM8: i32>(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        // If palignr is shifting the pair of vectors more than the size of two
        // lanes, emit zero.
        if IMM8 >= 32 {
            return _mm512_setzero_si512();
        }
        // If palignr is shifting the pair of input vectors more than one lane,
        // but less than two lanes, convert to shifting in zeroes.
        let (a, b) = if IMM8 > 16 {
            (_mm512_setzero_si512(), a)
        } else {
            (a, b)
        };
        let a = a.as_i8x64();
        let b = b.as_i8x64();
        if IMM8 == 16 {
            return transmute(a);
        }
        let r: i8x64 = match IMM8 % 16 {
            0 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                        59, 60, 61, 62, 63,
                    ],
                )
            }
            1 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 17, 18, 19, 20, 21,
                        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 80, 33, 34, 35, 36, 37, 38, 39, 40,
                        41, 42, 43, 44, 45, 46, 47, 96, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                        60, 61, 62, 63, 112,
                    ],
                )
            }
            2 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 18, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 31, 80, 81, 34, 35, 36, 37, 38, 39, 40, 41,
                        42, 43, 44, 45, 46, 47, 96, 97, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                        61, 62, 63, 112, 113,
                    ],
                )
            }
            3 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 31, 80, 81, 82, 35, 36, 37, 38, 39, 40, 41,
                        42, 43, 44, 45, 46, 47, 96, 97, 98, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                        61, 62, 63, 112, 113, 114,
                    ],
                )
            }
            4 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, 31, 80, 81, 82, 83, 36, 37, 38, 39, 40, 41, 42,
                        43, 44, 45, 46, 47, 96, 97, 98, 99, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                        62, 63, 112, 113, 114, 115,
                    ],
                )
            }
            5 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 21, 22, 23, 24,
                        25, 26, 27, 28, 29, 30, 31, 80, 81, 82, 83, 84, 37, 38, 39, 40, 41, 42, 43,
                        44, 45, 46, 47, 96, 97, 98, 99, 100, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                        62, 63, 112, 113, 114, 115, 116,
                    ],
                )
            }
            6 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 22, 23, 24, 25,
                        26, 27, 28, 29, 30, 31, 80, 81, 82, 83, 84, 85, 38, 39, 40, 41, 42, 43, 44,
                        45, 46, 47, 96, 97, 98, 99, 100, 101, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                        63, 112, 113, 114, 115, 116, 117,
                    ],
                )
            }
            7 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 23, 24, 25,
                        26, 27, 28, 29, 30, 31, 80, 81, 82, 83, 84, 85, 86, 39, 40, 41, 42, 43, 44,
                        45, 46, 47, 96, 97, 98, 99, 100, 101, 102, 55, 56, 57, 58, 59, 60, 61, 62,
                        63, 112, 113, 114, 115, 116, 117, 118,
                    ],
                )
            }
            8 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 24, 25, 26,
                        27, 28, 29, 30, 31, 80, 81, 82, 83, 84, 85, 86, 87, 40, 41, 42, 43, 44, 45,
                        46, 47, 96, 97, 98, 99, 100, 101, 102, 103, 56, 57, 58, 59, 60, 61, 62, 63,
                        112, 113, 114, 115, 116, 117, 118, 119,
                    ],
                )
            }
            9 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 25, 26, 27,
                        28, 29, 30, 31, 80, 81, 82, 83, 84, 85, 86, 87, 88, 41, 42, 43, 44, 45, 46,
                        47, 96, 97, 98, 99, 100, 101, 102, 103, 104, 57, 58, 59, 60, 61, 62, 63,
                        112, 113, 114, 115, 116, 117, 118, 119, 120,
                    ],
                )
            }
            10 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 26, 27, 28,
                        29, 30, 31, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 42, 43, 44, 45, 46, 47,
                        96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 58, 59, 60, 61, 62, 63, 112,
                        113, 114, 115, 116, 117, 118, 119, 120, 121,
                    ],
                )
            }
            11 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 27, 28, 29,
                        30, 31, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 43, 44, 45, 46, 47, 96,
                        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 59, 60, 61, 62, 63, 112,
                        113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                    ],
                )
            }
            12 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 28, 29, 30,
                        31, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 44, 45, 46, 47, 96, 97,
                        98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 60, 61, 62, 63, 112, 113,
                        114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
                    ],
                )
            }
            13 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 29, 30, 31,
                        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 45, 46, 47, 96, 97, 98,
                        99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 61, 62, 63, 112, 113, 114,
                        115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
                    ],
                )
            }
            14 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 30, 31, 80,
                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 46, 47, 96, 97, 98, 99,
                        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 62, 63, 112, 113, 114,
                        115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                    ],
                )
            }
            15 => {
                simd_shuffle!(
                    b,
                    a,
                    [
                        15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 31, 80, 81,
                        82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 47, 96, 97, 98, 99,
                        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 63, 112, 113, 114,
                        115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
                    ],
                )
            }
            _ => unreachable_unchecked(),
        };
        transmute(r)
    }
}

/// Concatenate pairs of 16-byte blocks in a and b into a 32-byte temporary result, shift the result right by imm8 bytes, and store the low 16 bytes in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_alignr_epi8&expand=264)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpalignr, IMM8 = 1))]
#[rustc_legacy_const_generics(4)]
pub fn _mm512_mask_alignr_epi8<const IMM8: i32>(
    src: __m512i,
    k: __mmask64,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let r = _mm512_alignr_epi8::<IMM8>(a, b);
        transmute(simd_select_bitmask(k, r.as_i8x64(), src.as_i8x64()))
    }
}

/// Concatenate pairs of 16-byte blocks in a and b into a 32-byte temporary result, shift the result right by imm8 bytes, and store the low 16 bytes in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_alignr_epi8&expand=265)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpalignr, IMM8 = 1))]
#[rustc_legacy_const_generics(3)]
pub fn _mm512_maskz_alignr_epi8<const IMM8: i32>(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let r = _mm512_alignr_epi8::<IMM8>(a, b);
        transmute(simd_select_bitmask(k, r.as_i8x64(), i8x64::ZERO))
    }
}

/// Concatenate pairs of 16-byte blocks in a and b into a 32-byte temporary result, shift the result right by imm8 bytes, and store the low 16 bytes in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_alignr_epi8&expand=261)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(4)]
#[cfg_attr(test, assert_instr(vpalignr, IMM8 = 5))]
pub fn _mm256_mask_alignr_epi8<const IMM8: i32>(
    src: __m256i,
    k: __mmask32,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let r = _mm256_alignr_epi8::<IMM8>(a, b);
        transmute(simd_select_bitmask(k, r.as_i8x32(), src.as_i8x32()))
    }
}

/// Concatenate pairs of 16-byte blocks in a and b into a 32-byte temporary result, shift the result right by imm8 bytes, and store the low 16 bytes in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_alignr_epi8&expand=262)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpalignr, IMM8 = 5))]
pub fn _mm256_maskz_alignr_epi8<const IMM8: i32>(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let r = _mm256_alignr_epi8::<IMM8>(a, b);
        transmute(simd_select_bitmask(k, r.as_i8x32(), i8x32::ZERO))
    }
}

/// Concatenate pairs of 16-byte blocks in a and b into a 32-byte temporary result, shift the result right by imm8 bytes, and store the low 16 bytes in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_alignr_epi8&expand=258)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(4)]
#[cfg_attr(test, assert_instr(vpalignr, IMM8 = 5))]
pub fn _mm_mask_alignr_epi8<const IMM8: i32>(
    src: __m128i,
    k: __mmask16,
    a: __m128i,
    b: __m128i,
) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let r = _mm_alignr_epi8::<IMM8>(a, b);
        transmute(simd_select_bitmask(k, r.as_i8x16(), src.as_i8x16()))
    }
}

/// Concatenate pairs of 16-byte blocks in a and b into a 32-byte temporary result, shift the result right by imm8 bytes, and store the low 16 bytes in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_alignr_epi8&expand=259)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpalignr, IMM8 = 5))]
pub fn _mm_maskz_alignr_epi8<const IMM8: i32>(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        static_assert_uimm_bits!(IMM8, 8);
        let r = _mm_alignr_epi8::<IMM8>(a, b);
        transmute(simd_select_bitmask(k, r.as_i8x16(), i8x16::ZERO))
    }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtsepi16_storeu_epi8&expand=1812)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm512_mask_cvtsepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask32, a: __m512i) {
    vpmovswbmem(mem_addr, a.as_i16x32(), k);
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtsepi16_storeu_epi8&expand=1811)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm256_mask_cvtsepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask16, a: __m256i) {
    vpmovswbmem256(mem_addr, a.as_i16x16(), k);
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtsepi16_storeu_epi8&expand=1810)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm_mask_cvtsepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask8, a: __m128i) {
    vpmovswbmem128(mem_addr, a.as_i16x8(), k);
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepi16_storeu_epi8&expand=1412)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm512_mask_cvtepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask32, a: __m512i) {
    vpmovwbmem(mem_addr, a.as_i16x32(), k);
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepi16_storeu_epi8&expand=1411)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm256_mask_cvtepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask16, a: __m256i) {
    vpmovwbmem256(mem_addr, a.as_i16x16(), k);
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepi16_storeu_epi8&expand=1410)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm_mask_cvtepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask8, a: __m128i) {
    vpmovwbmem128(mem_addr, a.as_i16x8(), k);
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtusepi16_storeu_epi8&expand=2047)
#[inline]
#[target_feature(enable = "avx512bw")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm512_mask_cvtusepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask32, a: __m512i) {
    vpmovuswbmem(mem_addr, a.as_i16x32(), k);
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtusepi16_storeu_epi8&expand=2046)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm256_mask_cvtusepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask16, a: __m256i) {
    vpmovuswbmem256(mem_addr, a.as_i16x16(), k);
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtusepi16_storeu_epi8&expand=2045)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm_mask_cvtusepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask8, a: __m128i) {
    vpmovuswbmem128(mem_addr, a.as_i16x8(), k);
}

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.x86.avx512.pmul.hr.sw.512"]
    fn vpmulhrsw(a: i16x32, b: i16x32) -> i16x32;

    #[link_name = "llvm.x86.avx512.pmaddw.d.512"]
    fn vpmaddwd(a: i16x32, b: i16x32) -> i32x16;
    #[link_name = "llvm.x86.avx512.pmaddubs.w.512"]
    fn vpmaddubsw(a: i8x64, b: i8x64) -> i16x32;

    #[link_name = "llvm.x86.avx512.packssdw.512"]
    fn vpackssdw(a: i32x16, b: i32x16) -> i16x32;
    #[link_name = "llvm.x86.avx512.packsswb.512"]
    fn vpacksswb(a: i16x32, b: i16x32) -> i8x64;
    #[link_name = "llvm.x86.avx512.packusdw.512"]
    fn vpackusdw(a: i32x16, b: i32x16) -> u16x32;
    #[link_name = "llvm.x86.avx512.packuswb.512"]
    fn vpackuswb(a: i16x32, b: i16x32) -> u8x64;

    #[link_name = "llvm.x86.avx512.psll.w.512"]
    fn vpsllw(a: i16x32, count: i16x8) -> i16x32;

    #[link_name = "llvm.x86.avx512.psllv.w.512"]
    fn vpsllvw(a: i16x32, b: i16x32) -> i16x32;
    #[link_name = "llvm.x86.avx512.psllv.w.256"]
    fn vpsllvw256(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx512.psllv.w.128"]
    fn vpsllvw128(a: i16x8, b: i16x8) -> i16x8;

    #[link_name = "llvm.x86.avx512.psrl.w.512"]
    fn vpsrlw(a: i16x32, count: i16x8) -> i16x32;

    #[link_name = "llvm.x86.avx512.psrlv.w.512"]
    fn vpsrlvw(a: i16x32, b: i16x32) -> i16x32;
    #[link_name = "llvm.x86.avx512.psrlv.w.256"]
    fn vpsrlvw256(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx512.psrlv.w.128"]
    fn vpsrlvw128(a: i16x8, b: i16x8) -> i16x8;

    #[link_name = "llvm.x86.avx512.psra.w.512"]
    fn vpsraw(a: i16x32, count: i16x8) -> i16x32;

    #[link_name = "llvm.x86.avx512.psrav.w.512"]
    fn vpsravw(a: i16x32, count: i16x32) -> i16x32;
    #[link_name = "llvm.x86.avx512.psrav.w.256"]
    fn vpsravw256(a: i16x16, count: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx512.psrav.w.128"]
    fn vpsravw128(a: i16x8, count: i16x8) -> i16x8;

    #[link_name = "llvm.x86.avx512.vpermi2var.hi.512"]
    fn vpermi2w(a: i16x32, idx: i16x32, b: i16x32) -> i16x32;
    #[link_name = "llvm.x86.avx512.vpermi2var.hi.256"]
    fn vpermi2w256(a: i16x16, idx: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx512.vpermi2var.hi.128"]
    fn vpermi2w128(a: i16x8, idx: i16x8, b: i16x8) -> i16x8;

    #[link_name = "llvm.x86.avx512.permvar.hi.512"]
    fn vpermw(a: i16x32, idx: i16x32) -> i16x32;
    #[link_name = "llvm.x86.avx512.permvar.hi.256"]
    fn vpermw256(a: i16x16, idx: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx512.permvar.hi.128"]
    fn vpermw128(a: i16x8, idx: i16x8) -> i16x8;

    #[link_name = "llvm.x86.avx512.pshuf.b.512"]
    fn vpshufb(a: i8x64, b: i8x64) -> i8x64;

    #[link_name = "llvm.x86.avx512.psad.bw.512"]
    fn vpsadbw(a: u8x64, b: u8x64) -> u64x8;

    #[link_name = "llvm.x86.avx512.dbpsadbw.512"]
    fn vdbpsadbw(a: u8x64, b: u8x64, imm8: i32) -> u16x32;
    #[link_name = "llvm.x86.avx512.dbpsadbw.256"]
    fn vdbpsadbw256(a: u8x32, b: u8x32, imm8: i32) -> u16x16;
    #[link_name = "llvm.x86.avx512.dbpsadbw.128"]
    fn vdbpsadbw128(a: u8x16, b: u8x16, imm8: i32) -> u16x8;

    #[link_name = "llvm.x86.avx512.mask.pmovs.wb.512"]
    fn vpmovswb(a: i16x32, src: i8x32, mask: u32) -> i8x32;
    #[link_name = "llvm.x86.avx512.mask.pmovs.wb.256"]
    fn vpmovswb256(a: i16x16, src: i8x16, mask: u16) -> i8x16;
    #[link_name = "llvm.x86.avx512.mask.pmovs.wb.128"]
    fn vpmovswb128(a: i16x8, src: i8x16, mask: u8) -> i8x16;

    #[link_name = "llvm.x86.avx512.mask.pmovus.wb.512"]
    fn vpmovuswb(a: u16x32, src: u8x32, mask: u32) -> u8x32;
    #[link_name = "llvm.x86.avx512.mask.pmovus.wb.256"]
    fn vpmovuswb256(a: u16x16, src: u8x16, mask: u16) -> u8x16;
    #[link_name = "llvm.x86.avx512.mask.pmovus.wb.128"]
    fn vpmovuswb128(a: u16x8, src: u8x16, mask: u8) -> u8x16;

    #[link_name = "llvm.x86.avx512.mask.pmovs.wb.mem.512"]
    fn vpmovswbmem(mem_addr: *mut i8, a: i16x32, mask: u32);
    #[link_name = "llvm.x86.avx512.mask.pmovs.wb.mem.256"]
    fn vpmovswbmem256(mem_addr: *mut i8, a: i16x16, mask: u16);
    #[link_name = "llvm.x86.avx512.mask.pmovs.wb.mem.128"]
    fn vpmovswbmem128(mem_addr: *mut i8, a: i16x8, mask: u8);

    #[link_name = "llvm.x86.avx512.mask.pmov.wb.mem.512"]
    fn vpmovwbmem(mem_addr: *mut i8, a: i16x32, mask: u32);
    #[link_name = "llvm.x86.avx512.mask.pmov.wb.mem.256"]
    fn vpmovwbmem256(mem_addr: *mut i8, a: i16x16, mask: u16);
    #[link_name = "llvm.x86.avx512.mask.pmov.wb.mem.128"]
    fn vpmovwbmem128(mem_addr: *mut i8, a: i16x8, mask: u8);

    #[link_name = "llvm.x86.avx512.mask.pmovus.wb.mem.512"]
    fn vpmovuswbmem(mem_addr: *mut i8, a: i16x32, mask: u32);
    #[link_name = "llvm.x86.avx512.mask.pmovus.wb.mem.256"]
    fn vpmovuswbmem256(mem_addr: *mut i8, a: i16x16, mask: u16);
    #[link_name = "llvm.x86.avx512.mask.pmovus.wb.mem.128"]
    fn vpmovuswbmem128(mem_addr: *mut i8, a: i16x8, mask: u8);

    #[link_name = "llvm.x86.avx512.mask.loadu.b.128"]
    fn loaddqu8_128(mem_addr: *const i8, a: i8x16, mask: u16) -> i8x16;
    #[link_name = "llvm.x86.avx512.mask.loadu.w.128"]
    fn loaddqu16_128(mem_addr: *const i16, a: i16x8, mask: u8) -> i16x8;
    #[link_name = "llvm.x86.avx512.mask.loadu.b.256"]
    fn loaddqu8_256(mem_addr: *const i8, a: i8x32, mask: u32) -> i8x32;
    #[link_name = "llvm.x86.avx512.mask.loadu.w.256"]
    fn loaddqu16_256(mem_addr: *const i16, a: i16x16, mask: u16) -> i16x16;
    #[link_name = "llvm.x86.avx512.mask.loadu.b.512"]
    fn loaddqu8_512(mem_addr: *const i8, a: i8x64, mask: u64) -> i8x64;
    #[link_name = "llvm.x86.avx512.mask.loadu.w.512"]
    fn loaddqu16_512(mem_addr: *const i16, a: i16x32, mask: u32) -> i16x32;

    #[link_name = "llvm.x86.avx512.mask.storeu.b.128"]
    fn storedqu8_128(mem_addr: *mut i8, a: i8x16, mask: u16);
    #[link_name = "llvm.x86.avx512.mask.storeu.w.128"]
    fn storedqu16_128(mem_addr: *mut i16, a: i16x8, mask: u8);
    #[link_name = "llvm.x86.avx512.mask.storeu.b.256"]
    fn storedqu8_256(mem_addr: *mut i8, a: i8x32, mask: u32);
    #[link_name = "llvm.x86.avx512.mask.storeu.w.256"]
    fn storedqu16_256(mem_addr: *mut i16, a: i16x16, mask: u16);
    #[link_name = "llvm.x86.avx512.mask.storeu.b.512"]
    fn storedqu8_512(mem_addr: *mut i8, a: i8x64, mask: u64);
    #[link_name = "llvm.x86.avx512.mask.storeu.w.512"]
    fn storedqu16_512(mem_addr: *mut i16, a: i16x32, mask: u32);

}

#[cfg(test)]
mod tests {

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;
    use crate::hint::black_box;
    use crate::mem::{self};

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_abs_epi16() {
        let a = _mm512_set1_epi16(-1);
        let r = _mm512_abs_epi16(a);
        let e = _mm512_set1_epi16(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_abs_epi16() {
        let a = _mm512_set1_epi16(-1);
        let r = _mm512_mask_abs_epi16(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_abs_epi16(a, 0b00000000_11111111_00000000_11111111, a);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_abs_epi16() {
        let a = _mm512_set1_epi16(-1);
        let r = _mm512_maskz_abs_epi16(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_abs_epi16(0b00000000_11111111_00000000_11111111, a);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_abs_epi16() {
        let a = _mm256_set1_epi16(-1);
        let r = _mm256_mask_abs_epi16(a, 0, a);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_abs_epi16(a, 0b00000000_11111111, a);
        let e = _mm256_set_epi16(-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_abs_epi16() {
        let a = _mm256_set1_epi16(-1);
        let r = _mm256_maskz_abs_epi16(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_abs_epi16(0b00000000_11111111, a);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_abs_epi16() {
        let a = _mm_set1_epi16(-1);
        let r = _mm_mask_abs_epi16(a, 0, a);
        assert_eq_m128i(r, a);
        let r = _mm_mask_abs_epi16(a, 0b00001111, a);
        let e = _mm_set_epi16(-1, -1, -1, -1, 1, 1, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_abs_epi16() {
        let a = _mm_set1_epi16(-1);
        let r = _mm_maskz_abs_epi16(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_abs_epi16(0b00001111, a);
        let e = _mm_set_epi16(0, 0, 0, 0, 1, 1, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_abs_epi8() {
        let a = _mm512_set1_epi8(-1);
        let r = _mm512_abs_epi8(a);
        let e = _mm512_set1_epi8(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_abs_epi8() {
        let a = _mm512_set1_epi8(-1);
        let r = _mm512_mask_abs_epi8(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_abs_epi8(
            a,
            0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111,
            a,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                                -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                                -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                                -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_abs_epi8() {
        let a = _mm512_set1_epi8(-1);
        let r = _mm512_maskz_abs_epi8(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_abs_epi8(
            0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111,
            a,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_abs_epi8() {
        let a = _mm256_set1_epi8(-1);
        let r = _mm256_mask_abs_epi8(a, 0, a);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_abs_epi8(a, 0b00000000_11111111_00000000_11111111, a);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                                -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_abs_epi8() {
        let a = _mm256_set1_epi8(-1);
        let r = _mm256_maskz_abs_epi8(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_abs_epi8(0b00000000_11111111_00000000_11111111, a);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_abs_epi8() {
        let a = _mm_set1_epi8(-1);
        let r = _mm_mask_abs_epi8(a, 0, a);
        assert_eq_m128i(r, a);
        let r = _mm_mask_abs_epi8(a, 0b00000000_11111111, a);
        let e = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_abs_epi8() {
        let a = _mm_set1_epi8(-1);
        let r = _mm_maskz_abs_epi8(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_abs_epi8(0b00000000_11111111, a);
        #[rustfmt::skip]
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_add_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(2);
        let r = _mm512_add_epi16(a, b);
        let e = _mm512_set1_epi16(3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_add_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(2);
        let r = _mm512_mask_add_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_add_epi16(a, 0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3,
                                 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_add_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(2);
        let r = _mm512_maskz_add_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_add_epi16(0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
                                 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_add_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(2);
        let r = _mm256_mask_add_epi16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_add_epi16(a, 0b00000000_11111111, a, b);
        let e = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_add_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(2);
        let r = _mm256_maskz_add_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_add_epi16(0b00000000_11111111, a, b);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_add_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(2);
        let r = _mm_mask_add_epi16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_add_epi16(a, 0b00001111, a, b);
        let e = _mm_set_epi16(1, 1, 1, 1, 3, 3, 3, 3);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_add_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(2);
        let r = _mm_maskz_add_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_add_epi16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, 3, 3, 3, 3);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_add_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(2);
        let r = _mm512_add_epi8(a, b);
        let e = _mm512_set1_epi8(3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_add_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(2);
        let r = _mm512_mask_add_epi8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_add_epi8(
            a,
            0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_add_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(2);
        let r = _mm512_maskz_add_epi8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_add_epi8(
            0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
                                0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
                                0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
                                0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_add_epi8() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(2);
        let r = _mm256_mask_add_epi8(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_add_epi8(a, 0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_add_epi8() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(2);
        let r = _mm256_maskz_add_epi8(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_add_epi8(0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
                                0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_add_epi8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(2);
        let r = _mm_mask_add_epi8(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_add_epi8(a, 0b00000000_11111111, a, b);
        let e = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_add_epi8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(2);
        let r = _mm_maskz_add_epi8(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_add_epi8(0b00000000_11111111, a, b);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_adds_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(u16::MAX as i16);
        let r = _mm512_adds_epu16(a, b);
        let e = _mm512_set1_epi16(u16::MAX as i16);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_adds_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(u16::MAX as i16);
        let r = _mm512_mask_adds_epu16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_adds_epu16(a, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_adds_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(u16::MAX as i16);
        let r = _mm512_maskz_adds_epu16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_adds_epu16(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_adds_epu16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(u16::MAX as i16);
        let r = _mm256_mask_adds_epu16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_adds_epu16(a, 0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_adds_epu16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(u16::MAX as i16);
        let r = _mm256_maskz_adds_epu16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_adds_epu16(0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_adds_epu16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(u16::MAX as i16);
        let r = _mm_mask_adds_epu16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_adds_epu16(a, 0b00001111, a, b);
        #[rustfmt::skip]
        let e = _mm_set_epi16(1, 1, 1, 1, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_adds_epu16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(u16::MAX as i16);
        let r = _mm_maskz_adds_epu16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_adds_epu16(0b00001111, a, b);
        #[rustfmt::skip]
        let e = _mm_set_epi16(0, 0, 0, 0, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_adds_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(u8::MAX as i8);
        let r = _mm512_adds_epu8(a, b);
        let e = _mm512_set1_epi8(u8::MAX as i8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_adds_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(u8::MAX as i8);
        let r = _mm512_mask_adds_epu8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_adds_epu8(
            a,
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_adds_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(u8::MAX as i8);
        let r = _mm512_maskz_adds_epu8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_adds_epu8(
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_adds_epu8() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(u8::MAX as i8);
        let r = _mm256_mask_adds_epu8(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_adds_epu8(a, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_adds_epu8() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(u8::MAX as i8);
        let r = _mm256_maskz_adds_epu8(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_adds_epu8(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_adds_epu8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(u8::MAX as i8);
        let r = _mm_mask_adds_epu8(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_adds_epu8(a, 0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_adds_epu8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(u8::MAX as i8);
        let r = _mm_maskz_adds_epu8(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_adds_epu8(0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_adds_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_adds_epi16(a, b);
        let e = _mm512_set1_epi16(i16::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_adds_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_mask_adds_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_adds_epi16(a, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_adds_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_maskz_adds_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_adds_epi16(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_adds_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(i16::MAX);
        let r = _mm256_mask_adds_epi16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_adds_epi16(a, 0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_adds_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(i16::MAX);
        let r = _mm256_maskz_adds_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_adds_epi16(0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_adds_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(i16::MAX);
        let r = _mm_mask_adds_epi16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_adds_epi16(a, 0b00001111, a, b);
        let e = _mm_set_epi16(1, 1, 1, 1, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_adds_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(i16::MAX);
        let r = _mm_maskz_adds_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_adds_epi16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_adds_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(i8::MAX);
        let r = _mm512_adds_epi8(a, b);
        let e = _mm512_set1_epi8(i8::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_adds_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(i8::MAX);
        let r = _mm512_mask_adds_epi8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_adds_epi8(
            a,
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_adds_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(i8::MAX);
        let r = _mm512_maskz_adds_epi8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_adds_epi8(
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_adds_epi8() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(i8::MAX);
        let r = _mm256_mask_adds_epi8(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_adds_epi8(a, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_adds_epi8() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(i8::MAX);
        let r = _mm256_maskz_adds_epi8(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_adds_epi8(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_adds_epi8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(i8::MAX);
        let r = _mm_mask_adds_epi8(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_adds_epi8(a, 0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_adds_epi8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(i8::MAX);
        let r = _mm_maskz_adds_epi8(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_adds_epi8(0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_sub_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(2);
        let r = _mm512_sub_epi16(a, b);
        let e = _mm512_set1_epi16(-1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_sub_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(2);
        let r = _mm512_mask_sub_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_sub_epi16(a, 0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                                 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_sub_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(2);
        let r = _mm512_maskz_sub_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_sub_epi16(0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
                                 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_sub_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(2);
        let r = _mm256_mask_sub_epi16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_sub_epi16(a, 0b00000000_11111111, a, b);
        let e = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_sub_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(2);
        let r = _mm256_maskz_sub_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_sub_epi16(0b00000000_11111111, a, b);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_sub_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(2);
        let r = _mm_mask_sub_epi16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_sub_epi16(a, 0b00001111, a, b);
        let e = _mm_set_epi16(1, 1, 1, 1, -1, -1, -1, -1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_sub_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(2);
        let r = _mm_maskz_sub_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_sub_epi16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, -1, -1, -1, -1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_sub_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(2);
        let r = _mm512_sub_epi8(a, b);
        let e = _mm512_set1_epi8(-1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_sub_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(2);
        let r = _mm512_mask_sub_epi8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_sub_epi8(
            a,
            0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                                1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                                1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                                1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_sub_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(2);
        let r = _mm512_maskz_sub_epi8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_sub_epi8(
            0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
                                0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
                                0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
                                0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_sub_epi8() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(2);
        let r = _mm256_mask_sub_epi8(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_sub_epi8(a, 0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                                1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_sub_epi8() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(2);
        let r = _mm256_maskz_sub_epi8(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_sub_epi8(0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
                                0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_sub_epi8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(2);
        let r = _mm_mask_sub_epi8(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_sub_epi8(a, 0b00000000_11111111, a, b);
        let e = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_sub_epi8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(2);
        let r = _mm_maskz_sub_epi8(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_sub_epi8(0b00000000_11111111, a, b);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_subs_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(u16::MAX as i16);
        let r = _mm512_subs_epu16(a, b);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_subs_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(u16::MAX as i16);
        let r = _mm512_mask_subs_epu16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_subs_epu16(a, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_subs_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(u16::MAX as i16);
        let r = _mm512_maskz_subs_epu16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_subs_epu16(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_subs_epu16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(u16::MAX as i16);
        let r = _mm256_mask_subs_epu16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_subs_epu16(a, 0b00000000_00001111, a, b);
        let e = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_subs_epu16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(u16::MAX as i16);
        let r = _mm256_maskz_subs_epu16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_subs_epu16(0b00000000_00001111, a, b);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_subs_epu16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(u16::MAX as i16);
        let r = _mm_mask_subs_epu16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_subs_epu16(a, 0b00001111, a, b);
        let e = _mm_set_epi16(1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_subs_epu16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(u16::MAX as i16);
        let r = _mm_maskz_subs_epu16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_subs_epu16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_subs_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(u8::MAX as i8);
        let r = _mm512_subs_epu8(a, b);
        let e = _mm512_set1_epi8(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_subs_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(u8::MAX as i8);
        let r = _mm512_mask_subs_epu8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_subs_epu8(
            a,
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_subs_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(u8::MAX as i8);
        let r = _mm512_maskz_subs_epu8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_subs_epu8(
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_subs_epu8() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(u8::MAX as i8);
        let r = _mm256_mask_subs_epu8(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_subs_epu8(a, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_subs_epu8() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(u8::MAX as i8);
        let r = _mm256_maskz_subs_epu8(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_subs_epu8(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_subs_epu8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(u8::MAX as i8);
        let r = _mm_mask_subs_epu8(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_subs_epu8(a, 0b00000000_00001111, a, b);
        let e = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_subs_epu8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(u8::MAX as i8);
        let r = _mm_maskz_subs_epu8(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_subs_epu8(0b00000000_00001111, a, b);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_subs_epi16() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_subs_epi16(a, b);
        let e = _mm512_set1_epi16(i16::MIN);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_subs_epi16() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_mask_subs_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_subs_epi16(a, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, i16::MIN, i16::MIN, i16::MIN, i16::MIN);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_subs_epi16() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_maskz_subs_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_subs_epi16(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i16::MIN, i16::MIN, i16::MIN, i16::MIN);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_subs_epi16() {
        let a = _mm256_set1_epi16(-1);
        let b = _mm256_set1_epi16(i16::MAX);
        let r = _mm256_mask_subs_epi16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_subs_epi16(a, 0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, i16::MIN, i16::MIN, i16::MIN, i16::MIN);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_subs_epi16() {
        let a = _mm256_set1_epi16(-1);
        let b = _mm256_set1_epi16(i16::MAX);
        let r = _mm256_maskz_subs_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_subs_epi16(0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i16::MIN, i16::MIN, i16::MIN, i16::MIN);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_subs_epi16() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_set1_epi16(i16::MAX);
        let r = _mm_mask_subs_epi16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_subs_epi16(a, 0b00001111, a, b);
        let e = _mm_set_epi16(-1, -1, -1, -1, i16::MIN, i16::MIN, i16::MIN, i16::MIN);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_subs_epi16() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_set1_epi16(i16::MAX);
        let r = _mm_maskz_subs_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_subs_epi16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, i16::MIN, i16::MIN, i16::MIN, i16::MIN);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_subs_epi8() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(i8::MAX);
        let r = _mm512_subs_epi8(a, b);
        let e = _mm512_set1_epi8(i8::MIN);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_subs_epi8() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(i8::MAX);
        let r = _mm512_mask_subs_epi8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_subs_epi8(
            a,
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, i8::MIN, i8::MIN, i8::MIN, i8::MIN);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_subs_epi8() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(i8::MAX);
        let r = _mm512_maskz_subs_epi8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_subs_epi8(
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i8::MIN, i8::MIN, i8::MIN, i8::MIN);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_subs_epi8() {
        let a = _mm256_set1_epi8(-1);
        let b = _mm256_set1_epi8(i8::MAX);
        let r = _mm256_mask_subs_epi8(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_subs_epi8(a, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, i8::MIN, i8::MIN, i8::MIN, i8::MIN);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_subs_epi8() {
        let a = _mm256_set1_epi8(-1);
        let b = _mm256_set1_epi8(i8::MAX);
        let r = _mm256_maskz_subs_epi8(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_subs_epi8(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i8::MIN, i8::MIN, i8::MIN, i8::MIN);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_subs_epi8() {
        let a = _mm_set1_epi8(-1);
        let b = _mm_set1_epi8(i8::MAX);
        let r = _mm_mask_subs_epi8(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_subs_epi8(a, 0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, i8::MIN, i8::MIN, i8::MIN, i8::MIN);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_subs_epi8() {
        let a = _mm_set1_epi8(-1);
        let b = _mm_set1_epi8(i8::MAX);
        let r = _mm_maskz_subs_epi8(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_subs_epi8(0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i8::MIN, i8::MIN, i8::MIN, i8::MIN);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mulhi_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mulhi_epu16(a, b);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_mulhi_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mask_mulhi_epu16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_mulhi_epu16(a, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_mulhi_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_maskz_mulhi_epu16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_mulhi_epu16(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_mulhi_epu16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_mask_mulhi_epu16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_mulhi_epu16(a, 0b00000000_00001111, a, b);
        let e = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_mulhi_epu16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_maskz_mulhi_epu16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_mulhi_epu16(0b00000000_00001111, a, b);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_mulhi_epu16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let r = _mm_mask_mulhi_epu16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_mulhi_epu16(a, 0b00001111, a, b);
        let e = _mm_set_epi16(1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_mulhi_epu16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let r = _mm_maskz_mulhi_epu16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_mulhi_epu16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mulhi_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mulhi_epi16(a, b);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_mulhi_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mask_mulhi_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_mulhi_epi16(a, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_mulhi_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_maskz_mulhi_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_mulhi_epi16(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_mulhi_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_mask_mulhi_epi16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_mulhi_epi16(a, 0b00000000_00001111, a, b);
        let e = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_mulhi_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_maskz_mulhi_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_mulhi_epi16(0b00000000_00001111, a, b);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_mulhi_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let r = _mm_mask_mulhi_epi16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_mulhi_epi16(a, 0b00001111, a, b);
        let e = _mm_set_epi16(1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_mulhi_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let r = _mm_maskz_mulhi_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_mulhi_epi16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mulhrs_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mulhrs_epi16(a, b);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_mulhrs_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mask_mulhrs_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_mulhrs_epi16(a, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_mulhrs_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_maskz_mulhrs_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_mulhrs_epi16(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_mulhrs_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_mask_mulhrs_epi16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_mulhrs_epi16(a, 0b00000000_00001111, a, b);
        let e = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_mulhrs_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_maskz_mulhrs_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_mulhrs_epi16(0b00000000_00001111, a, b);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_mulhrs_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let r = _mm_mask_mulhrs_epi16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_mulhrs_epi16(a, 0b00001111, a, b);
        let e = _mm_set_epi16(1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_mulhrs_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let r = _mm_maskz_mulhrs_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_mulhrs_epi16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mullo_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mullo_epi16(a, b);
        let e = _mm512_set1_epi16(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_mullo_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mask_mullo_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_mullo_epi16(a, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_mullo_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_maskz_mullo_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_mullo_epi16(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_mullo_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_mask_mullo_epi16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_mullo_epi16(a, 0b00000000_00001111, a, b);
        let e = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_mullo_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_maskz_mullo_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_mullo_epi16(0b00000000_00001111, a, b);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_mullo_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let r = _mm_mask_mullo_epi16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_mullo_epi16(a, 0b00001111, a, b);
        let e = _mm_set_epi16(1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_mullo_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let r = _mm_maskz_mullo_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_mullo_epi16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, 1, 1, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_max_epu16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_max_epu16(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                 15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_max_epu16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_max_epu16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_max_epu16(a, 0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_max_epu16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_max_epu16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_max_epu16(0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_mask_max_epu16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_mask_max_epu16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_max_epu16(a, 0b00000000_11111111, a, b);
        let e = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_maskz_max_epu16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_maskz_max_epu16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_max_epu16(0b00000000_11111111, a, b);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_mask_max_epu16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_mask_max_epu16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_max_epu16(a, 0b00001111, a, b);
        let e = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_maskz_max_epu16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_maskz_max_epu16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_max_epu16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, 4, 5, 6, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_max_epu8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_max_epu8(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_max_epu8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_max_epu8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_max_epu8(
            a,
            0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_max_epu8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_max_epu8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_max_epu8(
            0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_mask_max_epu8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm256_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_mask_max_epu8(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_max_epu8(a, 0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_maskz_max_epu8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm256_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_maskz_max_epu8(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_max_epu8(0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_mask_max_epu8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_mask_max_epu8(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_max_epu8(a, 0b00000000_11111111, a, b);
        let e = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_maskz_max_epu8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_maskz_max_epu8(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_max_epu8(0b00000000_11111111, a, b);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_max_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_max_epi16(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                 15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_max_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_max_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_max_epi16(a, 0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_max_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_max_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_max_epi16(0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_mask_max_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_mask_max_epi16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_max_epi16(a, 0b00000000_11111111, a, b);
        let e = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_maskz_max_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_maskz_max_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_max_epi16(0b00000000_11111111, a, b);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_mask_max_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_mask_max_epi16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_max_epi16(a, 0b00001111, a, b);
        let e = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_maskz_max_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_maskz_max_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_max_epi16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, 4, 5, 6, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_max_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_max_epi8(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_max_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_max_epi8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_max_epi8(
            a,
            0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_max_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_max_epi8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_max_epi8(
            0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_mask_max_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm256_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_mask_max_epi8(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_max_epi8(a, 0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_maskz_max_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm256_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_maskz_max_epi8(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_max_epi8(0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_mask_max_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_mask_max_epi8(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_max_epi8(a, 0b00000000_11111111, a, b);
        let e = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_maskz_max_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_maskz_max_epi8(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_max_epi8(0b00000000_11111111, a, b);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_min_epu16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_min_epu16(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_min_epu16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_min_epu16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_min_epu16(a, 0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_min_epu16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_min_epu16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_min_epu16(0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_mask_min_epu16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_mask_min_epu16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_min_epu16(a, 0b00000000_11111111, a, b);
        let e = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_maskz_min_epu16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_maskz_min_epu16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_min_epu16(0b00000000_11111111, a, b);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_mask_min_epu16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_mask_min_epu16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_min_epu16(a, 0b00001111, a, b);
        let e = _mm_set_epi16(0, 1, 2, 3, 3, 2, 1, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_maskz_min_epu16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_maskz_min_epu16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_min_epu16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, 3, 2, 1, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_min_epu8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_min_epu8(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_min_epu8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_min_epu8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_min_epu8(
            a,
            0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_min_epu8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_min_epu8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_min_epu8(
            0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_mask_min_epu8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm256_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_mask_min_epu8(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_min_epu8(a, 0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_maskz_min_epu8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm256_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_maskz_min_epu8(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_min_epu8(0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_mask_min_epu8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_mask_min_epu8(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_min_epu8(a, 0b00000000_11111111, a, b);
        let e = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_maskz_min_epu8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_maskz_min_epu8(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_min_epu8(0b00000000_11111111, a, b);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_min_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_min_epi16(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_min_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_min_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_min_epi16(a, 0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_min_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_min_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_min_epi16(0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_mask_min_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_mask_min_epi16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_min_epi16(a, 0b00000000_11111111, a, b);
        let e = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_maskz_min_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_maskz_min_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_min_epi16(0b00000000_11111111, a, b);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_mask_min_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_mask_min_epi16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_min_epi16(a, 0b00001111, a, b);
        let e = _mm_set_epi16(0, 1, 2, 3, 3, 2, 1, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_maskz_min_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_maskz_min_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_min_epi16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, 3, 2, 1, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_min_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_min_epi8(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_min_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_min_epi8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_min_epi8(
            a,
            0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_min_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_min_epi8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_min_epi8(
            0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_mask_min_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm256_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_mask_min_epi8(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_min_epi8(a, 0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_maskz_min_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let b = _mm256_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm256_maskz_min_epi8(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_min_epi8(0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_mask_min_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_mask_min_epi8(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_min_epi8(a, 0b00000000_11111111, a, b);
        let e = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_maskz_min_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm_maskz_min_epi8(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_min_epi8(0b00000000_11111111, a, b);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmplt_epu16_mask() {
        let a = _mm512_set1_epi16(-2);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmplt_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmplt_epu16_mask() {
        let a = _mm512_set1_epi16(-2);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmplt_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmplt_epu16_mask() {
        let a = _mm256_set1_epi16(-2);
        let b = _mm256_set1_epi16(-1);
        let m = _mm256_cmplt_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmplt_epu16_mask() {
        let a = _mm256_set1_epi16(-2);
        let b = _mm256_set1_epi16(-1);
        let mask = 0b01010101_01010101;
        let r = _mm256_mask_cmplt_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmplt_epu16_mask() {
        let a = _mm_set1_epi16(-2);
        let b = _mm_set1_epi16(-1);
        let m = _mm_cmplt_epu16_mask(a, b);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmplt_epu16_mask() {
        let a = _mm_set1_epi16(-2);
        let b = _mm_set1_epi16(-1);
        let mask = 0b01010101;
        let r = _mm_mask_cmplt_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmplt_epu8_mask() {
        let a = _mm512_set1_epi8(-2);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmplt_epu8_mask(a, b);
        assert_eq!(
            m,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
        );
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmplt_epu8_mask() {
        let a = _mm512_set1_epi8(-2);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmplt_epu8_mask(mask, a, b);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmplt_epu8_mask() {
        let a = _mm256_set1_epi8(-2);
        let b = _mm256_set1_epi8(-1);
        let m = _mm256_cmplt_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmplt_epu8_mask() {
        let a = _mm256_set1_epi8(-2);
        let b = _mm256_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmplt_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmplt_epu8_mask() {
        let a = _mm_set1_epi8(-2);
        let b = _mm_set1_epi8(-1);
        let m = _mm_cmplt_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmplt_epu8_mask() {
        let a = _mm_set1_epi8(-2);
        let b = _mm_set1_epi8(-1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmplt_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmplt_epi16_mask() {
        let a = _mm512_set1_epi16(-2);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmplt_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmplt_epi16_mask() {
        let a = _mm512_set1_epi16(-2);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmplt_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmplt_epi16_mask() {
        let a = _mm256_set1_epi16(-2);
        let b = _mm256_set1_epi16(-1);
        let m = _mm256_cmplt_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmplt_epi16_mask() {
        let a = _mm256_set1_epi16(-2);
        let b = _mm256_set1_epi16(-1);
        let mask = 0b01010101_01010101;
        let r = _mm256_mask_cmplt_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmplt_epi16_mask() {
        let a = _mm_set1_epi16(-2);
        let b = _mm_set1_epi16(-1);
        let m = _mm_cmplt_epi16_mask(a, b);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmplt_epi16_mask() {
        let a = _mm_set1_epi16(-2);
        let b = _mm_set1_epi16(-1);
        let mask = 0b01010101;
        let r = _mm_mask_cmplt_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmplt_epi8_mask() {
        let a = _mm512_set1_epi8(-2);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmplt_epi8_mask(a, b);
        assert_eq!(
            m,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
        );
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmplt_epi8_mask() {
        let a = _mm512_set1_epi8(-2);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmplt_epi8_mask(mask, a, b);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmplt_epi8_mask() {
        let a = _mm256_set1_epi8(-2);
        let b = _mm256_set1_epi8(-1);
        let m = _mm256_cmplt_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmplt_epi8_mask() {
        let a = _mm256_set1_epi8(-2);
        let b = _mm256_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmplt_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmplt_epi8_mask() {
        let a = _mm_set1_epi8(-2);
        let b = _mm_set1_epi8(-1);
        let m = _mm_cmplt_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmplt_epi8_mask() {
        let a = _mm_set1_epi8(-2);
        let b = _mm_set1_epi8(-1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmplt_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpgt_epu16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(1);
        let m = _mm512_cmpgt_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpgt_epu16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpgt_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpgt_epu16_mask() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(1);
        let m = _mm256_cmpgt_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpgt_epu16_mask() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(1);
        let mask = 0b01010101_01010101;
        let r = _mm256_mask_cmpgt_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpgt_epu16_mask() {
        let a = _mm_set1_epi16(2);
        let b = _mm_set1_epi16(1);
        let m = _mm_cmpgt_epu16_mask(a, b);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpgt_epu16_mask() {
        let a = _mm_set1_epi16(2);
        let b = _mm_set1_epi16(1);
        let mask = 0b01010101;
        let r = _mm_mask_cmpgt_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpgt_epu8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmpgt_epu8_mask(a, b);
        assert_eq!(
            m,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
        );
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpgt_epu8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpgt_epu8_mask(mask, a, b);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpgt_epu8_mask() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(1);
        let m = _mm256_cmpgt_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpgt_epu8_mask() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmpgt_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpgt_epu8_mask() {
        let a = _mm_set1_epi8(2);
        let b = _mm_set1_epi8(1);
        let m = _mm_cmpgt_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpgt_epu8_mask() {
        let a = _mm_set1_epi8(2);
        let b = _mm_set1_epi8(1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmpgt_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpgt_epi16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmpgt_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpgt_epi16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpgt_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpgt_epi16_mask() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(-1);
        let m = _mm256_cmpgt_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpgt_epi16_mask() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(-1);
        let mask = 0b001010101_01010101;
        let r = _mm256_mask_cmpgt_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpgt_epi16_mask() {
        let a = _mm_set1_epi16(2);
        let b = _mm_set1_epi16(-1);
        let m = _mm_cmpgt_epi16_mask(a, b);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpgt_epi16_mask() {
        let a = _mm_set1_epi16(2);
        let b = _mm_set1_epi16(-1);
        let mask = 0b01010101;
        let r = _mm_mask_cmpgt_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpgt_epi8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmpgt_epi8_mask(a, b);
        assert_eq!(
            m,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
        );
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpgt_epi8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpgt_epi8_mask(mask, a, b);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpgt_epi8_mask() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(-1);
        let m = _mm256_cmpgt_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpgt_epi8_mask() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmpgt_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpgt_epi8_mask() {
        let a = _mm_set1_epi8(2);
        let b = _mm_set1_epi8(-1);
        let m = _mm_cmpgt_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpgt_epi8_mask() {
        let a = _mm_set1_epi8(2);
        let b = _mm_set1_epi8(-1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmpgt_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmple_epu16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmple_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmple_epu16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmple_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmple_epu16_mask() {
        let a = _mm256_set1_epi16(-1);
        let b = _mm256_set1_epi16(-1);
        let m = _mm256_cmple_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmple_epu16_mask() {
        let a = _mm256_set1_epi16(-1);
        let b = _mm256_set1_epi16(-1);
        let mask = 0b01010101_01010101;
        let r = _mm256_mask_cmple_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmple_epu16_mask() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_set1_epi16(-1);
        let m = _mm_cmple_epu16_mask(a, b);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmple_epu16_mask() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_set1_epi16(-1);
        let mask = 0b01010101;
        let r = _mm_mask_cmple_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmple_epu8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmple_epu8_mask(a, b);
        assert_eq!(
            m,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
        );
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmple_epu8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmple_epu8_mask(mask, a, b);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmple_epu8_mask() {
        let a = _mm256_set1_epi8(-1);
        let b = _mm256_set1_epi8(-1);
        let m = _mm256_cmple_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmple_epu8_mask() {
        let a = _mm256_set1_epi8(-1);
        let b = _mm256_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmple_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmple_epu8_mask() {
        let a = _mm_set1_epi8(-1);
        let b = _mm_set1_epi8(-1);
        let m = _mm_cmple_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmple_epu8_mask() {
        let a = _mm_set1_epi8(-1);
        let b = _mm_set1_epi8(-1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmple_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmple_epi16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmple_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmple_epi16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmple_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmple_epi16_mask() {
        let a = _mm256_set1_epi16(-1);
        let b = _mm256_set1_epi16(-1);
        let m = _mm256_cmple_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmple_epi16_mask() {
        let a = _mm256_set1_epi16(-1);
        let b = _mm256_set1_epi16(-1);
        let mask = 0b01010101_01010101;
        let r = _mm256_mask_cmple_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmple_epi16_mask() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_set1_epi16(-1);
        let m = _mm_cmple_epi16_mask(a, b);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmple_epi16_mask() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_set1_epi16(-1);
        let mask = 0b01010101;
        let r = _mm_mask_cmple_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmple_epi8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmple_epi8_mask(a, b);
        assert_eq!(
            m,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
        );
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmple_epi8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmple_epi8_mask(mask, a, b);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmple_epi8_mask() {
        let a = _mm256_set1_epi8(-1);
        let b = _mm256_set1_epi8(-1);
        let m = _mm256_cmple_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmple_epi8_mask() {
        let a = _mm256_set1_epi8(-1);
        let b = _mm256_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmple_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmple_epi8_mask() {
        let a = _mm_set1_epi8(-1);
        let b = _mm_set1_epi8(-1);
        let m = _mm_cmple_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmple_epi8_mask() {
        let a = _mm_set1_epi8(-1);
        let b = _mm_set1_epi8(-1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmple_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpge_epu16_mask() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let m = _mm512_cmpge_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpge_epu16_mask() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpge_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpge_epu16_mask() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let m = _mm256_cmpge_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpge_epu16_mask() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let mask = 0b01010101_01010101;
        let r = _mm256_mask_cmpge_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpge_epu16_mask() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let m = _mm_cmpge_epu16_mask(a, b);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpge_epu16_mask() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let mask = 0b01010101;
        let r = _mm_mask_cmpge_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpge_epu8_mask() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmpge_epu8_mask(a, b);
        assert_eq!(
            m,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
        );
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpge_epu8_mask() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpge_epu8_mask(mask, a, b);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpge_epu8_mask() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(1);
        let m = _mm256_cmpge_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpge_epu8_mask() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmpge_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpge_epu8_mask() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(1);
        let m = _mm_cmpge_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpge_epu8_mask() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmpge_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpge_epi16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmpge_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpge_epi16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpge_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpge_epi16_mask() {
        let a = _mm256_set1_epi16(-1);
        let b = _mm256_set1_epi16(-1);
        let m = _mm256_cmpge_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpge_epi16_mask() {
        let a = _mm256_set1_epi16(-1);
        let b = _mm256_set1_epi16(-1);
        let mask = 0b01010101_01010101;
        let r = _mm256_mask_cmpge_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpge_epi16_mask() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_set1_epi16(-1);
        let m = _mm_cmpge_epi16_mask(a, b);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpge_epi16_mask() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_set1_epi16(-1);
        let mask = 0b01010101;
        let r = _mm_mask_cmpge_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpge_epi8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmpge_epi8_mask(a, b);
        assert_eq!(
            m,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
        );
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpge_epi8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpge_epi8_mask(mask, a, b);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpge_epi8_mask() {
        let a = _mm256_set1_epi8(-1);
        let b = _mm256_set1_epi8(-1);
        let m = _mm256_cmpge_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpge_epi8_mask() {
        let a = _mm256_set1_epi8(-1);
        let b = _mm256_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmpge_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpge_epi8_mask() {
        let a = _mm_set1_epi8(-1);
        let b = _mm_set1_epi8(-1);
        let m = _mm_cmpge_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpge_epi8_mask() {
        let a = _mm_set1_epi8(-1);
        let b = _mm_set1_epi8(-1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmpge_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpeq_epu16_mask() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let m = _mm512_cmpeq_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpeq_epu16_mask() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpeq_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpeq_epu16_mask() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let m = _mm256_cmpeq_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpeq_epu16_mask() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let mask = 0b01010101_01010101;
        let r = _mm256_mask_cmpeq_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpeq_epu16_mask() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let m = _mm_cmpeq_epu16_mask(a, b);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpeq_epu16_mask() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let mask = 0b01010101;
        let r = _mm_mask_cmpeq_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpeq_epu8_mask() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmpeq_epu8_mask(a, b);
        assert_eq!(
            m,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
        );
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpeq_epu8_mask() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpeq_epu8_mask(mask, a, b);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpeq_epu8_mask() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(1);
        let m = _mm256_cmpeq_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpeq_epu8_mask() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmpeq_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpeq_epu8_mask() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(1);
        let m = _mm_cmpeq_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpeq_epu8_mask() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmpeq_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpeq_epi16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmpeq_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpeq_epi16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpeq_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpeq_epi16_mask() {
        let a = _mm256_set1_epi16(-1);
        let b = _mm256_set1_epi16(-1);
        let m = _mm256_cmpeq_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpeq_epi16_mask() {
        let a = _mm256_set1_epi16(-1);
        let b = _mm256_set1_epi16(-1);
        let mask = 0b01010101_01010101;
        let r = _mm256_mask_cmpeq_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpeq_epi16_mask() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_set1_epi16(-1);
        let m = _mm_cmpeq_epi16_mask(a, b);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpeq_epi16_mask() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_set1_epi16(-1);
        let mask = 0b01010101;
        let r = _mm_mask_cmpeq_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpeq_epi8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmpeq_epi8_mask(a, b);
        assert_eq!(
            m,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
        );
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpeq_epi8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpeq_epi8_mask(mask, a, b);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpeq_epi8_mask() {
        let a = _mm256_set1_epi8(-1);
        let b = _mm256_set1_epi8(-1);
        let m = _mm256_cmpeq_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpeq_epi8_mask() {
        let a = _mm256_set1_epi8(-1);
        let b = _mm256_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmpeq_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpeq_epi8_mask() {
        let a = _mm_set1_epi8(-1);
        let b = _mm_set1_epi8(-1);
        let m = _mm_cmpeq_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpeq_epi8_mask() {
        let a = _mm_set1_epi8(-1);
        let b = _mm_set1_epi8(-1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmpeq_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpneq_epu16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(1);
        let m = _mm512_cmpneq_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpneq_epu16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpneq_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpneq_epu16_mask() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(1);
        let m = _mm256_cmpneq_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpneq_epu16_mask() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(1);
        let mask = 0b01010101_01010101;
        let r = _mm256_mask_cmpneq_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpneq_epu16_mask() {
        let a = _mm_set1_epi16(2);
        let b = _mm_set1_epi16(1);
        let m = _mm_cmpneq_epu16_mask(a, b);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpneq_epu16_mask() {
        let a = _mm_set1_epi16(2);
        let b = _mm_set1_epi16(1);
        let mask = 0b01010101;
        let r = _mm_mask_cmpneq_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpneq_epu8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmpneq_epu8_mask(a, b);
        assert_eq!(
            m,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
        );
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpneq_epu8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpneq_epu8_mask(mask, a, b);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpneq_epu8_mask() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(1);
        let m = _mm256_cmpneq_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpneq_epu8_mask() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmpneq_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpneq_epu8_mask() {
        let a = _mm_set1_epi8(2);
        let b = _mm_set1_epi8(1);
        let m = _mm_cmpneq_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpneq_epu8_mask() {
        let a = _mm_set1_epi8(2);
        let b = _mm_set1_epi8(1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmpneq_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpneq_epi16_mask() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmpneq_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpneq_epi16_mask() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpneq_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpneq_epi16_mask() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(-1);
        let m = _mm256_cmpneq_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpneq_epi16_mask() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(-1);
        let mask = 0b01010101_01010101;
        let r = _mm256_mask_cmpneq_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpneq_epi16_mask() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(-1);
        let m = _mm_cmpneq_epi16_mask(a, b);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpneq_epi16_mask() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(-1);
        let mask = 0b01010101;
        let r = _mm_mask_cmpneq_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpneq_epi8_mask() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmpneq_epi8_mask(a, b);
        assert_eq!(
            m,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
        );
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpneq_epi8_mask() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpneq_epi8_mask(mask, a, b);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmpneq_epi8_mask() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(-1);
        let m = _mm256_cmpneq_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmpneq_epi8_mask() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmpneq_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmpneq_epi8_mask() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(-1);
        let m = _mm_cmpneq_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmpneq_epi8_mask() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(-1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmpneq_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmp_epu16_mask() {
        let a = _mm512_set1_epi16(0);
        let b = _mm512_set1_epi16(1);
        let m = _mm512_cmp_epu16_mask::<_MM_CMPINT_LT>(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmp_epu16_mask() {
        let a = _mm512_set1_epi16(0);
        let b = _mm512_set1_epi16(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmp_epu16_mask::<_MM_CMPINT_LT>(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmp_epu16_mask() {
        let a = _mm256_set1_epi16(0);
        let b = _mm256_set1_epi16(1);
        let m = _mm256_cmp_epu16_mask::<_MM_CMPINT_LT>(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmp_epu16_mask() {
        let a = _mm256_set1_epi16(0);
        let b = _mm256_set1_epi16(1);
        let mask = 0b01010101_01010101;
        let r = _mm256_mask_cmp_epu16_mask::<_MM_CMPINT_LT>(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmp_epu16_mask() {
        let a = _mm_set1_epi16(0);
        let b = _mm_set1_epi16(1);
        let m = _mm_cmp_epu16_mask::<_MM_CMPINT_LT>(a, b);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmp_epu16_mask() {
        let a = _mm_set1_epi16(0);
        let b = _mm_set1_epi16(1);
        let mask = 0b01010101;
        let r = _mm_mask_cmp_epu16_mask::<_MM_CMPINT_LT>(mask, a, b);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmp_epu8_mask() {
        let a = _mm512_set1_epi8(0);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmp_epu8_mask::<_MM_CMPINT_LT>(a, b);
        assert_eq!(
            m,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
        );
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmp_epu8_mask() {
        let a = _mm512_set1_epi8(0);
        let b = _mm512_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmp_epu8_mask::<_MM_CMPINT_LT>(mask, a, b);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmp_epu8_mask() {
        let a = _mm256_set1_epi8(0);
        let b = _mm256_set1_epi8(1);
        let m = _mm256_cmp_epu8_mask::<_MM_CMPINT_LT>(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmp_epu8_mask() {
        let a = _mm256_set1_epi8(0);
        let b = _mm256_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmp_epu8_mask::<_MM_CMPINT_LT>(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmp_epu8_mask() {
        let a = _mm_set1_epi8(0);
        let b = _mm_set1_epi8(1);
        let m = _mm_cmp_epu8_mask::<_MM_CMPINT_LT>(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmp_epu8_mask() {
        let a = _mm_set1_epi8(0);
        let b = _mm_set1_epi8(1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmp_epu8_mask::<_MM_CMPINT_LT>(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmp_epi16_mask() {
        let a = _mm512_set1_epi16(0);
        let b = _mm512_set1_epi16(1);
        let m = _mm512_cmp_epi16_mask::<_MM_CMPINT_LT>(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmp_epi16_mask() {
        let a = _mm512_set1_epi16(0);
        let b = _mm512_set1_epi16(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmp_epi16_mask::<_MM_CMPINT_LT>(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmp_epi16_mask() {
        let a = _mm256_set1_epi16(0);
        let b = _mm256_set1_epi16(1);
        let m = _mm256_cmp_epi16_mask::<_MM_CMPINT_LT>(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmp_epi16_mask() {
        let a = _mm256_set1_epi16(0);
        let b = _mm256_set1_epi16(1);
        let mask = 0b01010101_01010101;
        let r = _mm256_mask_cmp_epi16_mask::<_MM_CMPINT_LT>(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmp_epi16_mask() {
        let a = _mm_set1_epi16(0);
        let b = _mm_set1_epi16(1);
        let m = _mm_cmp_epi16_mask::<_MM_CMPINT_LT>(a, b);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmp_epi16_mask() {
        let a = _mm_set1_epi16(0);
        let b = _mm_set1_epi16(1);
        let mask = 0b01010101;
        let r = _mm_mask_cmp_epi16_mask::<_MM_CMPINT_LT>(mask, a, b);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmp_epi8_mask() {
        let a = _mm512_set1_epi8(0);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmp_epi8_mask::<_MM_CMPINT_LT>(a, b);
        assert_eq!(
            m,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111
        );
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmp_epi8_mask() {
        let a = _mm512_set1_epi8(0);
        let b = _mm512_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmp_epi8_mask::<_MM_CMPINT_LT>(mask, a, b);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmp_epi8_mask() {
        let a = _mm256_set1_epi8(0);
        let b = _mm256_set1_epi8(1);
        let m = _mm256_cmp_epi8_mask::<_MM_CMPINT_LT>(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmp_epi8_mask() {
        let a = _mm256_set1_epi8(0);
        let b = _mm256_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmp_epi8_mask::<_MM_CMPINT_LT>(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmp_epi8_mask() {
        let a = _mm_set1_epi8(0);
        let b = _mm_set1_epi8(1);
        let m = _mm_cmp_epi8_mask::<_MM_CMPINT_LT>(a, b);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmp_epi8_mask() {
        let a = _mm_set1_epi8(0);
        let b = _mm_set1_epi8(1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmp_epi8_mask::<_MM_CMPINT_LT>(mask, a, b);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_add_epi16() {
        let a = _mm256_set1_epi16(1);
        let e = _mm256_reduce_add_epi16(a);
        assert_eq!(16, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_add_epi16() {
        let a = _mm256_set1_epi16(1);
        let e = _mm256_mask_reduce_add_epi16(0b11111111_00000000, a);
        assert_eq!(8, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_add_epi16() {
        let a = _mm_set1_epi16(1);
        let e = _mm_reduce_add_epi16(a);
        assert_eq!(8, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_add_epi16() {
        let a = _mm_set1_epi16(1);
        let e = _mm_mask_reduce_add_epi16(0b11110000, a);
        assert_eq!(4, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_add_epi8() {
        let a = _mm256_set1_epi8(1);
        let e = _mm256_reduce_add_epi8(a);
        assert_eq!(32, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_add_epi8() {
        let a = _mm256_set1_epi8(1);
        let e = _mm256_mask_reduce_add_epi8(0b11111111_00000000_11111111_00000000, a);
        assert_eq!(16, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_add_epi8() {
        let a = _mm_set1_epi8(1);
        let e = _mm_reduce_add_epi8(a);
        assert_eq!(16, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_add_epi8() {
        let a = _mm_set1_epi8(1);
        let e = _mm_mask_reduce_add_epi8(0b11111111_00000000, a);
        assert_eq!(8, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_and_epi16() {
        let a = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);
        let e = _mm256_reduce_and_epi16(a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_and_epi16() {
        let a = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);
        let e = _mm256_mask_reduce_and_epi16(0b11111111_00000000, a);
        assert_eq!(1, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_and_epi16() {
        let a = _mm_set_epi16(1, 1, 1, 1, 2, 2, 2, 2);
        let e = _mm_reduce_and_epi16(a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_and_epi16() {
        let a = _mm_set_epi16(1, 1, 1, 1, 2, 2, 2, 2);
        let e = _mm_mask_reduce_and_epi16(0b11110000, a);
        assert_eq!(1, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_and_epi8() {
        let a = _mm256_set_epi8(
            1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
            2, 2, 2,
        );
        let e = _mm256_reduce_and_epi8(a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_and_epi8() {
        let a = _mm256_set_epi8(
            1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
            2, 2, 2,
        );
        let e = _mm256_mask_reduce_and_epi8(0b11111111_00000000_11111111_00000000, a);
        assert_eq!(1, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_and_epi8() {
        let a = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);
        let e = _mm_reduce_and_epi8(a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_and_epi8() {
        let a = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);
        let e = _mm_mask_reduce_and_epi8(0b11111111_00000000, a);
        assert_eq!(1, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_mul_epi16() {
        let a = _mm256_set_epi16(2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1);
        let e = _mm256_reduce_mul_epi16(a);
        assert_eq!(256, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_mul_epi16() {
        let a = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);
        let e = _mm256_mask_reduce_mul_epi16(0b11111111_00000000, a);
        assert_eq!(1, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_mul_epi16() {
        let a = _mm_set_epi16(2, 2, 2, 2, 1, 1, 1, 1);
        let e = _mm_reduce_mul_epi16(a);
        assert_eq!(16, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_mul_epi16() {
        let a = _mm_set_epi16(1, 1, 1, 1, 2, 2, 2, 2);
        let e = _mm_mask_reduce_mul_epi16(0b11110000, a);
        assert_eq!(1, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_mul_epi8() {
        let a = _mm256_set_epi8(
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2,
        );
        let e = _mm256_reduce_mul_epi8(a);
        assert_eq!(64, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_mul_epi8() {
        let a = _mm256_set_epi8(
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2,
        );
        let e = _mm256_mask_reduce_mul_epi8(0b11111111_00000000_11111111_00000000, a);
        assert_eq!(1, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_mul_epi8() {
        let a = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2);
        let e = _mm_reduce_mul_epi8(a);
        assert_eq!(8, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_mul_epi8() {
        let a = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2);
        let e = _mm_mask_reduce_mul_epi8(0b11111111_00000000, a);
        assert_eq!(1, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_max_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: i16 = _mm256_reduce_max_epi16(a);
        assert_eq!(15, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_max_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: i16 = _mm256_mask_reduce_max_epi16(0b11111111_00000000, a);
        assert_eq!(7, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_max_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i16 = _mm_reduce_max_epi16(a);
        assert_eq!(7, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_max_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i16 = _mm_mask_reduce_max_epi16(0b11110000, a);
        assert_eq!(3, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_max_epi8() {
        let a = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        let e: i8 = _mm256_reduce_max_epi8(a);
        assert_eq!(31, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_max_epi8() {
        let a = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        let e: i8 = _mm256_mask_reduce_max_epi8(0b1111111111111111_0000000000000000, a);
        assert_eq!(15, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_max_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: i8 = _mm_reduce_max_epi8(a);
        assert_eq!(15, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_max_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: i8 = _mm_mask_reduce_max_epi8(0b11111111_00000000, a);
        assert_eq!(7, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_max_epu16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u16 = _mm256_reduce_max_epu16(a);
        assert_eq!(15, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_max_epu16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u16 = _mm256_mask_reduce_max_epu16(0b11111111_00000000, a);
        assert_eq!(7, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_max_epu16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16 = _mm_reduce_max_epu16(a);
        assert_eq!(7, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_max_epu16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16 = _mm_mask_reduce_max_epu16(0b11110000, a);
        assert_eq!(3, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_max_epu8() {
        let a = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        let e: u8 = _mm256_reduce_max_epu8(a);
        assert_eq!(31, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_max_epu8() {
        let a = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        let e: u8 = _mm256_mask_reduce_max_epu8(0b1111111111111111_0000000000000000, a);
        assert_eq!(15, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_max_epu8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8 = _mm_reduce_max_epu8(a);
        assert_eq!(15, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_max_epu8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8 = _mm_mask_reduce_max_epu8(0b11111111_00000000, a);
        assert_eq!(7, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_min_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: i16 = _mm256_reduce_min_epi16(a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_min_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: i16 = _mm256_mask_reduce_min_epi16(0b11111111_00000000, a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_min_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i16 = _mm_reduce_min_epi16(a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_min_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i16 = _mm_mask_reduce_min_epi16(0b11110000, a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_min_epi8() {
        let a = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        let e: i8 = _mm256_reduce_min_epi8(a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_min_epi8() {
        let a = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        let e: i8 = _mm256_mask_reduce_min_epi8(0b1111111111111111_0000000000000000, a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_min_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: i8 = _mm_reduce_min_epi8(a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_min_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: i8 = _mm_mask_reduce_min_epi8(0b11111111_00000000, a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_min_epu16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u16 = _mm256_reduce_min_epu16(a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_min_epu16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u16 = _mm256_mask_reduce_min_epu16(0b11111111_00000000, a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_min_epu16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16 = _mm_reduce_min_epu16(a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_min_epu16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16 = _mm_mask_reduce_min_epu16(0b11110000, a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_min_epu8() {
        let a = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        let e: u8 = _mm256_reduce_min_epu8(a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_min_epu8() {
        let a = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        let e: u8 = _mm256_mask_reduce_min_epu8(0b1111111111111111_0000000000000000, a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_min_epu8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8 = _mm_reduce_min_epu8(a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_min_epu8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8 = _mm_mask_reduce_min_epu8(0b11111111_00000000, a);
        assert_eq!(0, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_or_epi16() {
        let a = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);
        let e = _mm256_reduce_or_epi16(a);
        assert_eq!(3, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_or_epi16() {
        let a = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);
        let e = _mm256_mask_reduce_or_epi16(0b11111111_00000000, a);
        assert_eq!(1, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_or_epi16() {
        let a = _mm_set_epi16(1, 1, 1, 1, 2, 2, 2, 2);
        let e = _mm_reduce_or_epi16(a);
        assert_eq!(3, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_or_epi16() {
        let a = _mm_set_epi16(1, 1, 1, 1, 2, 2, 2, 2);
        let e = _mm_mask_reduce_or_epi16(0b11110000, a);
        assert_eq!(1, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_reduce_or_epi8() {
        let a = _mm256_set_epi8(
            1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
            2, 2, 2,
        );
        let e = _mm256_reduce_or_epi8(a);
        assert_eq!(3, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_reduce_or_epi8() {
        let a = _mm256_set_epi8(
            1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
            2, 2, 2,
        );
        let e = _mm256_mask_reduce_or_epi8(0b11111111_00000000_11111111_00000000, a);
        assert_eq!(1, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_reduce_or_epi8() {
        let a = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);
        let e = _mm_reduce_or_epi8(a);
        assert_eq!(3, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_reduce_or_epi8() {
        let a = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);
        let e = _mm_mask_reduce_or_epi8(0b11111111_00000000, a);
        assert_eq!(1, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_loadu_epi16() {
        #[rustfmt::skip]
        let a: [i16; 32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
        let r = _mm512_loadu_epi16(&a[0]);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_loadu_epi16() {
        let a: [i16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let r = _mm256_loadu_epi16(&a[0]);
        let e = _mm256_set_epi16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_loadu_epi16() {
        let a: [i16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let r = _mm_loadu_epi16(&a[0]);
        let e = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_loadu_epi8() {
        #[rustfmt::skip]
        let a: [i8; 64] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                           1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
        let r = _mm512_loadu_epi8(&a[0]);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_loadu_epi8() {
        #[rustfmt::skip]
        let a: [i8; 32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
        let r = _mm256_loadu_epi8(&a[0]);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_loadu_epi8() {
        let a: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let r = _mm_loadu_epi8(&a[0]);
        let e = _mm_set_epi8(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_storeu_epi16() {
        let a = _mm512_set1_epi16(9);
        let mut r = _mm512_undefined_epi32();
        _mm512_storeu_epi16(&mut r as *mut _ as *mut i16, a);
        assert_eq_m512i(r, a);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_storeu_epi16() {
        let a = _mm256_set1_epi16(9);
        let mut r = _mm256_set1_epi32(0);
        _mm256_storeu_epi16(&mut r as *mut _ as *mut i16, a);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_storeu_epi16() {
        let a = _mm_set1_epi16(9);
        let mut r = _mm_set1_epi32(0);
        _mm_storeu_epi16(&mut r as *mut _ as *mut i16, a);
        assert_eq_m128i(r, a);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_storeu_epi8() {
        let a = _mm512_set1_epi8(9);
        let mut r = _mm512_undefined_epi32();
        _mm512_storeu_epi8(&mut r as *mut _ as *mut i8, a);
        assert_eq_m512i(r, a);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_storeu_epi8() {
        let a = _mm256_set1_epi8(9);
        let mut r = _mm256_set1_epi32(0);
        _mm256_storeu_epi8(&mut r as *mut _ as *mut i8, a);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_storeu_epi8() {
        let a = _mm_set1_epi8(9);
        let mut r = _mm_set1_epi32(0);
        _mm_storeu_epi8(&mut r as *mut _ as *mut i8, a);
        assert_eq_m128i(r, a);
    }

    #[simd_test(enable = "avx512f,avx512bw")]
    unsafe fn test_mm512_mask_loadu_epi16() {
        let src = _mm512_set1_epi16(42);
        let a = &[
            1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let p = a.as_ptr();
        let m = 0b10101010_11001100_11101000_11001010;
        let r = _mm512_mask_loadu_epi16(src, m, black_box(p));
        let e = &[
            42_i16, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16, 42, 42, 19, 20, 42, 42,
            23, 24, 42, 26, 42, 28, 42, 30, 42, 32,
        ];
        let e = _mm512_loadu_epi16(e.as_ptr());
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512bw")]
    unsafe fn test_mm512_maskz_loadu_epi16() {
        let a = &[
            1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let p = a.as_ptr();
        let m = 0b10101010_11001100_11101000_11001010;
        let r = _mm512_maskz_loadu_epi16(m, black_box(p));
        let e = &[
            0_i16, 2, 0, 4, 0, 0, 7, 8, 0, 0, 0, 12, 0, 14, 15, 16, 0, 0, 19, 20, 0, 0, 23, 24, 0,
            26, 0, 28, 0, 30, 0, 32,
        ];
        let e = _mm512_loadu_epi16(e.as_ptr());
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512bw")]
    unsafe fn test_mm512_mask_storeu_epi16() {
        let mut r = [42_i16; 32];
        let a = &[
            1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let a = _mm512_loadu_epi16(a.as_ptr());
        let m = 0b10101010_11001100_11101000_11001010;
        _mm512_mask_storeu_epi16(r.as_mut_ptr(), m, a);
        let e = &[
            42_i16, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16, 42, 42, 19, 20, 42, 42,
            23, 24, 42, 26, 42, 28, 42, 30, 42, 32,
        ];
        let e = _mm512_loadu_epi16(e.as_ptr());
        assert_eq_m512i(_mm512_loadu_epi16(r.as_ptr()), e);
    }

    #[simd_test(enable = "avx512f,avx512bw")]
    unsafe fn test_mm512_mask_loadu_epi8() {
        let src = _mm512_set1_epi8(42);
        let a = &[
            1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ];
        let p = a.as_ptr();
        let m = 0b00000000_11111111_11111111_00000000_10101010_11001100_11101000_11001010;
        let r = _mm512_mask_loadu_epi8(src, m, black_box(p));
        let e = &[
            42_i8, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16, 42, 42, 19, 20, 42, 42,
            23, 24, 42, 26, 42, 28, 42, 30, 42, 32, 42, 42, 42, 42, 42, 42, 42, 42, 41, 42, 43, 44,
            45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 42, 42, 42, 42, 42, 42, 42, 42,
        ];
        let e = _mm512_loadu_epi8(e.as_ptr());
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512bw")]
    unsafe fn test_mm512_maskz_loadu_epi8() {
        let a = &[
            1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ];
        let p = a.as_ptr();
        let m = 0b00000000_11111111_11111111_00000000_10101010_11001100_11101000_11001010;
        let r = _mm512_maskz_loadu_epi8(m, black_box(p));
        let e = &[
            0_i8, 2, 0, 4, 0, 0, 7, 8, 0, 0, 0, 12, 0, 14, 15, 16, 0, 0, 19, 20, 0, 0, 23, 24, 0,
            26, 0, 28, 0, 30, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let e = _mm512_loadu_epi8(e.as_ptr());
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512bw")]
    unsafe fn test_mm512_mask_storeu_epi8() {
        let mut r = [42_i8; 64];
        let a = &[
            1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ];
        let a = _mm512_loadu_epi8(a.as_ptr());
        let m = 0b00000000_11111111_11111111_00000000_10101010_11001100_11101000_11001010;
        _mm512_mask_storeu_epi8(r.as_mut_ptr(), m, a);
        let e = &[
            42_i8, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16, 42, 42, 19, 20, 42, 42,
            23, 24, 42, 26, 42, 28, 42, 30, 42, 32, 42, 42, 42, 42, 42, 42, 42, 42, 41, 42, 43, 44,
            45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 42, 42, 42, 42, 42, 42, 42, 42,
        ];
        let e = _mm512_loadu_epi8(e.as_ptr());
        assert_eq_m512i(_mm512_loadu_epi8(r.as_ptr()), e);
    }

    #[simd_test(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_loadu_epi16() {
        let src = _mm256_set1_epi16(42);
        let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let p = a.as_ptr();
        let m = 0b11101000_11001010;
        let r = _mm256_mask_loadu_epi16(src, m, black_box(p));
        let e = &[
            42_i16, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16,
        ];
        let e = _mm256_loadu_epi16(e.as_ptr());
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_loadu_epi16() {
        let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let p = a.as_ptr();
        let m = 0b11101000_11001010;
        let r = _mm256_maskz_loadu_epi16(m, black_box(p));
        let e = &[0_i16, 2, 0, 4, 0, 0, 7, 8, 0, 0, 0, 12, 0, 14, 15, 16];
        let e = _mm256_loadu_epi16(e.as_ptr());
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_storeu_epi16() {
        let mut r = [42_i16; 16];
        let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let a = _mm256_loadu_epi16(a.as_ptr());
        let m = 0b11101000_11001010;
        _mm256_mask_storeu_epi16(r.as_mut_ptr(), m, a);
        let e = &[
            42_i16, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16,
        ];
        let e = _mm256_loadu_epi16(e.as_ptr());
        assert_eq_m256i(_mm256_loadu_epi16(r.as_ptr()), e);
    }

    #[simd_test(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_loadu_epi8() {
        let src = _mm256_set1_epi8(42);
        let a = &[
            1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let p = a.as_ptr();
        let m = 0b10101010_11001100_11101000_11001010;
        let r = _mm256_mask_loadu_epi8(src, m, black_box(p));
        let e = &[
            42_i8, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16, 42, 42, 19, 20, 42, 42,
            23, 24, 42, 26, 42, 28, 42, 30, 42, 32,
        ];
        let e = _mm256_loadu_epi8(e.as_ptr());
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_loadu_epi8() {
        let a = &[
            1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let p = a.as_ptr();
        let m = 0b10101010_11001100_11101000_11001010;
        let r = _mm256_maskz_loadu_epi8(m, black_box(p));
        let e = &[
            0_i8, 2, 0, 4, 0, 0, 7, 8, 0, 0, 0, 12, 0, 14, 15, 16, 0, 0, 19, 20, 0, 0, 23, 24, 0,
            26, 0, 28, 0, 30, 0, 32,
        ];
        let e = _mm256_loadu_epi8(e.as_ptr());
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_storeu_epi8() {
        let mut r = [42_i8; 32];
        let a = &[
            1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let a = _mm256_loadu_epi8(a.as_ptr());
        let m = 0b10101010_11001100_11101000_11001010;
        _mm256_mask_storeu_epi8(r.as_mut_ptr(), m, a);
        let e = &[
            42_i8, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16, 42, 42, 19, 20, 42, 42,
            23, 24, 42, 26, 42, 28, 42, 30, 42, 32,
        ];
        let e = _mm256_loadu_epi8(e.as_ptr());
        assert_eq_m256i(_mm256_loadu_epi8(r.as_ptr()), e);
    }

    #[simd_test(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn test_mm_mask_loadu_epi16() {
        let src = _mm_set1_epi16(42);
        let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8];
        let p = a.as_ptr();
        let m = 0b11001010;
        let r = _mm_mask_loadu_epi16(src, m, black_box(p));
        let e = &[42_i16, 2, 42, 4, 42, 42, 7, 8];
        let e = _mm_loadu_epi16(e.as_ptr());
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_loadu_epi16() {
        let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8];
        let p = a.as_ptr();
        let m = 0b11001010;
        let r = _mm_maskz_loadu_epi16(m, black_box(p));
        let e = &[0_i16, 2, 0, 4, 0, 0, 7, 8];
        let e = _mm_loadu_epi16(e.as_ptr());
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn test_mm_mask_storeu_epi16() {
        let mut r = [42_i16; 8];
        let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8];
        let a = _mm_loadu_epi16(a.as_ptr());
        let m = 0b11001010;
        _mm_mask_storeu_epi16(r.as_mut_ptr(), m, a);
        let e = &[42_i16, 2, 42, 4, 42, 42, 7, 8];
        let e = _mm_loadu_epi16(e.as_ptr());
        assert_eq_m128i(_mm_loadu_epi16(r.as_ptr()), e);
    }

    #[simd_test(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn test_mm_mask_loadu_epi8() {
        let src = _mm_set1_epi8(42);
        let a = &[1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let p = a.as_ptr();
        let m = 0b11101000_11001010;
        let r = _mm_mask_loadu_epi8(src, m, black_box(p));
        let e = &[
            42_i8, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16,
        ];
        let e = _mm_loadu_epi8(e.as_ptr());
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_loadu_epi8() {
        let a = &[1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let p = a.as_ptr();
        let m = 0b11101000_11001010;
        let r = _mm_maskz_loadu_epi8(m, black_box(p));
        let e = &[0_i8, 2, 0, 4, 0, 0, 7, 8, 0, 0, 0, 12, 0, 14, 15, 16];
        let e = _mm_loadu_epi8(e.as_ptr());
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn test_mm_mask_storeu_epi8() {
        let mut r = [42_i8; 16];
        let a = &[1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let a = _mm_loadu_epi8(a.as_ptr());
        let m = 0b11101000_11001010;
        _mm_mask_storeu_epi8(r.as_mut_ptr(), m, a);
        let e = &[
            42_i8, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16,
        ];
        let e = _mm_loadu_epi8(e.as_ptr());
        assert_eq_m128i(_mm_loadu_epi8(r.as_ptr()), e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_madd_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_madd_epi16(a, b);
        let e = _mm512_set1_epi32(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_madd_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mask_madd_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_madd_epi16(a, 0b00000000_00001111, a, b);
        let e = _mm512_set_epi32(
            1 << 16 | 1,
            1 << 16 | 1,
            1 << 16 | 1,
            1 << 16 | 1,
            1 << 16 | 1,
            1 << 16 | 1,
            1 << 16 | 1,
            1 << 16 | 1,
            1 << 16 | 1,
            1 << 16 | 1,
            1 << 16 | 1,
            1 << 16 | 1,
            2,
            2,
            2,
            2,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_madd_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_maskz_madd_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_madd_epi16(0b00000000_00001111, a, b);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_madd_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_mask_madd_epi16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_madd_epi16(a, 0b00001111, a, b);
        let e = _mm256_set_epi32(
            1 << 16 | 1,
            1 << 16 | 1,
            1 << 16 | 1,
            1 << 16 | 1,
            2,
            2,
            2,
            2,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_madd_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_maskz_madd_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_madd_epi16(0b00001111, a, b);
        let e = _mm256_set_epi32(0, 0, 0, 0, 2, 2, 2, 2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_madd_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let r = _mm_mask_madd_epi16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_madd_epi16(a, 0b00001111, a, b);
        let e = _mm_set_epi32(2, 2, 2, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_madd_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let r = _mm_maskz_madd_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_madd_epi16(0b00001111, a, b);
        let e = _mm_set_epi32(2, 2, 2, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maddubs_epi16() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(1);
        let r = _mm512_maddubs_epi16(a, b);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_maddubs_epi16() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(1);
        let src = _mm512_set1_epi16(1);
        let r = _mm512_mask_maddubs_epi16(src, 0, a, b);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_add_epi16(src, 0b00000000_00000000_00000000_00000001, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1<<9|2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_maddubs_epi16() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(1);
        let r = _mm512_maskz_maddubs_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_maddubs_epi16(0b00000000_11111111_00000000_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2,
                                 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_maddubs_epi16() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(1);
        let src = _mm256_set1_epi16(1);
        let r = _mm256_mask_maddubs_epi16(src, 0, a, b);
        assert_eq_m256i(r, src);
        let r = _mm256_mask_add_epi16(src, 0b00000000_00000001, a, b);
        let e = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 << 9 | 2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_maddubs_epi16() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(1);
        let r = _mm256_maskz_maddubs_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_maddubs_epi16(0b00000000_11111111, a, b);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_maddubs_epi16() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(1);
        let src = _mm_set1_epi16(1);
        let r = _mm_mask_maddubs_epi16(src, 0, a, b);
        assert_eq_m128i(r, src);
        let r = _mm_mask_add_epi16(src, 0b00000001, a, b);
        let e = _mm_set_epi16(1, 1, 1, 1, 1, 1, 1, 1 << 9 | 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_maddubs_epi16() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(1);
        let r = _mm_maskz_maddubs_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_maddubs_epi16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, 2, 2, 2, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_packs_epi32() {
        let a = _mm512_set1_epi32(i32::MAX);
        let b = _mm512_set1_epi32(1);
        let r = _mm512_packs_epi32(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, i16::MAX, i16::MAX, i16::MAX, i16::MAX, 1, 1, 1, 1, i16::MAX, i16::MAX, i16::MAX, i16::MAX,
                                 1, 1, 1, 1, i16::MAX, i16::MAX, i16::MAX, i16::MAX, 1, 1, 1, 1, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_packs_epi32() {
        let a = _mm512_set1_epi32(i32::MAX);
        let b = _mm512_set1_epi32(1 << 16 | 1);
        let r = _mm512_mask_packs_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_packs_epi32(b, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_packs_epi32() {
        let a = _mm512_set1_epi32(i32::MAX);
        let b = _mm512_set1_epi32(1);
        let r = _mm512_maskz_packs_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_packs_epi32(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_packs_epi32() {
        let a = _mm256_set1_epi32(i32::MAX);
        let b = _mm256_set1_epi32(1 << 16 | 1);
        let r = _mm256_mask_packs_epi32(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_packs_epi32(b, 0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_packs_epi32() {
        let a = _mm256_set1_epi32(i32::MAX);
        let b = _mm256_set1_epi32(1);
        let r = _mm256_maskz_packs_epi32(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_packs_epi32(0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_packs_epi32() {
        let a = _mm_set1_epi32(i32::MAX);
        let b = _mm_set1_epi32(1 << 16 | 1);
        let r = _mm_mask_packs_epi32(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_packs_epi32(b, 0b00001111, a, b);
        let e = _mm_set_epi16(1, 1, 1, 1, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_packs_epi32() {
        let a = _mm_set1_epi32(i32::MAX);
        let b = _mm_set1_epi32(1);
        let r = _mm_maskz_packs_epi32(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_packs_epi32(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_packs_epi16() {
        let a = _mm512_set1_epi16(i16::MAX);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_packs_epi16(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX,
                                1, 1, 1, 1, 1, 1, 1, 1, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX,
                                1, 1, 1, 1, 1, 1, 1, 1, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX,
                                1, 1, 1, 1, 1, 1, 1, 1, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_packs_epi16() {
        let a = _mm512_set1_epi16(i16::MAX);
        let b = _mm512_set1_epi16(1 << 8 | 1);
        let r = _mm512_mask_packs_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_packs_epi16(
            b,
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_packs_epi16() {
        let a = _mm512_set1_epi16(i16::MAX);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_maskz_packs_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_packs_epi16(
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_packs_epi16() {
        let a = _mm256_set1_epi16(i16::MAX);
        let b = _mm256_set1_epi16(1 << 8 | 1);
        let r = _mm256_mask_packs_epi16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_packs_epi16(b, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_packs_epi16() {
        let a = _mm256_set1_epi16(i16::MAX);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_maskz_packs_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_packs_epi16(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_packs_epi16() {
        let a = _mm_set1_epi16(i16::MAX);
        let b = _mm_set1_epi16(1 << 8 | 1);
        let r = _mm_mask_packs_epi16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_packs_epi16(b, 0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_packs_epi16() {
        let a = _mm_set1_epi16(i16::MAX);
        let b = _mm_set1_epi16(1);
        let r = _mm_maskz_packs_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_packs_epi16(0b00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_packus_epi32() {
        let a = _mm512_set1_epi32(-1);
        let b = _mm512_set1_epi32(1);
        let r = _mm512_packus_epi32(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                                 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_packus_epi32() {
        let a = _mm512_set1_epi32(-1);
        let b = _mm512_set1_epi32(1 << 16 | 1);
        let r = _mm512_mask_packus_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_packus_epi32(b, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_packus_epi32() {
        let a = _mm512_set1_epi32(-1);
        let b = _mm512_set1_epi32(1);
        let r = _mm512_maskz_packus_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_packus_epi32(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_packus_epi32() {
        let a = _mm256_set1_epi32(-1);
        let b = _mm256_set1_epi32(1 << 16 | 1);
        let r = _mm256_mask_packus_epi32(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_packus_epi32(b, 0b00000000_00001111, a, b);
        let e = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_packus_epi32() {
        let a = _mm256_set1_epi32(-1);
        let b = _mm256_set1_epi32(1);
        let r = _mm256_maskz_packus_epi32(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_packus_epi32(0b00000000_00001111, a, b);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_packus_epi32() {
        let a = _mm_set1_epi32(-1);
        let b = _mm_set1_epi32(1 << 16 | 1);
        let r = _mm_mask_packus_epi32(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_packus_epi32(b, 0b00001111, a, b);
        let e = _mm_set_epi16(1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_packus_epi32() {
        let a = _mm_set1_epi32(-1);
        let b = _mm_set1_epi32(1);
        let r = _mm_maskz_packus_epi32(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_packus_epi32(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_packus_epi16() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_packus_epi16(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_packus_epi16() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(1 << 8 | 1);
        let r = _mm512_mask_packus_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_packus_epi16(
            b,
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_packus_epi16() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_maskz_packus_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_packus_epi16(
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_packus_epi16() {
        let a = _mm256_set1_epi16(-1);
        let b = _mm256_set1_epi16(1 << 8 | 1);
        let r = _mm256_mask_packus_epi16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_packus_epi16(b, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_packus_epi16() {
        let a = _mm256_set1_epi16(-1);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_maskz_packus_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_packus_epi16(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_packus_epi16() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_set1_epi16(1 << 8 | 1);
        let r = _mm_mask_packus_epi16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_packus_epi16(b, 0b00000000_00001111, a, b);
        let e = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_packus_epi16() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_set1_epi16(1);
        let r = _mm_maskz_packus_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_packus_epi16(0b00000000_00001111, a, b);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_avg_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_avg_epu16(a, b);
        let e = _mm512_set1_epi16(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_avg_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mask_avg_epu16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_avg_epu16(a, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_avg_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_maskz_avg_epu16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_avg_epu16(0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_avg_epu16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_mask_avg_epu16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_avg_epu16(a, 0b00000000_00001111, a, b);
        let e = _mm256_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_avg_epu16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_maskz_avg_epu16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_avg_epu16(0b00000000_00001111, a, b);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_avg_epu16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let r = _mm_mask_avg_epu16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_avg_epu16(a, 0b00001111, a, b);
        let e = _mm_set_epi16(1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_avg_epu16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(1);
        let r = _mm_maskz_avg_epu16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_avg_epu16(0b00001111, a, b);
        let e = _mm_set_epi16(0, 0, 0, 0, 1, 1, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_avg_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(1);
        let r = _mm512_avg_epu8(a, b);
        let e = _mm512_set1_epi8(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_avg_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(1);
        let r = _mm512_mask_avg_epu8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_avg_epu8(
            a,
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_avg_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(1);
        let r = _mm512_maskz_avg_epu8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_avg_epu8(
            0b00000000_000000000_00000000_00000000_00000000_0000000_00000000_00001111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_avg_epu8() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(1);
        let r = _mm256_mask_avg_epu8(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_avg_epu8(a, 0b00000000_00000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_avg_epu8() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(1);
        let r = _mm256_maskz_avg_epu8(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_avg_epu8(0b00000000_0000000_00000000_00001111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_avg_epu8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(1);
        let r = _mm_mask_avg_epu8(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_avg_epu8(a, 0b00000000_00001111, a, b);
        let e = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_avg_epu8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(1);
        let r = _mm_maskz_avg_epu8(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_avg_epu8(0b00000000_00001111, a, b);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_sll_epi16() {
        let a = _mm512_set1_epi16(1 << 15);
        let count = _mm_set1_epi16(2);
        let r = _mm512_sll_epi16(a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_sll_epi16() {
        let a = _mm512_set1_epi16(1 << 15);
        let count = _mm_set1_epi16(2);
        let r = _mm512_mask_sll_epi16(a, 0, a, count);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_sll_epi16(a, 0b11111111_11111111_11111111_11111111, a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_sll_epi16() {
        let a = _mm512_set1_epi16(1 << 15);
        let count = _mm_set1_epi16(2);
        let r = _mm512_maskz_sll_epi16(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_sll_epi16(0b11111111_11111111_11111111_11111111, a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_sll_epi16() {
        let a = _mm256_set1_epi16(1 << 15);
        let count = _mm_set1_epi16(2);
        let r = _mm256_mask_sll_epi16(a, 0, a, count);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_sll_epi16(a, 0b11111111_11111111, a, count);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_sll_epi16() {
        let a = _mm256_set1_epi16(1 << 15);
        let count = _mm_set1_epi16(2);
        let r = _mm256_maskz_sll_epi16(0, a, count);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_sll_epi16(0b11111111_11111111, a, count);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_sll_epi16() {
        let a = _mm_set1_epi16(1 << 15);
        let count = _mm_set1_epi16(2);
        let r = _mm_mask_sll_epi16(a, 0, a, count);
        assert_eq_m128i(r, a);
        let r = _mm_mask_sll_epi16(a, 0b11111111, a, count);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_sll_epi16() {
        let a = _mm_set1_epi16(1 << 15);
        let count = _mm_set1_epi16(2);
        let r = _mm_maskz_sll_epi16(0, a, count);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_sll_epi16(0b11111111, a, count);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_slli_epi16() {
        let a = _mm512_set1_epi16(1 << 15);
        let r = _mm512_slli_epi16::<1>(a);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_slli_epi16() {
        let a = _mm512_set1_epi16(1 << 15);
        let r = _mm512_mask_slli_epi16::<1>(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_slli_epi16::<1>(a, 0b11111111_11111111_11111111_11111111, a);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_slli_epi16() {
        let a = _mm512_set1_epi16(1 << 15);
        let r = _mm512_maskz_slli_epi16::<1>(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_slli_epi16::<1>(0b11111111_11111111_11111111_11111111, a);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_slli_epi16() {
        let a = _mm256_set1_epi16(1 << 15);
        let r = _mm256_mask_slli_epi16::<1>(a, 0, a);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_slli_epi16::<1>(a, 0b11111111_11111111, a);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_slli_epi16() {
        let a = _mm256_set1_epi16(1 << 15);
        let r = _mm256_maskz_slli_epi16::<1>(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_slli_epi16::<1>(0b11111111_11111111, a);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_slli_epi16() {
        let a = _mm_set1_epi16(1 << 15);
        let r = _mm_mask_slli_epi16::<1>(a, 0, a);
        assert_eq_m128i(r, a);
        let r = _mm_mask_slli_epi16::<1>(a, 0b11111111, a);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_slli_epi16() {
        let a = _mm_set1_epi16(1 << 15);
        let r = _mm_maskz_slli_epi16::<1>(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_slli_epi16::<1>(0b11111111, a);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_sllv_epi16() {
        let a = _mm512_set1_epi16(1 << 15);
        let count = _mm512_set1_epi16(2);
        let r = _mm512_sllv_epi16(a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_sllv_epi16() {
        let a = _mm512_set1_epi16(1 << 15);
        let count = _mm512_set1_epi16(2);
        let r = _mm512_mask_sllv_epi16(a, 0, a, count);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_sllv_epi16(a, 0b11111111_11111111_11111111_11111111, a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_sllv_epi16() {
        let a = _mm512_set1_epi16(1 << 15);
        let count = _mm512_set1_epi16(2);
        let r = _mm512_maskz_sllv_epi16(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_sllv_epi16(0b11111111_11111111_11111111_11111111, a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_sllv_epi16() {
        let a = _mm256_set1_epi16(1 << 15);
        let count = _mm256_set1_epi16(2);
        let r = _mm256_sllv_epi16(a, count);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_sllv_epi16() {
        let a = _mm256_set1_epi16(1 << 15);
        let count = _mm256_set1_epi16(2);
        let r = _mm256_mask_sllv_epi16(a, 0, a, count);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_sllv_epi16(a, 0b11111111_11111111, a, count);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_sllv_epi16() {
        let a = _mm256_set1_epi16(1 << 15);
        let count = _mm256_set1_epi16(2);
        let r = _mm256_maskz_sllv_epi16(0, a, count);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_sllv_epi16(0b11111111_11111111, a, count);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_sllv_epi16() {
        let a = _mm_set1_epi16(1 << 15);
        let count = _mm_set1_epi16(2);
        let r = _mm_sllv_epi16(a, count);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_sllv_epi16() {
        let a = _mm_set1_epi16(1 << 15);
        let count = _mm_set1_epi16(2);
        let r = _mm_mask_sllv_epi16(a, 0, a, count);
        assert_eq_m128i(r, a);
        let r = _mm_mask_sllv_epi16(a, 0b11111111, a, count);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_sllv_epi16() {
        let a = _mm_set1_epi16(1 << 15);
        let count = _mm_set1_epi16(2);
        let r = _mm_maskz_sllv_epi16(0, a, count);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_sllv_epi16(0b11111111, a, count);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_srl_epi16() {
        let a = _mm512_set1_epi16(1 << 1);
        let count = _mm_set1_epi16(2);
        let r = _mm512_srl_epi16(a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_srl_epi16() {
        let a = _mm512_set1_epi16(1 << 1);
        let count = _mm_set1_epi16(2);
        let r = _mm512_mask_srl_epi16(a, 0, a, count);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_srl_epi16(a, 0b11111111_11111111_11111111_11111111, a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_srl_epi16() {
        let a = _mm512_set1_epi16(1 << 1);
        let count = _mm_set1_epi16(2);
        let r = _mm512_maskz_srl_epi16(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_srl_epi16(0b11111111_11111111_11111111_11111111, a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_srl_epi16() {
        let a = _mm256_set1_epi16(1 << 1);
        let count = _mm_set1_epi16(2);
        let r = _mm256_mask_srl_epi16(a, 0, a, count);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_srl_epi16(a, 0b11111111_11111111, a, count);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_srl_epi16() {
        let a = _mm256_set1_epi16(1 << 1);
        let count = _mm_set1_epi16(2);
        let r = _mm256_maskz_srl_epi16(0, a, count);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_srl_epi16(0b11111111_11111111, a, count);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_srl_epi16() {
        let a = _mm_set1_epi16(1 << 1);
        let count = _mm_set1_epi16(2);
        let r = _mm_mask_srl_epi16(a, 0, a, count);
        assert_eq_m128i(r, a);
        let r = _mm_mask_srl_epi16(a, 0b11111111, a, count);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_srl_epi16() {
        let a = _mm_set1_epi16(1 << 1);
        let count = _mm_set1_epi16(2);
        let r = _mm_maskz_srl_epi16(0, a, count);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_srl_epi16(0b11111111, a, count);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_srli_epi16() {
        let a = _mm512_set1_epi16(1 << 1);
        let r = _mm512_srli_epi16::<2>(a);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_srli_epi16() {
        let a = _mm512_set1_epi16(1 << 1);
        let r = _mm512_mask_srli_epi16::<2>(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_srli_epi16::<2>(a, 0b11111111_11111111_11111111_11111111, a);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_srli_epi16() {
        let a = _mm512_set1_epi16(1 << 1);
        let r = _mm512_maskz_srli_epi16::<2>(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_srli_epi16::<2>(0b11111111_11111111_11111111_11111111, a);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_srli_epi16() {
        let a = _mm256_set1_epi16(1 << 1);
        let r = _mm256_mask_srli_epi16::<2>(a, 0, a);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_srli_epi16::<2>(a, 0b11111111_11111111, a);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_srli_epi16() {
        let a = _mm256_set1_epi16(1 << 1);
        let r = _mm256_maskz_srli_epi16::<2>(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_srli_epi16::<2>(0b11111111_11111111, a);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_srli_epi16() {
        let a = _mm_set1_epi16(1 << 1);
        let r = _mm_mask_srli_epi16::<2>(a, 0, a);
        assert_eq_m128i(r, a);
        let r = _mm_mask_srli_epi16::<2>(a, 0b11111111, a);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_srli_epi16() {
        let a = _mm_set1_epi16(1 << 1);
        let r = _mm_maskz_srli_epi16::<2>(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_srli_epi16::<2>(0b11111111, a);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_srlv_epi16() {
        let a = _mm512_set1_epi16(1 << 1);
        let count = _mm512_set1_epi16(2);
        let r = _mm512_srlv_epi16(a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_srlv_epi16() {
        let a = _mm512_set1_epi16(1 << 1);
        let count = _mm512_set1_epi16(2);
        let r = _mm512_mask_srlv_epi16(a, 0, a, count);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_srlv_epi16(a, 0b11111111_11111111_11111111_11111111, a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_srlv_epi16() {
        let a = _mm512_set1_epi16(1 << 1);
        let count = _mm512_set1_epi16(2);
        let r = _mm512_maskz_srlv_epi16(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_srlv_epi16(0b11111111_11111111_11111111_11111111, a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_srlv_epi16() {
        let a = _mm256_set1_epi16(1 << 1);
        let count = _mm256_set1_epi16(2);
        let r = _mm256_srlv_epi16(a, count);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_srlv_epi16() {
        let a = _mm256_set1_epi16(1 << 1);
        let count = _mm256_set1_epi16(2);
        let r = _mm256_mask_srlv_epi16(a, 0, a, count);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_srlv_epi16(a, 0b11111111_11111111, a, count);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_srlv_epi16() {
        let a = _mm256_set1_epi16(1 << 1);
        let count = _mm256_set1_epi16(2);
        let r = _mm256_maskz_srlv_epi16(0, a, count);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_srlv_epi16(0b11111111_11111111, a, count);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_srlv_epi16() {
        let a = _mm_set1_epi16(1 << 1);
        let count = _mm_set1_epi16(2);
        let r = _mm_srlv_epi16(a, count);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_srlv_epi16() {
        let a = _mm_set1_epi16(1 << 1);
        let count = _mm_set1_epi16(2);
        let r = _mm_mask_srlv_epi16(a, 0, a, count);
        assert_eq_m128i(r, a);
        let r = _mm_mask_srlv_epi16(a, 0b11111111, a, count);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_srlv_epi16() {
        let a = _mm_set1_epi16(1 << 1);
        let count = _mm_set1_epi16(2);
        let r = _mm_maskz_srlv_epi16(0, a, count);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_srlv_epi16(0b11111111, a, count);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_sra_epi16() {
        let a = _mm512_set1_epi16(8);
        let count = _mm_set1_epi16(1);
        let r = _mm512_sra_epi16(a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_sra_epi16() {
        let a = _mm512_set1_epi16(8);
        let count = _mm_set1_epi16(1);
        let r = _mm512_mask_sra_epi16(a, 0, a, count);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_sra_epi16(a, 0b11111111_11111111_11111111_11111111, a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_sra_epi16() {
        let a = _mm512_set1_epi16(8);
        let count = _mm_set1_epi16(1);
        let r = _mm512_maskz_sra_epi16(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_sra_epi16(0b11111111_11111111_11111111_11111111, a, count);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_sra_epi16() {
        let a = _mm256_set1_epi16(8);
        let count = _mm_set1_epi16(1);
        let r = _mm256_mask_sra_epi16(a, 0, a, count);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_sra_epi16(a, 0b11111111_11111111, a, count);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_sra_epi16() {
        let a = _mm256_set1_epi16(8);
        let count = _mm_set1_epi16(1);
        let r = _mm256_maskz_sra_epi16(0, a, count);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_sra_epi16(0b11111111_11111111, a, count);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_sra_epi16() {
        let a = _mm_set1_epi16(8);
        let count = _mm_set1_epi16(1);
        let r = _mm_mask_sra_epi16(a, 0, a, count);
        assert_eq_m128i(r, a);
        let r = _mm_mask_sra_epi16(a, 0b11111111, a, count);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_sra_epi16() {
        let a = _mm_set1_epi16(8);
        let count = _mm_set1_epi16(1);
        let r = _mm_maskz_sra_epi16(0, a, count);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_sra_epi16(0b11111111, a, count);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_srai_epi16() {
        let a = _mm512_set1_epi16(8);
        let r = _mm512_srai_epi16::<2>(a);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_srai_epi16() {
        let a = _mm512_set1_epi16(8);
        let r = _mm512_mask_srai_epi16::<2>(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_srai_epi16::<2>(a, 0b11111111_11111111_11111111_11111111, a);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_srai_epi16() {
        let a = _mm512_set1_epi16(8);
        let r = _mm512_maskz_srai_epi16::<2>(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_srai_epi16::<2>(0b11111111_11111111_11111111_11111111, a);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_srai_epi16() {
        let a = _mm256_set1_epi16(8);
        let r = _mm256_mask_srai_epi16::<2>(a, 0, a);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_srai_epi16::<2>(a, 0b11111111_11111111, a);
        let e = _mm256_set1_epi16(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_srai_epi16() {
        let a = _mm256_set1_epi16(8);
        let r = _mm256_maskz_srai_epi16::<2>(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_srai_epi16::<2>(0b11111111_11111111, a);
        let e = _mm256_set1_epi16(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_srai_epi16() {
        let a = _mm_set1_epi16(8);
        let r = _mm_mask_srai_epi16::<2>(a, 0, a);
        assert_eq_m128i(r, a);
        let r = _mm_mask_srai_epi16::<2>(a, 0b11111111, a);
        let e = _mm_set1_epi16(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_srai_epi16() {
        let a = _mm_set1_epi16(8);
        let r = _mm_maskz_srai_epi16::<2>(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_srai_epi16::<2>(0b11111111, a);
        let e = _mm_set1_epi16(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_srav_epi16() {
        let a = _mm512_set1_epi16(8);
        let count = _mm512_set1_epi16(2);
        let r = _mm512_srav_epi16(a, count);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_srav_epi16() {
        let a = _mm512_set1_epi16(8);
        let count = _mm512_set1_epi16(2);
        let r = _mm512_mask_srav_epi16(a, 0, a, count);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_srav_epi16(a, 0b11111111_11111111_11111111_11111111, a, count);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_srav_epi16() {
        let a = _mm512_set1_epi16(8);
        let count = _mm512_set1_epi16(2);
        let r = _mm512_maskz_srav_epi16(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_srav_epi16(0b11111111_11111111_11111111_11111111, a, count);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_srav_epi16() {
        let a = _mm256_set1_epi16(8);
        let count = _mm256_set1_epi16(2);
        let r = _mm256_srav_epi16(a, count);
        let e = _mm256_set1_epi16(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_srav_epi16() {
        let a = _mm256_set1_epi16(8);
        let count = _mm256_set1_epi16(2);
        let r = _mm256_mask_srav_epi16(a, 0, a, count);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_srav_epi16(a, 0b11111111_11111111, a, count);
        let e = _mm256_set1_epi16(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_srav_epi16() {
        let a = _mm256_set1_epi16(8);
        let count = _mm256_set1_epi16(2);
        let r = _mm256_maskz_srav_epi16(0, a, count);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_srav_epi16(0b11111111_11111111, a, count);
        let e = _mm256_set1_epi16(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_srav_epi16() {
        let a = _mm_set1_epi16(8);
        let count = _mm_set1_epi16(2);
        let r = _mm_srav_epi16(a, count);
        let e = _mm_set1_epi16(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_srav_epi16() {
        let a = _mm_set1_epi16(8);
        let count = _mm_set1_epi16(2);
        let r = _mm_mask_srav_epi16(a, 0, a, count);
        assert_eq_m128i(r, a);
        let r = _mm_mask_srav_epi16(a, 0b11111111, a, count);
        let e = _mm_set1_epi16(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_srav_epi16() {
        let a = _mm_set1_epi16(8);
        let count = _mm_set1_epi16(2);
        let r = _mm_maskz_srav_epi16(0, a, count);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_srav_epi16(0b11111111, a, count);
        let e = _mm_set1_epi16(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_permutex2var_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        #[rustfmt::skip]
        let idx = _mm512_set_epi16(1, 1<<5, 2, 1<<5, 3, 1<<5, 4, 1<<5, 5, 1<<5, 6, 1<<5, 7, 1<<5, 8, 1<<5,
                                   9, 1<<5, 10, 1<<5, 11, 1<<5, 12, 1<<5, 13, 1<<5, 14, 1<<5, 15, 1<<5, 16, 1<<5);
        let b = _mm512_set1_epi16(100);
        let r = _mm512_permutex2var_epi16(a, idx, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            30, 100, 29, 100, 28, 100, 27, 100, 26, 100, 25, 100, 24, 100, 23, 100,
            22, 100, 21, 100, 20, 100, 19, 100, 18, 100, 17, 100, 16, 100, 15, 100,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_permutex2var_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        #[rustfmt::skip]
        let idx = _mm512_set_epi16(1, 1<<5, 2, 1<<5, 3, 1<<5, 4, 1<<5, 5, 1<<5, 6, 1<<5, 7, 1<<5, 8, 1<<5,
                                   9, 1<<5, 10, 1<<5, 11, 1<<5, 12, 1<<5, 13, 1<<5, 14, 1<<5, 15, 1<<5, 16, 1<<5);
        let b = _mm512_set1_epi16(100);
        let r = _mm512_mask_permutex2var_epi16(a, 0, idx, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_permutex2var_epi16(a, 0b11111111_11111111_11111111_11111111, idx, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            30, 100, 29, 100, 28, 100, 27, 100, 26, 100, 25, 100, 24, 100, 23, 100,
            22, 100, 21, 100, 20, 100, 19, 100, 18, 100, 17, 100, 16, 100, 15, 100,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_permutex2var_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        #[rustfmt::skip]
        let idx = _mm512_set_epi16(1, 1<<5, 2, 1<<5, 3, 1<<5, 4, 1<<5, 5, 1<<5, 6, 1<<5, 7, 1<<5, 8, 1<<5,
                                   9, 1<<5, 10, 1<<5, 11, 1<<5, 12, 1<<5, 13, 1<<5, 14, 1<<5, 15, 1<<5, 16, 1<<5);
        let b = _mm512_set1_epi16(100);
        let r = _mm512_maskz_permutex2var_epi16(0, a, idx, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_permutex2var_epi16(0b11111111_11111111_11111111_11111111, a, idx, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            30, 100, 29, 100, 28, 100, 27, 100, 26, 100, 25, 100, 24, 100, 23, 100,
            22, 100, 21, 100, 20, 100, 19, 100, 18, 100, 17, 100, 16, 100, 15, 100,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask2_permutex2var_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        #[rustfmt::skip]
        let idx = _mm512_set_epi16(1, 1<<5, 2, 1<<5, 3, 1<<5, 4, 1<<5, 5, 1<<5, 6, 1<<5, 7, 1<<5, 8, 1<<5,
                                   9, 1<<5, 10, 1<<5, 11, 1<<5, 12, 1<<5, 13, 1<<5, 14, 1<<5, 15, 1<<5, 16, 1<<5);
        let b = _mm512_set1_epi16(100);
        let r = _mm512_mask2_permutex2var_epi16(a, idx, 0, b);
        assert_eq_m512i(r, idx);
        let r = _mm512_mask2_permutex2var_epi16(a, idx, 0b11111111_11111111_11111111_11111111, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            30, 100, 29, 100, 28, 100, 27, 100, 26, 100, 25, 100, 24, 100, 23, 100,
            22, 100, 21, 100, 20, 100, 19, 100, 18, 100, 17, 100, 16, 100, 15, 100,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_permutex2var_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let idx = _mm256_set_epi16(1, 1<<4, 2, 1<<4, 3, 1<<4, 4, 1<<4, 5, 1<<4, 6, 1<<4, 7, 1<<4, 8, 1<<4);
        let b = _mm256_set1_epi16(100);
        let r = _mm256_permutex2var_epi16(a, idx, b);
        let e = _mm256_set_epi16(
            14, 100, 13, 100, 12, 100, 11, 100, 10, 100, 9, 100, 8, 100, 7, 100,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_permutex2var_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let idx = _mm256_set_epi16(1, 1<<4, 2, 1<<4, 3, 1<<4, 4, 1<<4, 5, 1<<4, 6, 1<<4, 7, 1<<4, 8, 1<<4);
        let b = _mm256_set1_epi16(100);
        let r = _mm256_mask_permutex2var_epi16(a, 0, idx, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_permutex2var_epi16(a, 0b11111111_11111111, idx, b);
        let e = _mm256_set_epi16(
            14, 100, 13, 100, 12, 100, 11, 100, 10, 100, 9, 100, 8, 100, 7, 100,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_permutex2var_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let idx = _mm256_set_epi16(1, 1<<4, 2, 1<<4, 3, 1<<4, 4, 1<<4, 5, 1<<4, 6, 1<<4, 7, 1<<4, 8, 1<<4);
        let b = _mm256_set1_epi16(100);
        let r = _mm256_maskz_permutex2var_epi16(0, a, idx, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_permutex2var_epi16(0b11111111_11111111, a, idx, b);
        let e = _mm256_set_epi16(
            14, 100, 13, 100, 12, 100, 11, 100, 10, 100, 9, 100, 8, 100, 7, 100,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask2_permutex2var_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let idx = _mm256_set_epi16(1, 1<<4, 2, 1<<4, 3, 1<<4, 4, 1<<4, 5, 1<<4, 6, 1<<4, 7, 1<<4, 8, 1<<4);
        let b = _mm256_set1_epi16(100);
        let r = _mm256_mask2_permutex2var_epi16(a, idx, 0, b);
        assert_eq_m256i(r, idx);
        let r = _mm256_mask2_permutex2var_epi16(a, idx, 0b11111111_11111111, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi16(
            14, 100, 13, 100, 12, 100, 11, 100, 10, 100, 9, 100, 8, 100, 7, 100,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_permutex2var_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let idx = _mm_set_epi16(1, 1 << 3, 2, 1 << 3, 3, 1 << 3, 4, 1 << 3);
        let b = _mm_set1_epi16(100);
        let r = _mm_permutex2var_epi16(a, idx, b);
        let e = _mm_set_epi16(6, 100, 5, 100, 4, 100, 3, 100);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_permutex2var_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let idx = _mm_set_epi16(1, 1 << 3, 2, 1 << 3, 3, 1 << 3, 4, 1 << 3);
        let b = _mm_set1_epi16(100);
        let r = _mm_mask_permutex2var_epi16(a, 0, idx, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_permutex2var_epi16(a, 0b11111111, idx, b);
        let e = _mm_set_epi16(6, 100, 5, 100, 4, 100, 3, 100);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_permutex2var_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let idx = _mm_set_epi16(1, 1 << 3, 2, 1 << 3, 3, 1 << 3, 4, 1 << 3);
        let b = _mm_set1_epi16(100);
        let r = _mm_maskz_permutex2var_epi16(0, a, idx, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_permutex2var_epi16(0b11111111, a, idx, b);
        let e = _mm_set_epi16(6, 100, 5, 100, 4, 100, 3, 100);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask2_permutex2var_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let idx = _mm_set_epi16(1, 1 << 3, 2, 1 << 3, 3, 1 << 3, 4, 1 << 3);
        let b = _mm_set1_epi16(100);
        let r = _mm_mask2_permutex2var_epi16(a, idx, 0, b);
        assert_eq_m128i(r, idx);
        let r = _mm_mask2_permutex2var_epi16(a, idx, 0b11111111, b);
        let e = _mm_set_epi16(6, 100, 5, 100, 4, 100, 3, 100);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_permutexvar_epi16() {
        let idx = _mm512_set1_epi16(1);
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm512_permutexvar_epi16(idx, a);
        let e = _mm512_set1_epi16(30);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_permutexvar_epi16() {
        let idx = _mm512_set1_epi16(1);
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm512_mask_permutexvar_epi16(a, 0, idx, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_permutexvar_epi16(a, 0b11111111_11111111_11111111_11111111, idx, a);
        let e = _mm512_set1_epi16(30);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_permutexvar_epi16() {
        let idx = _mm512_set1_epi16(1);
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm512_maskz_permutexvar_epi16(0, idx, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_permutexvar_epi16(0b11111111_11111111_11111111_11111111, idx, a);
        let e = _mm512_set1_epi16(30);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_permutexvar_epi16() {
        let idx = _mm256_set1_epi16(1);
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_permutexvar_epi16(idx, a);
        let e = _mm256_set1_epi16(14);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_permutexvar_epi16() {
        let idx = _mm256_set1_epi16(1);
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_mask_permutexvar_epi16(a, 0, idx, a);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_permutexvar_epi16(a, 0b11111111_11111111, idx, a);
        let e = _mm256_set1_epi16(14);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_permutexvar_epi16() {
        let idx = _mm256_set1_epi16(1);
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_maskz_permutexvar_epi16(0, idx, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_permutexvar_epi16(0b11111111_11111111, idx, a);
        let e = _mm256_set1_epi16(14);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_permutexvar_epi16() {
        let idx = _mm_set1_epi16(1);
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_permutexvar_epi16(idx, a);
        let e = _mm_set1_epi16(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_permutexvar_epi16() {
        let idx = _mm_set1_epi16(1);
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_mask_permutexvar_epi16(a, 0, idx, a);
        assert_eq_m128i(r, a);
        let r = _mm_mask_permutexvar_epi16(a, 0b11111111, idx, a);
        let e = _mm_set1_epi16(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_permutexvar_epi16() {
        let idx = _mm_set1_epi16(1);
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_maskz_permutexvar_epi16(0, idx, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_permutexvar_epi16(0b11111111, idx, a);
        let e = _mm_set1_epi16(6);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_blend_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(2);
        let r = _mm512_mask_blend_epi16(0b11111111_00000000_11111111_00000000, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_blend_epi16() {
        let a = _mm256_set1_epi16(1);
        let b = _mm256_set1_epi16(2);
        let r = _mm256_mask_blend_epi16(0b11111111_00000000, a, b);
        let e = _mm256_set_epi16(2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_blend_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(2);
        let r = _mm_mask_blend_epi16(0b11110000, a, b);
        let e = _mm_set_epi16(2, 2, 2, 2, 1, 1, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_blend_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(2);
        let r = _mm512_mask_blend_epi8(
            0b11111111_00000000_11111111_00000000_11111111_00000000_11111111_00000000,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_blend_epi8() {
        let a = _mm256_set1_epi8(1);
        let b = _mm256_set1_epi8(2);
        let r = _mm256_mask_blend_epi8(0b11111111_00000000_11111111_00000000, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_blend_epi8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(2);
        let r = _mm_mask_blend_epi8(0b11111111_00000000, a, b);
        let e = _mm_set_epi8(2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_broadcastw_epi16() {
        let a = _mm_set_epi16(17, 18, 19, 20, 21, 22, 23, 24);
        let r = _mm512_broadcastw_epi16(a);
        let e = _mm512_set1_epi16(24);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_broadcastw_epi16() {
        let src = _mm512_set1_epi16(1);
        let a = _mm_set_epi16(17, 18, 19, 20, 21, 22, 23, 24);
        let r = _mm512_mask_broadcastw_epi16(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_broadcastw_epi16(src, 0b11111111_11111111_11111111_11111111, a);
        let e = _mm512_set1_epi16(24);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_broadcastw_epi16() {
        let a = _mm_set_epi16(17, 18, 19, 20, 21, 22, 23, 24);
        let r = _mm512_maskz_broadcastw_epi16(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_broadcastw_epi16(0b11111111_11111111_11111111_11111111, a);
        let e = _mm512_set1_epi16(24);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_broadcastw_epi16() {
        let src = _mm256_set1_epi16(1);
        let a = _mm_set_epi16(17, 18, 19, 20, 21, 22, 23, 24);
        let r = _mm256_mask_broadcastw_epi16(src, 0, a);
        assert_eq_m256i(r, src);
        let r = _mm256_mask_broadcastw_epi16(src, 0b11111111_11111111, a);
        let e = _mm256_set1_epi16(24);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_broadcastw_epi16() {
        let a = _mm_set_epi16(17, 18, 19, 20, 21, 22, 23, 24);
        let r = _mm256_maskz_broadcastw_epi16(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_broadcastw_epi16(0b11111111_11111111, a);
        let e = _mm256_set1_epi16(24);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_broadcastw_epi16() {
        let src = _mm_set1_epi16(1);
        let a = _mm_set_epi16(17, 18, 19, 20, 21, 22, 23, 24);
        let r = _mm_mask_broadcastw_epi16(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm_mask_broadcastw_epi16(src, 0b11111111, a);
        let e = _mm_set1_epi16(24);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_broadcastw_epi16() {
        let a = _mm_set_epi16(17, 18, 19, 20, 21, 22, 23, 24);
        let r = _mm_maskz_broadcastw_epi16(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_broadcastw_epi16(0b11111111, a);
        let e = _mm_set1_epi16(24);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_broadcastb_epi8() {
        let a = _mm_set_epi8(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm512_broadcastb_epi8(a);
        let e = _mm512_set1_epi8(32);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_broadcastb_epi8() {
        let src = _mm512_set1_epi8(1);
        let a = _mm_set_epi8(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm512_mask_broadcastb_epi8(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_broadcastb_epi8(
            src,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
        );
        let e = _mm512_set1_epi8(32);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_broadcastb_epi8() {
        let a = _mm_set_epi8(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm512_maskz_broadcastb_epi8(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_broadcastb_epi8(
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
        );
        let e = _mm512_set1_epi8(32);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_broadcastb_epi8() {
        let src = _mm256_set1_epi8(1);
        let a = _mm_set_epi8(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm256_mask_broadcastb_epi8(src, 0, a);
        assert_eq_m256i(r, src);
        let r = _mm256_mask_broadcastb_epi8(src, 0b11111111_11111111_11111111_11111111, a);
        let e = _mm256_set1_epi8(32);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_broadcastb_epi8() {
        let a = _mm_set_epi8(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm256_maskz_broadcastb_epi8(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_broadcastb_epi8(0b11111111_11111111_11111111_11111111, a);
        let e = _mm256_set1_epi8(32);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_broadcastb_epi8() {
        let src = _mm_set1_epi8(1);
        let a = _mm_set_epi8(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm_mask_broadcastb_epi8(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm_mask_broadcastb_epi8(src, 0b11111111_11111111, a);
        let e = _mm_set1_epi8(32);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_broadcastb_epi8() {
        let a = _mm_set_epi8(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm_maskz_broadcastb_epi8(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_broadcastb_epi8(0b11111111_11111111, a);
        let e = _mm_set1_epi8(32);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_unpackhi_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64);
        let r = _mm512_unpackhi_epi16(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(33, 1,  34, 2,  35, 3,  36, 4,  41, 9,  42, 10, 43, 11, 44, 12,
                                 49, 17, 50, 18, 51, 19, 52, 20, 57, 25, 58, 26, 59, 27, 60, 28);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_unpackhi_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64);
        let r = _mm512_mask_unpackhi_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_unpackhi_epi16(a, 0b11111111_11111111_11111111_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(33, 1,  34, 2,  35, 3,  36, 4,  41, 9,  42, 10, 43, 11, 44, 12,
                                 49, 17, 50, 18, 51, 19, 52, 20, 57, 25, 58, 26, 59, 27, 60, 28);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_unpackhi_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64);
        let r = _mm512_maskz_unpackhi_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_unpackhi_epi16(0b11111111_11111111_11111111_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(33, 1,  34, 2,  35, 3,  36, 4,  41, 9,  42, 10, 43, 11, 44, 12,
                                 49, 17, 50, 18, 51, 19, 52, 20, 57, 25, 58, 26, 59, 27, 60, 28);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_unpackhi_epi16() {
        let a = _mm256_set_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm256_set_epi16(
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        );
        let r = _mm256_mask_unpackhi_epi16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_unpackhi_epi16(a, 0b11111111_11111111, a, b);
        let e = _mm256_set_epi16(33, 1, 34, 2, 35, 3, 36, 4, 41, 9, 42, 10, 43, 11, 44, 12);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_unpackhi_epi16() {
        let a = _mm256_set_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm256_set_epi16(
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        );
        let r = _mm256_maskz_unpackhi_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_unpackhi_epi16(0b11111111_11111111, a, b);
        let e = _mm256_set_epi16(33, 1, 34, 2, 35, 3, 36, 4, 41, 9, 42, 10, 43, 11, 44, 12);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_unpackhi_epi16() {
        let a = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_set_epi16(33, 34, 35, 36, 37, 38, 39, 40);
        let r = _mm_mask_unpackhi_epi16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_unpackhi_epi16(a, 0b11111111, a, b);
        let e = _mm_set_epi16(33, 1, 34, 2, 35, 3, 36, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_unpackhi_epi16() {
        let a = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_set_epi16(33, 34, 35, 36, 37, 38, 39, 40);
        let r = _mm_maskz_unpackhi_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_unpackhi_epi16(0b11111111, a, b);
        let e = _mm_set_epi16(33, 1, 34, 2, 35, 3, 36, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_unpackhi_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
                                81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
                                97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                                113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 0);
        let r = _mm512_unpackhi_epi8(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(65, 1,  66, 2,  67, 3,  68, 4,  69, 5,  70, 6,  71, 7,  72, 8,
                                81, 17, 82, 18, 83, 19, 84, 20, 85, 21, 86, 22, 87, 23, 88, 24,
                                97, 33, 98, 34, 99, 35, 100, 36, 101, 37, 102, 38, 103, 39, 104, 40,
                                113, 49, 114, 50, 115, 51, 116, 52, 117, 53, 118, 54, 119, 55, 120, 56);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_unpackhi_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
                                81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
                                97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                                113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 0);
        let r = _mm512_mask_unpackhi_epi8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_unpackhi_epi8(
            a,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(65, 1,  66, 2,  67, 3,  68, 4,  69, 5,  70, 6,  71, 7,  72, 8,
                                81, 17, 82, 18, 83, 19, 84, 20, 85, 21, 86, 22, 87, 23, 88, 24,
                                97, 33, 98, 34, 99, 35, 100, 36, 101, 37, 102, 38, 103, 39, 104, 40,
                                113, 49, 114, 50, 115, 51, 116, 52, 117, 53, 118, 54, 119, 55, 120, 56);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_unpackhi_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
                                81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
                                97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                                113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 0);
        let r = _mm512_maskz_unpackhi_epi8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_unpackhi_epi8(
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(65, 1,  66, 2,  67, 3,  68, 4,  69, 5,  70, 6,  71, 7,  72, 8,
                                81, 17, 82, 18, 83, 19, 84, 20, 85, 21, 86, 22, 87, 23, 88, 24,
                                97, 33, 98, 34, 99, 35, 100, 36, 101, 37, 102, 38, 103, 39, 104, 40,
                                113, 49, 114, 50, 115, 51, 116, 52, 117, 53, 118, 54, 119, 55, 120, 56);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_unpackhi_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
        #[rustfmt::skip]
        let b = _mm256_set_epi8(65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
                                81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96);
        let r = _mm256_mask_unpackhi_epi8(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_unpackhi_epi8(a, 0b11111111_11111111_11111111_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(65, 1,  66, 2,  67, 3,  68, 4,  69, 5,  70, 6,  71, 7,  72, 8,
                                81, 17, 82, 18, 83, 19, 84, 20, 85, 21, 86, 22, 87, 23, 88, 24);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_unpackhi_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
        #[rustfmt::skip]
        let b = _mm256_set_epi8(65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
                                81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96);
        let r = _mm256_maskz_unpackhi_epi8(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_unpackhi_epi8(0b11111111_11111111_11111111_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(65, 1,  66, 2,  67, 3,  68, 4,  69, 5,  70, 6,  71, 7,  72, 8,
                                81, 17, 82, 18, 83, 19, 84, 20, 85, 21, 86, 22, 87, 23, 88, 24);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_unpackhi_epi8() {
        let a = _mm_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm_set_epi8(
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        );
        let r = _mm_mask_unpackhi_epi8(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_unpackhi_epi8(a, 0b11111111_11111111, a, b);
        let e = _mm_set_epi8(65, 1, 66, 2, 67, 3, 68, 4, 69, 5, 70, 6, 71, 7, 72, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_unpackhi_epi8() {
        let a = _mm_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm_set_epi8(
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        );
        let r = _mm_maskz_unpackhi_epi8(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_unpackhi_epi8(0b11111111_11111111, a, b);
        let e = _mm_set_epi8(65, 1, 66, 2, 67, 3, 68, 4, 69, 5, 70, 6, 71, 7, 72, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_unpacklo_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64);
        let r = _mm512_unpacklo_epi16(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(37, 5,  38, 6,  39, 7,  40, 8,  45, 13, 46, 14, 47, 15, 48, 16,
                                 53, 21, 54, 22, 55, 23, 56, 24, 61, 29, 62, 30, 63, 31, 64, 32);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_unpacklo_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64);
        let r = _mm512_mask_unpacklo_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_unpacklo_epi16(a, 0b11111111_11111111_11111111_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(37, 5,  38, 6,  39, 7,  40, 8,  45, 13, 46, 14, 47, 15, 48, 16,
                                 53, 21, 54, 22, 55, 23, 56, 24, 61, 29, 62, 30, 63, 31, 64, 32);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_unpacklo_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
        #[rustfmt::skip]
        let b = _mm512_set_epi16(33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64);
        let r = _mm512_maskz_unpacklo_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_unpacklo_epi16(0b11111111_11111111_11111111_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(37, 5,  38, 6,  39, 7,  40, 8,  45, 13, 46, 14, 47, 15, 48, 16,
                                 53, 21, 54, 22, 55, 23, 56, 24, 61, 29, 62, 30, 63, 31, 64, 32);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_unpacklo_epi16() {
        let a = _mm256_set_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm256_set_epi16(
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        );
        let r = _mm256_mask_unpacklo_epi16(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_unpacklo_epi16(a, 0b11111111_11111111, a, b);
        let e = _mm256_set_epi16(37, 5, 38, 6, 39, 7, 40, 8, 45, 13, 46, 14, 47, 15, 48, 16);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_unpacklo_epi16() {
        let a = _mm256_set_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm256_set_epi16(
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        );
        let r = _mm256_maskz_unpacklo_epi16(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_unpacklo_epi16(0b11111111_11111111, a, b);
        let e = _mm256_set_epi16(37, 5, 38, 6, 39, 7, 40, 8, 45, 13, 46, 14, 47, 15, 48, 16);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_unpacklo_epi16() {
        let a = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_set_epi16(33, 34, 35, 36, 37, 38, 39, 40);
        let r = _mm_mask_unpacklo_epi16(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_unpacklo_epi16(a, 0b11111111, a, b);
        let e = _mm_set_epi16(37, 5, 38, 6, 39, 7, 40, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_unpacklo_epi16() {
        let a = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_set_epi16(33, 34, 35, 36, 37, 38, 39, 40);
        let r = _mm_maskz_unpacklo_epi16(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_unpacklo_epi16(0b11111111, a, b);
        let e = _mm_set_epi16(37, 5, 38, 6, 39, 7, 40, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_unpacklo_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
                                81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
                                97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                                113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 0);
        let r = _mm512_unpacklo_epi8(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(73,  9,  74,  10, 75,  11, 76,  12, 77,  13, 78,  14, 79,  15, 80,  16,
                                89,  25, 90,  26, 91,  27, 92,  28, 93,  29, 94,  30, 95,  31, 96,  32,
                                105, 41, 106, 42, 107, 43, 108, 44, 109, 45, 110, 46, 111, 47, 112, 48,
                                121, 57, 122, 58, 123, 59, 124, 60, 125, 61, 126, 62, 127, 63, 0,   64);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_unpacklo_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
                                81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
                                97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                                113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 0);
        let r = _mm512_mask_unpacklo_epi8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_unpacklo_epi8(
            a,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(73,  9,  74,  10, 75,  11, 76,  12, 77,  13, 78,  14, 79,  15, 80,  16,
                                89,  25, 90,  26, 91,  27, 92,  28, 93,  29, 94,  30, 95,  31, 96,  32,
                                105, 41, 106, 42, 107, 43, 108, 44, 109, 45, 110, 46, 111, 47, 112, 48,
                                121, 57, 122, 58, 123, 59, 124, 60, 125, 61, 126, 62, 127, 63, 0,   64);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_unpacklo_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
                                81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
                                97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                                113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 0);
        let r = _mm512_maskz_unpacklo_epi8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_unpacklo_epi8(
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(73,  9,  74,  10, 75,  11, 76,  12, 77,  13, 78,  14, 79,  15, 80,  16,
                                89,  25, 90,  26, 91,  27, 92,  28, 93,  29, 94,  30, 95,  31, 96,  32,
                                105, 41, 106, 42, 107, 43, 108, 44, 109, 45, 110, 46, 111, 47, 112, 48,
                                121, 57, 122, 58, 123, 59, 124, 60, 125, 61, 126, 62, 127, 63, 0,   64);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_unpacklo_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
        #[rustfmt::skip]
        let b = _mm256_set_epi8(65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
                                81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96);
        let r = _mm256_mask_unpacklo_epi8(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_unpacklo_epi8(a, 0b11111111_11111111_11111111_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(73,  9,  74,  10, 75,  11, 76,  12, 77,  13, 78,  14, 79,  15, 80,  16,
                                89,  25, 90,  26, 91,  27, 92,  28, 93,  29, 94,  30, 95,  31, 96,  32);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_unpacklo_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
        #[rustfmt::skip]
        let b = _mm256_set_epi8(65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
                                81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96);
        let r = _mm256_maskz_unpacklo_epi8(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_unpacklo_epi8(0b11111111_11111111_11111111_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(73,  9,  74,  10, 75,  11, 76,  12, 77,  13, 78,  14, 79,  15, 80,  16,
                                89,  25, 90,  26, 91,  27, 92,  28, 93,  29, 94,  30, 95,  31, 96,  32);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_unpacklo_epi8() {
        let a = _mm_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm_set_epi8(
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        );
        let r = _mm_mask_unpacklo_epi8(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_unpacklo_epi8(a, 0b11111111_11111111, a, b);
        let e = _mm_set_epi8(
            73, 9, 74, 10, 75, 11, 76, 12, 77, 13, 78, 14, 79, 15, 80, 16,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_unpacklo_epi8() {
        let a = _mm_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm_set_epi8(
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        );
        let r = _mm_maskz_unpacklo_epi8(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_unpacklo_epi8(0b11111111_11111111, a, b);
        let e = _mm_set_epi8(
            73, 9, 74, 10, 75, 11, 76, 12, 77, 13, 78, 14, 79, 15, 80, 16,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_mov_epi16() {
        let src = _mm512_set1_epi16(1);
        let a = _mm512_set1_epi16(2);
        let r = _mm512_mask_mov_epi16(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_mov_epi16(src, 0b11111111_11111111_11111111_11111111, a);
        assert_eq_m512i(r, a);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_mov_epi16() {
        let a = _mm512_set1_epi16(2);
        let r = _mm512_maskz_mov_epi16(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_mov_epi16(0b11111111_11111111_11111111_11111111, a);
        assert_eq_m512i(r, a);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_mov_epi16() {
        let src = _mm256_set1_epi16(1);
        let a = _mm256_set1_epi16(2);
        let r = _mm256_mask_mov_epi16(src, 0, a);
        assert_eq_m256i(r, src);
        let r = _mm256_mask_mov_epi16(src, 0b11111111_11111111, a);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_mov_epi16() {
        let a = _mm256_set1_epi16(2);
        let r = _mm256_maskz_mov_epi16(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_mov_epi16(0b11111111_11111111, a);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_mov_epi16() {
        let src = _mm_set1_epi16(1);
        let a = _mm_set1_epi16(2);
        let r = _mm_mask_mov_epi16(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm_mask_mov_epi16(src, 0b11111111, a);
        assert_eq_m128i(r, a);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_mov_epi16() {
        let a = _mm_set1_epi16(2);
        let r = _mm_maskz_mov_epi16(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_mov_epi16(0b11111111, a);
        assert_eq_m128i(r, a);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_mov_epi8() {
        let src = _mm512_set1_epi8(1);
        let a = _mm512_set1_epi8(2);
        let r = _mm512_mask_mov_epi8(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_mov_epi8(
            src,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
        );
        assert_eq_m512i(r, a);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_mov_epi8() {
        let a = _mm512_set1_epi8(2);
        let r = _mm512_maskz_mov_epi8(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_mov_epi8(
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
        );
        assert_eq_m512i(r, a);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_mov_epi8() {
        let src = _mm256_set1_epi8(1);
        let a = _mm256_set1_epi8(2);
        let r = _mm256_mask_mov_epi8(src, 0, a);
        assert_eq_m256i(r, src);
        let r = _mm256_mask_mov_epi8(src, 0b11111111_11111111_11111111_11111111, a);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_mov_epi8() {
        let a = _mm256_set1_epi8(2);
        let r = _mm256_maskz_mov_epi8(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_mov_epi8(0b11111111_11111111_11111111_11111111, a);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_mov_epi8() {
        let src = _mm_set1_epi8(1);
        let a = _mm_set1_epi8(2);
        let r = _mm_mask_mov_epi8(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm_mask_mov_epi8(src, 0b11111111_11111111, a);
        assert_eq_m128i(r, a);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_mov_epi8() {
        let a = _mm_set1_epi8(2);
        let r = _mm_maskz_mov_epi8(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_mov_epi8(0b11111111_11111111, a);
        assert_eq_m128i(r, a);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_set1_epi16() {
        let src = _mm512_set1_epi16(2);
        let a: i16 = 11;
        let r = _mm512_mask_set1_epi16(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_set1_epi16(src, 0b11111111_11111111_11111111_11111111, a);
        let e = _mm512_set1_epi16(11);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_set1_epi16() {
        let a: i16 = 11;
        let r = _mm512_maskz_set1_epi16(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_set1_epi16(0b11111111_11111111_11111111_11111111, a);
        let e = _mm512_set1_epi16(11);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_set1_epi16() {
        let src = _mm256_set1_epi16(2);
        let a: i16 = 11;
        let r = _mm256_mask_set1_epi16(src, 0, a);
        assert_eq_m256i(r, src);
        let r = _mm256_mask_set1_epi16(src, 0b11111111_11111111, a);
        let e = _mm256_set1_epi16(11);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_set1_epi16() {
        let a: i16 = 11;
        let r = _mm256_maskz_set1_epi16(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_set1_epi16(0b11111111_11111111, a);
        let e = _mm256_set1_epi16(11);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_set1_epi16() {
        let src = _mm_set1_epi16(2);
        let a: i16 = 11;
        let r = _mm_mask_set1_epi16(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm_mask_set1_epi16(src, 0b11111111, a);
        let e = _mm_set1_epi16(11);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_set1_epi16() {
        let a: i16 = 11;
        let r = _mm_maskz_set1_epi16(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_set1_epi16(0b11111111, a);
        let e = _mm_set1_epi16(11);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_set1_epi8() {
        let src = _mm512_set1_epi8(2);
        let a: i8 = 11;
        let r = _mm512_mask_set1_epi8(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_set1_epi8(
            src,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
        );
        let e = _mm512_set1_epi8(11);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_set1_epi8() {
        let a: i8 = 11;
        let r = _mm512_maskz_set1_epi8(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_set1_epi8(
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
        );
        let e = _mm512_set1_epi8(11);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_set1_epi8() {
        let src = _mm256_set1_epi8(2);
        let a: i8 = 11;
        let r = _mm256_mask_set1_epi8(src, 0, a);
        assert_eq_m256i(r, src);
        let r = _mm256_mask_set1_epi8(src, 0b11111111_11111111_11111111_11111111, a);
        let e = _mm256_set1_epi8(11);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_set1_epi8() {
        let a: i8 = 11;
        let r = _mm256_maskz_set1_epi8(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_set1_epi8(0b11111111_11111111_11111111_11111111, a);
        let e = _mm256_set1_epi8(11);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_set1_epi8() {
        let src = _mm_set1_epi8(2);
        let a: i8 = 11;
        let r = _mm_mask_set1_epi8(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm_mask_set1_epi8(src, 0b11111111_11111111, a);
        let e = _mm_set1_epi8(11);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_set1_epi8() {
        let a: i8 = 11;
        let r = _mm_maskz_set1_epi8(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_set1_epi8(0b11111111_11111111, a);
        let e = _mm_set1_epi8(11);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_shufflelo_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            0, 1, 2, 3, 7, 6, 6, 4, 8, 9, 10, 11, 15, 14, 14, 12,
            16, 17, 18, 19, 23, 22, 22, 20, 24, 25, 26, 27, 31, 30, 30, 28,
        );
        let r = _mm512_shufflelo_epi16::<0b00_01_01_11>(a);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_shufflelo_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        let r = _mm512_mask_shufflelo_epi16::<0b00_01_01_11>(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shufflelo_epi16::<0b00_01_01_11>(
            a,
            0b11111111_11111111_11111111_11111111,
            a,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            0, 1, 2, 3, 7, 6, 6, 4, 8, 9, 10, 11, 15, 14, 14, 12,
            16, 17, 18, 19, 23, 22, 22, 20, 24, 25, 26, 27, 31, 30, 30, 28,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_shufflelo_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        let r = _mm512_maskz_shufflelo_epi16::<0b00_01_01_11>(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r =
            _mm512_maskz_shufflelo_epi16::<0b00_01_01_11>(0b11111111_11111111_11111111_11111111, a);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            0, 1, 2, 3, 7, 6, 6, 4, 8, 9, 10, 11, 15, 14, 14, 12,
            16, 17, 18, 19, 23, 22, 22, 20, 24, 25, 26, 27, 31, 30, 30, 28,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_shufflelo_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_mask_shufflelo_epi16::<0b00_01_01_11>(a, 0, a);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shufflelo_epi16::<0b00_01_01_11>(a, 0b11111111_11111111, a);
        let e = _mm256_set_epi16(0, 1, 2, 3, 7, 6, 6, 4, 8, 9, 10, 11, 15, 14, 14, 12);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_shufflelo_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_maskz_shufflelo_epi16::<0b00_01_01_11>(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shufflelo_epi16::<0b00_01_01_11>(0b11111111_11111111, a);
        let e = _mm256_set_epi16(0, 1, 2, 3, 7, 6, 6, 4, 8, 9, 10, 11, 15, 14, 14, 12);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_shufflelo_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_mask_shufflelo_epi16::<0b00_01_01_11>(a, 0, a);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shufflelo_epi16::<0b00_01_01_11>(a, 0b11111111, a);
        let e = _mm_set_epi16(0, 1, 2, 3, 7, 6, 6, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_shufflelo_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_maskz_shufflelo_epi16::<0b00_01_01_11>(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shufflelo_epi16::<0b00_01_01_11>(0b11111111, a);
        let e = _mm_set_epi16(0, 1, 2, 3, 7, 6, 6, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_shufflehi_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            3, 2, 2, 0, 4, 5, 6, 7, 11, 10, 10, 8, 12, 13, 14, 15,
            19, 18, 18, 16, 20, 21, 22, 23, 27, 26, 26, 24, 28, 29, 30, 31,
        );
        let r = _mm512_shufflehi_epi16::<0b00_01_01_11>(a);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_shufflehi_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        let r = _mm512_mask_shufflehi_epi16::<0b00_01_01_11>(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shufflehi_epi16::<0b00_01_01_11>(
            a,
            0b11111111_11111111_11111111_11111111,
            a,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            3, 2, 2, 0, 4, 5, 6, 7, 11, 10, 10, 8, 12, 13, 14, 15,
            19, 18, 18, 16, 20, 21, 22, 23, 27, 26, 26, 24, 28, 29, 30, 31,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_shufflehi_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        let r = _mm512_maskz_shufflehi_epi16::<0b00_01_01_11>(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r =
            _mm512_maskz_shufflehi_epi16::<0b00_01_01_11>(0b11111111_11111111_11111111_11111111, a);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            3, 2, 2, 0, 4, 5, 6, 7, 11, 10, 10, 8, 12, 13, 14, 15,
            19, 18, 18, 16, 20, 21, 22, 23, 27, 26, 26, 24, 28, 29, 30, 31,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_shufflehi_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_mask_shufflehi_epi16::<0b00_01_01_11>(a, 0, a);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shufflehi_epi16::<0b00_01_01_11>(a, 0b11111111_11111111, a);
        let e = _mm256_set_epi16(3, 2, 2, 0, 4, 5, 6, 7, 11, 10, 10, 8, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_shufflehi_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_maskz_shufflehi_epi16::<0b00_01_01_11>(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shufflehi_epi16::<0b00_01_01_11>(0b11111111_11111111, a);
        let e = _mm256_set_epi16(3, 2, 2, 0, 4, 5, 6, 7, 11, 10, 10, 8, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_shufflehi_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_mask_shufflehi_epi16::<0b00_01_01_11>(a, 0, a);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shufflehi_epi16::<0b00_01_01_11>(a, 0b11111111, a);
        let e = _mm_set_epi16(3, 2, 2, 0, 4, 5, 6, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_shufflehi_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_maskz_shufflehi_epi16::<0b00_01_01_11>(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shufflehi_epi16::<0b00_01_01_11>(0b11111111, a);
        let e = _mm_set_epi16(3, 2, 2, 0, 4, 5, 6, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_shuffle_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let b = _mm512_set1_epi8(1);
        let r = _mm512_shuffle_epi8(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                                30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
                                46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
                                62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_shuffle_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let b = _mm512_set1_epi8(1);
        let r = _mm512_mask_shuffle_epi8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shuffle_epi8(
            a,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                                30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
                                46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
                                62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_shuffle_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let b = _mm512_set1_epi8(1);
        let r = _mm512_maskz_shuffle_epi8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_shuffle_epi8(
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                                30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
                                46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
                                62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_shuffle_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let b = _mm256_set1_epi8(1);
        let r = _mm256_mask_shuffle_epi8(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shuffle_epi8(a, 0b11111111_11111111_11111111_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                                30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_shuffle_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let b = _mm256_set1_epi8(1);
        let r = _mm256_maskz_shuffle_epi8(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shuffle_epi8(0b11111111_11111111_11111111_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                                30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_shuffle_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm_set1_epi8(1);
        let r = _mm_mask_shuffle_epi8(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shuffle_epi8(a, 0b11111111_11111111, a, b);
        let e = _mm_set_epi8(
            14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_shuffle_epi8() {
        #[rustfmt::skip]
        let a = _mm_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15);
        let b = _mm_set1_epi8(1);
        let r = _mm_maskz_shuffle_epi8(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shuffle_epi8(0b11111111_11111111, a, b);
        let e = _mm_set_epi8(
            14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_test_epi16_mask() {
        let a = _mm512_set1_epi16(1 << 0);
        let b = _mm512_set1_epi16(1 << 0 | 1 << 1);
        let r = _mm512_test_epi16_mask(a, b);
        let e: __mmask32 = 0b11111111_11111111_11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_test_epi16_mask() {
        let a = _mm512_set1_epi16(1 << 0);
        let b = _mm512_set1_epi16(1 << 0 | 1 << 1);
        let r = _mm512_mask_test_epi16_mask(0, a, b);
        assert_eq!(r, 0);
        let r = _mm512_mask_test_epi16_mask(0b11111111_11111111_11111111_11111111, a, b);
        let e: __mmask32 = 0b11111111_11111111_11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_test_epi16_mask() {
        let a = _mm256_set1_epi16(1 << 0);
        let b = _mm256_set1_epi16(1 << 0 | 1 << 1);
        let r = _mm256_test_epi16_mask(a, b);
        let e: __mmask16 = 0b11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_test_epi16_mask() {
        let a = _mm256_set1_epi16(1 << 0);
        let b = _mm256_set1_epi16(1 << 0 | 1 << 1);
        let r = _mm256_mask_test_epi16_mask(0, a, b);
        assert_eq!(r, 0);
        let r = _mm256_mask_test_epi16_mask(0b11111111_11111111, a, b);
        let e: __mmask16 = 0b11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_test_epi16_mask() {
        let a = _mm_set1_epi16(1 << 0);
        let b = _mm_set1_epi16(1 << 0 | 1 << 1);
        let r = _mm_test_epi16_mask(a, b);
        let e: __mmask8 = 0b11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_test_epi16_mask() {
        let a = _mm_set1_epi16(1 << 0);
        let b = _mm_set1_epi16(1 << 0 | 1 << 1);
        let r = _mm_mask_test_epi16_mask(0, a, b);
        assert_eq!(r, 0);
        let r = _mm_mask_test_epi16_mask(0b11111111, a, b);
        let e: __mmask8 = 0b11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_test_epi8_mask() {
        let a = _mm512_set1_epi8(1 << 0);
        let b = _mm512_set1_epi8(1 << 0 | 1 << 1);
        let r = _mm512_test_epi8_mask(a, b);
        let e: __mmask64 =
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_test_epi8_mask() {
        let a = _mm512_set1_epi8(1 << 0);
        let b = _mm512_set1_epi8(1 << 0 | 1 << 1);
        let r = _mm512_mask_test_epi8_mask(0, a, b);
        assert_eq!(r, 0);
        let r = _mm512_mask_test_epi8_mask(
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
            b,
        );
        let e: __mmask64 =
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_test_epi8_mask() {
        let a = _mm256_set1_epi8(1 << 0);
        let b = _mm256_set1_epi8(1 << 0 | 1 << 1);
        let r = _mm256_test_epi8_mask(a, b);
        let e: __mmask32 = 0b11111111_11111111_11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_test_epi8_mask() {
        let a = _mm256_set1_epi8(1 << 0);
        let b = _mm256_set1_epi8(1 << 0 | 1 << 1);
        let r = _mm256_mask_test_epi8_mask(0, a, b);
        assert_eq!(r, 0);
        let r = _mm256_mask_test_epi8_mask(0b11111111_11111111_11111111_11111111, a, b);
        let e: __mmask32 = 0b11111111_11111111_11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_test_epi8_mask() {
        let a = _mm_set1_epi8(1 << 0);
        let b = _mm_set1_epi8(1 << 0 | 1 << 1);
        let r = _mm_test_epi8_mask(a, b);
        let e: __mmask16 = 0b11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_test_epi8_mask() {
        let a = _mm_set1_epi8(1 << 0);
        let b = _mm_set1_epi8(1 << 0 | 1 << 1);
        let r = _mm_mask_test_epi8_mask(0, a, b);
        assert_eq!(r, 0);
        let r = _mm_mask_test_epi8_mask(0b11111111_11111111, a, b);
        let e: __mmask16 = 0b11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_testn_epi16_mask() {
        let a = _mm512_set1_epi16(1 << 0);
        let b = _mm512_set1_epi16(1 << 0 | 1 << 1);
        let r = _mm512_testn_epi16_mask(a, b);
        let e: __mmask32 = 0b00000000_00000000_00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_testn_epi16_mask() {
        let a = _mm512_set1_epi16(1 << 0);
        let b = _mm512_set1_epi16(1 << 0 | 1 << 1);
        let r = _mm512_mask_testn_epi16_mask(0, a, b);
        assert_eq!(r, 0);
        let r = _mm512_mask_testn_epi16_mask(0b11111111_11111111_11111111_11111111, a, b);
        let e: __mmask32 = 0b00000000_00000000_00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_testn_epi16_mask() {
        let a = _mm256_set1_epi16(1 << 0);
        let b = _mm256_set1_epi16(1 << 0 | 1 << 1);
        let r = _mm256_testn_epi16_mask(a, b);
        let e: __mmask16 = 0b00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_testn_epi16_mask() {
        let a = _mm256_set1_epi16(1 << 0);
        let b = _mm256_set1_epi16(1 << 0 | 1 << 1);
        let r = _mm256_mask_testn_epi16_mask(0, a, b);
        assert_eq!(r, 0);
        let r = _mm256_mask_testn_epi16_mask(0b11111111_11111111, a, b);
        let e: __mmask16 = 0b00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_testn_epi16_mask() {
        let a = _mm_set1_epi16(1 << 0);
        let b = _mm_set1_epi16(1 << 0 | 1 << 1);
        let r = _mm_testn_epi16_mask(a, b);
        let e: __mmask8 = 0b00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_testn_epi16_mask() {
        let a = _mm_set1_epi16(1 << 0);
        let b = _mm_set1_epi16(1 << 0 | 1 << 1);
        let r = _mm_mask_testn_epi16_mask(0, a, b);
        assert_eq!(r, 0);
        let r = _mm_mask_testn_epi16_mask(0b11111111, a, b);
        let e: __mmask8 = 0b00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_testn_epi8_mask() {
        let a = _mm512_set1_epi8(1 << 0);
        let b = _mm512_set1_epi8(1 << 0 | 1 << 1);
        let r = _mm512_testn_epi8_mask(a, b);
        let e: __mmask64 =
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_testn_epi8_mask() {
        let a = _mm512_set1_epi8(1 << 0);
        let b = _mm512_set1_epi8(1 << 0 | 1 << 1);
        let r = _mm512_mask_testn_epi8_mask(0, a, b);
        assert_eq!(r, 0);
        let r = _mm512_mask_testn_epi8_mask(
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
            b,
        );
        let e: __mmask64 =
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_testn_epi8_mask() {
        let a = _mm256_set1_epi8(1 << 0);
        let b = _mm256_set1_epi8(1 << 0 | 1 << 1);
        let r = _mm256_testn_epi8_mask(a, b);
        let e: __mmask32 = 0b00000000_00000000_00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_testn_epi8_mask() {
        let a = _mm256_set1_epi8(1 << 0);
        let b = _mm256_set1_epi8(1 << 0 | 1 << 1);
        let r = _mm256_mask_testn_epi8_mask(0, a, b);
        assert_eq!(r, 0);
        let r = _mm256_mask_testn_epi8_mask(0b11111111_11111111_11111111_11111111, a, b);
        let e: __mmask32 = 0b00000000_00000000_00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_testn_epi8_mask() {
        let a = _mm_set1_epi8(1 << 0);
        let b = _mm_set1_epi8(1 << 0 | 1 << 1);
        let r = _mm_testn_epi8_mask(a, b);
        let e: __mmask16 = 0b00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_testn_epi8_mask() {
        let a = _mm_set1_epi8(1 << 0);
        let b = _mm_set1_epi8(1 << 0 | 1 << 1);
        let r = _mm_mask_testn_epi8_mask(0, a, b);
        assert_eq!(r, 0);
        let r = _mm_mask_testn_epi8_mask(0b11111111_11111111, a, b);
        let e: __mmask16 = 0b00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_store_mask64() {
        let a: __mmask64 =
            0b11111111_00000000_11111111_00000000_11111111_00000000_11111111_00000000;
        let mut r = 0;
        _store_mask64(&mut r, a);
        assert_eq!(r, a);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_store_mask32() {
        let a: __mmask32 = 0b11111111_00000000_11111111_00000000;
        let mut r = 0;
        _store_mask32(&mut r, a);
        assert_eq!(r, a);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_load_mask64() {
        let p: __mmask64 =
            0b11111111_00000000_11111111_00000000_11111111_00000000_11111111_00000000;
        let r = _load_mask64(&p);
        let e: __mmask64 =
            0b11111111_00000000_11111111_00000000_11111111_00000000_11111111_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_load_mask32() {
        let p: __mmask32 = 0b11111111_00000000_11111111_00000000;
        let r = _load_mask32(&p);
        let e: __mmask32 = 0b11111111_00000000_11111111_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_sad_epu8() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(4);
        let r = _mm512_sad_epu8(a, b);
        let e = _mm512_set1_epi64(16);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_dbsad_epu8() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(4);
        let r = _mm512_dbsad_epu8::<0>(a, b);
        let e = _mm512_set1_epi16(8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_dbsad_epu8() {
        let src = _mm512_set1_epi16(1);
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(4);
        let r = _mm512_mask_dbsad_epu8::<0>(src, 0, a, b);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_dbsad_epu8::<0>(src, 0b11111111_11111111_11111111_11111111, a, b);
        let e = _mm512_set1_epi16(8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_dbsad_epu8() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(4);
        let r = _mm512_maskz_dbsad_epu8::<0>(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_dbsad_epu8::<0>(0b11111111_11111111_11111111_11111111, a, b);
        let e = _mm512_set1_epi16(8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_dbsad_epu8() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_dbsad_epu8::<0>(a, b);
        let e = _mm256_set1_epi16(8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_dbsad_epu8() {
        let src = _mm256_set1_epi16(1);
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_mask_dbsad_epu8::<0>(src, 0, a, b);
        assert_eq_m256i(r, src);
        let r = _mm256_mask_dbsad_epu8::<0>(src, 0b11111111_11111111, a, b);
        let e = _mm256_set1_epi16(8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_dbsad_epu8() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_maskz_dbsad_epu8::<0>(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_dbsad_epu8::<0>(0b11111111_11111111, a, b);
        let e = _mm256_set1_epi16(8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_dbsad_epu8() {
        let a = _mm_set1_epi8(2);
        let b = _mm_set1_epi8(4);
        let r = _mm_dbsad_epu8::<0>(a, b);
        let e = _mm_set1_epi16(8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_dbsad_epu8() {
        let src = _mm_set1_epi16(1);
        let a = _mm_set1_epi8(2);
        let b = _mm_set1_epi8(4);
        let r = _mm_mask_dbsad_epu8::<0>(src, 0, a, b);
        assert_eq_m128i(r, src);
        let r = _mm_mask_dbsad_epu8::<0>(src, 0b11111111, a, b);
        let e = _mm_set1_epi16(8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_dbsad_epu8() {
        let a = _mm_set1_epi8(2);
        let b = _mm_set1_epi8(4);
        let r = _mm_maskz_dbsad_epu8::<0>(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_dbsad_epu8::<0>(0b11111111, a, b);
        let e = _mm_set1_epi16(8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_movepi16_mask() {
        let a = _mm512_set1_epi16(1 << 15);
        let r = _mm512_movepi16_mask(a);
        let e: __mmask32 = 0b11111111_11111111_11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_movepi16_mask() {
        let a = _mm256_set1_epi16(1 << 15);
        let r = _mm256_movepi16_mask(a);
        let e: __mmask16 = 0b11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_movepi16_mask() {
        let a = _mm_set1_epi16(1 << 15);
        let r = _mm_movepi16_mask(a);
        let e: __mmask8 = 0b11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_movepi8_mask() {
        let a = _mm512_set1_epi8(1 << 7);
        let r = _mm512_movepi8_mask(a);
        let e: __mmask64 =
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_movepi8_mask() {
        let a = _mm256_set1_epi8(1 << 7);
        let r = _mm256_movepi8_mask(a);
        let e: __mmask32 = 0b11111111_11111111_11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_movepi8_mask() {
        let a = _mm_set1_epi8(1 << 7);
        let r = _mm_movepi8_mask(a);
        let e: __mmask16 = 0b11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_movm_epi16() {
        let a: __mmask32 = 0b11111111_11111111_11111111_11111111;
        let r = _mm512_movm_epi16(a);
        let e = _mm512_set1_epi16(
            1 << 15
                | 1 << 14
                | 1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_movm_epi16() {
        let a: __mmask16 = 0b11111111_11111111;
        let r = _mm256_movm_epi16(a);
        let e = _mm256_set1_epi16(
            1 << 15
                | 1 << 14
                | 1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_movm_epi16() {
        let a: __mmask8 = 0b11111111;
        let r = _mm_movm_epi16(a);
        let e = _mm_set1_epi16(
            1 << 15
                | 1 << 14
                | 1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_movm_epi8() {
        let a: __mmask64 =
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111;
        let r = _mm512_movm_epi8(a);
        let e =
            _mm512_set1_epi8(1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_movm_epi8() {
        let a: __mmask32 = 0b11111111_11111111_11111111_11111111;
        let r = _mm256_movm_epi8(a);
        let e =
            _mm256_set1_epi8(1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_movm_epi8() {
        let a: __mmask16 = 0b11111111_11111111;
        let r = _mm_movm_epi8(a);
        let e =
            _mm_set1_epi8(1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_cvtmask32_u32() {
        let a: __mmask32 = 0b11001100_00110011_01100110_10011001;
        let r = _cvtmask32_u32(a);
        let e: u32 = 0b11001100_00110011_01100110_10011001;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_cvtu32_mask32() {
        let a: u32 = 0b11001100_00110011_01100110_10011001;
        let r = _cvtu32_mask32(a);
        let e: __mmask32 = 0b11001100_00110011_01100110_10011001;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kadd_mask32() {
        let a: __mmask32 = 11;
        let b: __mmask32 = 22;
        let r = _kadd_mask32(a, b);
        let e: __mmask32 = 33;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kadd_mask64() {
        let a: __mmask64 = 11;
        let b: __mmask64 = 22;
        let r = _kadd_mask64(a, b);
        let e: __mmask64 = 33;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kand_mask32() {
        let a: __mmask32 = 0b11001100_00110011_11001100_00110011;
        let b: __mmask32 = 0b11001100_00110011_11001100_00110011;
        let r = _kand_mask32(a, b);
        let e: __mmask32 = 0b11001100_00110011_11001100_00110011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kand_mask64() {
        let a: __mmask64 =
            0b11001100_00110011_11001100_00110011_11001100_00110011_11001100_00110011;
        let b: __mmask64 =
            0b11001100_00110011_11001100_00110011_11001100_00110011_11001100_00110011;
        let r = _kand_mask64(a, b);
        let e: __mmask64 =
            0b11001100_00110011_11001100_00110011_11001100_00110011_11001100_00110011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_knot_mask32() {
        let a: __mmask32 = 0b11001100_00110011_11001100_00110011;
        let r = _knot_mask32(a);
        let e: __mmask32 = 0b00110011_11001100_00110011_11001100;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_knot_mask64() {
        let a: __mmask64 =
            0b11001100_00110011_11001100_00110011_11001100_00110011_11001100_00110011;
        let r = _knot_mask64(a);
        let e: __mmask64 =
            0b00110011_11001100_00110011_11001100_00110011_11001100_00110011_11001100;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kandn_mask32() {
        let a: __mmask32 = 0b11001100_00110011_11001100_00110011;
        let b: __mmask32 = 0b11001100_00110011_11001100_00110011;
        let r = _kandn_mask32(a, b);
        let e: __mmask32 = 0b00000000_00000000_00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kandn_mask64() {
        let a: __mmask64 =
            0b11001100_00110011_11001100_00110011_11001100_00110011_11001100_00110011;
        let b: __mmask64 =
            0b11001100_00110011_11001100_00110011_11001100_00110011_11001100_00110011;
        let r = _kandn_mask64(a, b);
        let e: __mmask64 =
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kor_mask32() {
        let a: __mmask32 = 0b00110011_11001100_00110011_11001100;
        let b: __mmask32 = 0b11001100_00110011_11001100_00110011;
        let r = _kor_mask32(a, b);
        let e: __mmask32 = 0b11111111_11111111_11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kor_mask64() {
        let a: __mmask64 =
            0b00110011_11001100_00110011_11001100_00110011_11001100_00110011_11001100;
        let b: __mmask64 =
            0b11001100_00110011_11001100_00110011_11001100_00110011_11001100_00110011;
        let r = _kor_mask64(a, b);
        let e: __mmask64 =
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kxor_mask32() {
        let a: __mmask32 = 0b00110011_11001100_00110011_11001100;
        let b: __mmask32 = 0b11001100_00110011_11001100_00110011;
        let r = _kxor_mask32(a, b);
        let e: __mmask32 = 0b11111111_11111111_11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kxor_mask64() {
        let a: __mmask64 =
            0b00110011_11001100_00110011_11001100_00110011_11001100_00110011_11001100;
        let b: __mmask64 =
            0b11001100_00110011_11001100_00110011_11001100_00110011_11001100_00110011;
        let r = _kxor_mask64(a, b);
        let e: __mmask64 =
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kxnor_mask32() {
        let a: __mmask32 = 0b00110011_11001100_00110011_11001100;
        let b: __mmask32 = 0b11001100_00110011_11001100_00110011;
        let r = _kxnor_mask32(a, b);
        let e: __mmask32 = 0b00000000_00000000_00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kxnor_mask64() {
        let a: __mmask64 =
            0b00110011_11001100_00110011_11001100_00110011_11001100_00110011_11001100;
        let b: __mmask64 =
            0b11001100_00110011_11001100_00110011_11001100_00110011_11001100_00110011;
        let r = _kxnor_mask64(a, b);
        let e: __mmask64 =
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kortest_mask32_u8() {
        let a: __mmask32 = 0b0110100101101001_0110100101101001;
        let b: __mmask32 = 0b1011011010110110_1011011010110110;
        let mut all_ones: u8 = 0;
        let r = _kortest_mask32_u8(a, b, &mut all_ones);
        assert_eq!(r, 0);
        assert_eq!(all_ones, 1);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kortest_mask64_u8() {
        let a: __mmask64 = 0b0110100101101001_0110100101101001;
        let b: __mmask64 = 0b1011011010110110_1011011010110110;
        let mut all_ones: u8 = 0;
        let r = _kortest_mask64_u8(a, b, &mut all_ones);
        assert_eq!(r, 0);
        assert_eq!(all_ones, 0);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kortestc_mask32_u8() {
        let a: __mmask32 = 0b0110100101101001_0110100101101001;
        let b: __mmask32 = 0b1011011010110110_1011011010110110;
        let r = _kortestc_mask32_u8(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kortestc_mask64_u8() {
        let a: __mmask64 = 0b0110100101101001_0110100101101001;
        let b: __mmask64 = 0b1011011010110110_1011011010110110;
        let r = _kortestc_mask64_u8(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kortestz_mask32_u8() {
        let a: __mmask32 = 0b0110100101101001_0110100101101001;
        let b: __mmask32 = 0b1011011010110110_1011011010110110;
        let r = _kortestz_mask32_u8(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kortestz_mask64_u8() {
        let a: __mmask64 = 0b0110100101101001_0110100101101001;
        let b: __mmask64 = 0b1011011010110110_1011011010110110;
        let r = _kortestz_mask64_u8(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kshiftli_mask32() {
        let a: __mmask32 = 0b0110100101101001_0110100101101001;
        let r = _kshiftli_mask32::<3>(a);
        let e: __mmask32 = 0b0100101101001011_0100101101001000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kshiftli_mask64() {
        let a: __mmask64 = 0b0110100101101001_0110100101101001;
        let r = _kshiftli_mask64::<3>(a);
        let e: __mmask64 = 0b0110100101101001011_0100101101001000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kshiftri_mask32() {
        let a: __mmask32 = 0b0110100101101001_0110100101101001;
        let r = _kshiftri_mask32::<3>(a);
        let e: __mmask32 = 0b0000110100101101_0010110100101101;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_kshiftri_mask64() {
        let a: __mmask64 = 0b0110100101101001011_0100101101001000;
        let r = _kshiftri_mask64::<3>(a);
        let e: __mmask64 = 0b0110100101101001_0110100101101001;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_ktest_mask32_u8() {
        let a: __mmask32 = 0b0110100100111100_0110100100111100;
        let b: __mmask32 = 0b1001011011000011_1001011011000011;
        let mut and_not: u8 = 0;
        let r = _ktest_mask32_u8(a, b, &mut and_not);
        assert_eq!(r, 1);
        assert_eq!(and_not, 0);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_ktestc_mask32_u8() {
        let a: __mmask32 = 0b0110100100111100_0110100100111100;
        let b: __mmask32 = 0b1001011011000011_1001011011000011;
        let r = _ktestc_mask32_u8(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_ktestz_mask32_u8() {
        let a: __mmask32 = 0b0110100100111100_0110100100111100;
        let b: __mmask32 = 0b1001011011000011_1001011011000011;
        let r = _ktestz_mask32_u8(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_ktest_mask64_u8() {
        let a: __mmask64 = 0b0110100100111100_0110100100111100;
        let b: __mmask64 = 0b1001011011000011_1001011011000011;
        let mut and_not: u8 = 0;
        let r = _ktest_mask64_u8(a, b, &mut and_not);
        assert_eq!(r, 1);
        assert_eq!(and_not, 0);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_ktestc_mask64_u8() {
        let a: __mmask64 = 0b0110100100111100_0110100100111100;
        let b: __mmask64 = 0b1001011011000011_1001011011000011;
        let r = _ktestc_mask64_u8(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_ktestz_mask64_u8() {
        let a: __mmask64 = 0b0110100100111100_0110100100111100;
        let b: __mmask64 = 0b1001011011000011_1001011011000011;
        let r = _ktestz_mask64_u8(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_kunpackw() {
        let a: u32 = 0x00110011;
        let b: u32 = 0x00001011;
        let r = _mm512_kunpackw(a, b);
        let e: u32 = 0x00111011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_kunpackd() {
        let a: u64 = 0x11001100_00110011;
        let b: u64 = 0x00101110_00001011;
        let r = _mm512_kunpackd(a, b);
        let e: u64 = 0x00110011_00001011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cvtepi16_epi8() {
        let a = _mm512_set1_epi16(2);
        let r = _mm512_cvtepi16_epi8(a);
        let e = _mm256_set1_epi8(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cvtepi16_epi8() {
        let src = _mm256_set1_epi8(1);
        let a = _mm512_set1_epi16(2);
        let r = _mm512_mask_cvtepi16_epi8(src, 0, a);
        assert_eq_m256i(r, src);
        let r = _mm512_mask_cvtepi16_epi8(src, 0b11111111_11111111_11111111_11111111, a);
        let e = _mm256_set1_epi8(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_cvtepi16_epi8() {
        let a = _mm512_set1_epi16(2);
        let r = _mm512_maskz_cvtepi16_epi8(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm512_maskz_cvtepi16_epi8(0b11111111_11111111_11111111_11111111, a);
        let e = _mm256_set1_epi8(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cvtepi16_epi8() {
        let a = _mm256_set1_epi16(2);
        let r = _mm256_cvtepi16_epi8(a);
        let e = _mm_set1_epi8(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cvtepi16_epi8() {
        let src = _mm_set1_epi8(1);
        let a = _mm256_set1_epi16(2);
        let r = _mm256_mask_cvtepi16_epi8(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm256_mask_cvtepi16_epi8(src, 0b11111111_11111111, a);
        let e = _mm_set1_epi8(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_cvtepi16_epi8() {
        let a = _mm256_set1_epi16(2);
        let r = _mm256_maskz_cvtepi16_epi8(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm256_maskz_cvtepi16_epi8(0b11111111_11111111, a);
        let e = _mm_set1_epi8(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cvtepi16_epi8() {
        let a = _mm_set1_epi16(2);
        let r = _mm_cvtepi16_epi8(a);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cvtepi16_epi8() {
        let src = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        let a = _mm_set1_epi16(2);
        let r = _mm_mask_cvtepi16_epi8(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm_mask_cvtepi16_epi8(src, 0b11111111, a);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_cvtepi16_epi8() {
        let a = _mm_set1_epi16(2);
        let r = _mm_maskz_cvtepi16_epi8(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_cvtepi16_epi8(0b11111111, a);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cvtsepi16_epi8() {
        let a = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_cvtsepi16_epi8(a);
        let e = _mm256_set1_epi8(i8::MAX);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cvtsepi16_epi8() {
        let src = _mm256_set1_epi8(1);
        let a = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_mask_cvtsepi16_epi8(src, 0, a);
        assert_eq_m256i(r, src);
        let r = _mm512_mask_cvtsepi16_epi8(src, 0b11111111_11111111_11111111_11111111, a);
        let e = _mm256_set1_epi8(i8::MAX);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cvtsepi16_epi8() {
        let a = _mm256_set1_epi16(i16::MAX);
        let r = _mm256_cvtsepi16_epi8(a);
        let e = _mm_set1_epi8(i8::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cvtsepi16_epi8() {
        let src = _mm_set1_epi8(1);
        let a = _mm256_set1_epi16(i16::MAX);
        let r = _mm256_mask_cvtsepi16_epi8(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm256_mask_cvtsepi16_epi8(src, 0b11111111_11111111, a);
        let e = _mm_set1_epi8(i8::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_cvtsepi16_epi8() {
        let a = _mm256_set1_epi16(i16::MAX);
        let r = _mm256_maskz_cvtsepi16_epi8(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm256_maskz_cvtsepi16_epi8(0b11111111_11111111, a);
        let e = _mm_set1_epi8(i8::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cvtsepi16_epi8() {
        let a = _mm_set1_epi16(i16::MAX);
        let r = _mm_cvtsepi16_epi8(a);
        #[rustfmt::skip]
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cvtsepi16_epi8() {
        let src = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        let a = _mm_set1_epi16(i16::MAX);
        let r = _mm_mask_cvtsepi16_epi8(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm_mask_cvtsepi16_epi8(src, 0b11111111, a);
        #[rustfmt::skip]
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_cvtsepi16_epi8() {
        let a = _mm_set1_epi16(i16::MAX);
        let r = _mm_maskz_cvtsepi16_epi8(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_cvtsepi16_epi8(0b11111111, a);
        #[rustfmt::skip]
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_cvtsepi16_epi8() {
        let a = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_maskz_cvtsepi16_epi8(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm512_maskz_cvtsepi16_epi8(0b11111111_11111111_11111111_11111111, a);
        let e = _mm256_set1_epi8(i8::MAX);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cvtusepi16_epi8() {
        let a = _mm512_set1_epi16(i16::MIN);
        let r = _mm512_cvtusepi16_epi8(a);
        let e = _mm256_set1_epi8(-1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cvtusepi16_epi8() {
        let src = _mm256_set1_epi8(1);
        let a = _mm512_set1_epi16(i16::MIN);
        let r = _mm512_mask_cvtusepi16_epi8(src, 0, a);
        assert_eq_m256i(r, src);
        let r = _mm512_mask_cvtusepi16_epi8(src, 0b11111111_11111111_11111111_11111111, a);
        let e = _mm256_set1_epi8(-1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_cvtusepi16_epi8() {
        let a = _mm512_set1_epi16(i16::MIN);
        let r = _mm512_maskz_cvtusepi16_epi8(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm512_maskz_cvtusepi16_epi8(0b11111111_11111111_11111111_11111111, a);
        let e = _mm256_set1_epi8(-1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cvtusepi16_epi8() {
        let a = _mm256_set1_epi16(i16::MIN);
        let r = _mm256_cvtusepi16_epi8(a);
        let e = _mm_set1_epi8(-1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cvtusepi16_epi8() {
        let src = _mm_set1_epi8(1);
        let a = _mm256_set1_epi16(i16::MIN);
        let r = _mm256_mask_cvtusepi16_epi8(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm256_mask_cvtusepi16_epi8(src, 0b11111111_11111111, a);
        let e = _mm_set1_epi8(-1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_cvtusepi16_epi8() {
        let a = _mm256_set1_epi16(i16::MIN);
        let r = _mm256_maskz_cvtusepi16_epi8(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm256_maskz_cvtusepi16_epi8(0b11111111_11111111, a);
        let e = _mm_set1_epi8(-1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cvtusepi16_epi8() {
        let a = _mm_set1_epi16(i16::MIN);
        let r = _mm_cvtusepi16_epi8(a);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cvtusepi16_epi8() {
        let src = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        let a = _mm_set1_epi16(i16::MIN);
        let r = _mm_mask_cvtusepi16_epi8(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm_mask_cvtusepi16_epi8(src, 0b11111111, a);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_cvtusepi16_epi8() {
        let a = _mm_set1_epi16(i16::MIN);
        let r = _mm_maskz_cvtusepi16_epi8(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_cvtusepi16_epi8(0b11111111, a);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cvtepi8_epi16() {
        let a = _mm256_set1_epi8(2);
        let r = _mm512_cvtepi8_epi16(a);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cvtepi8_epi16() {
        let src = _mm512_set1_epi16(1);
        let a = _mm256_set1_epi8(2);
        let r = _mm512_mask_cvtepi8_epi16(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_cvtepi8_epi16(src, 0b11111111_11111111_11111111_11111111, a);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_cvtepi8_epi16() {
        let a = _mm256_set1_epi8(2);
        let r = _mm512_maskz_cvtepi8_epi16(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_cvtepi8_epi16(0b11111111_11111111_11111111_11111111, a);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cvtepi8_epi16() {
        let src = _mm256_set1_epi16(1);
        let a = _mm_set1_epi8(2);
        let r = _mm256_mask_cvtepi8_epi16(src, 0, a);
        assert_eq_m256i(r, src);
        let r = _mm256_mask_cvtepi8_epi16(src, 0b11111111_11111111, a);
        let e = _mm256_set1_epi16(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_cvtepi8_epi16() {
        let a = _mm_set1_epi8(2);
        let r = _mm256_maskz_cvtepi8_epi16(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_cvtepi8_epi16(0b11111111_11111111, a);
        let e = _mm256_set1_epi16(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cvtepi8_epi16() {
        let src = _mm_set1_epi16(1);
        let a = _mm_set1_epi8(2);
        let r = _mm_mask_cvtepi8_epi16(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm_mask_cvtepi8_epi16(src, 0b11111111, a);
        let e = _mm_set1_epi16(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_cvtepi8_epi16() {
        let a = _mm_set1_epi8(2);
        let r = _mm_maskz_cvtepi8_epi16(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_cvtepi8_epi16(0b11111111, a);
        let e = _mm_set1_epi16(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cvtepu8_epi16() {
        let a = _mm256_set1_epi8(2);
        let r = _mm512_cvtepu8_epi16(a);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cvtepu8_epi16() {
        let src = _mm512_set1_epi16(1);
        let a = _mm256_set1_epi8(2);
        let r = _mm512_mask_cvtepu8_epi16(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_cvtepu8_epi16(src, 0b11111111_11111111_11111111_11111111, a);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_cvtepu8_epi16() {
        let a = _mm256_set1_epi8(2);
        let r = _mm512_maskz_cvtepu8_epi16(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_cvtepu8_epi16(0b11111111_11111111_11111111_11111111, a);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cvtepu8_epi16() {
        let src = _mm256_set1_epi16(1);
        let a = _mm_set1_epi8(2);
        let r = _mm256_mask_cvtepu8_epi16(src, 0, a);
        assert_eq_m256i(r, src);
        let r = _mm256_mask_cvtepu8_epi16(src, 0b11111111_11111111, a);
        let e = _mm256_set1_epi16(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_cvtepu8_epi16() {
        let a = _mm_set1_epi8(2);
        let r = _mm256_maskz_cvtepu8_epi16(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_cvtepu8_epi16(0b11111111_11111111, a);
        let e = _mm256_set1_epi16(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cvtepu8_epi16() {
        let src = _mm_set1_epi16(1);
        let a = _mm_set1_epi8(2);
        let r = _mm_mask_cvtepu8_epi16(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm_mask_cvtepu8_epi16(src, 0b11111111, a);
        let e = _mm_set1_epi16(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_cvtepu8_epi16() {
        let a = _mm_set1_epi8(2);
        let r = _mm_maskz_cvtepu8_epi16(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_cvtepu8_epi16(0b11111111, a);
        let e = _mm_set1_epi16(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_bslli_epi128() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        );
        let r = _mm512_bslli_epi128::<9>(a);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_bsrli_epi128() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
            49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        );
        let r = _mm512_bsrli_epi128::<3>(a);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            0, 0, 0, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            0, 0, 0, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            0, 0, 0, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_alignr_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        );
        let b = _mm512_set1_epi8(1);
        let r = _mm512_alignr_epi8::<14>(a, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_alignr_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        );
        let b = _mm512_set1_epi8(1);
        let r = _mm512_mask_alignr_epi8::<14>(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_alignr_epi8::<14>(
            a,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_alignr_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        );
        let b = _mm512_set1_epi8(1);
        let r = _mm512_maskz_alignr_epi8::<14>(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_alignr_epi8::<14>(
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_alignr_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        );
        let b = _mm256_set1_epi8(1);
        let r = _mm256_mask_alignr_epi8::<14>(a, 0, a, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_alignr_epi8::<14>(a, 0b11111111_11111111_11111111_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_alignr_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        );
        let b = _mm256_set1_epi8(1);
        let r = _mm256_maskz_alignr_epi8::<14>(0, a, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_alignr_epi8::<14>(0b11111111_11111111_11111111_11111111, a, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_alignr_epi8() {
        let a = _mm_set_epi8(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0);
        let b = _mm_set1_epi8(1);
        let r = _mm_mask_alignr_epi8::<14>(a, 0, a, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_alignr_epi8::<14>(a, 0b11111111_11111111, a, b);
        let e = _mm_set_epi8(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_alignr_epi8() {
        let a = _mm_set_epi8(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0);
        let b = _mm_set1_epi8(1);
        let r = _mm_maskz_alignr_epi8::<14>(0, a, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_alignr_epi8::<14>(0b11111111_11111111, a, b);
        let e = _mm_set_epi8(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cvtsepi16_storeu_epi8() {
        let a = _mm512_set1_epi16(i16::MAX);
        let mut r = _mm256_undefined_si256();
        _mm512_mask_cvtsepi16_storeu_epi8(
            &mut r as *mut _ as *mut i8,
            0b11111111_11111111_11111111_11111111,
            a,
        );
        let e = _mm256_set1_epi8(i8::MAX);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cvtsepi16_storeu_epi8() {
        let a = _mm256_set1_epi16(i16::MAX);
        let mut r = _mm_undefined_si128();
        _mm256_mask_cvtsepi16_storeu_epi8(&mut r as *mut _ as *mut i8, 0b11111111_11111111, a);
        let e = _mm_set1_epi8(i8::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cvtsepi16_storeu_epi8() {
        let a = _mm_set1_epi16(i16::MAX);
        let mut r = _mm_set1_epi8(0);
        _mm_mask_cvtsepi16_storeu_epi8(&mut r as *mut _ as *mut i8, 0b11111111, a);
        #[rustfmt::skip]
        let e = _mm_set_epi8(
            0, 0, 0, 0, 0, 0, 0, 0,
            i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cvtepi16_storeu_epi8() {
        let a = _mm512_set1_epi16(8);
        let mut r = _mm256_undefined_si256();
        _mm512_mask_cvtepi16_storeu_epi8(
            &mut r as *mut _ as *mut i8,
            0b11111111_11111111_11111111_11111111,
            a,
        );
        let e = _mm256_set1_epi8(8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cvtepi16_storeu_epi8() {
        let a = _mm256_set1_epi16(8);
        let mut r = _mm_undefined_si128();
        _mm256_mask_cvtepi16_storeu_epi8(&mut r as *mut _ as *mut i8, 0b11111111_11111111, a);
        let e = _mm_set1_epi8(8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cvtepi16_storeu_epi8() {
        let a = _mm_set1_epi16(8);
        let mut r = _mm_set1_epi8(0);
        _mm_mask_cvtepi16_storeu_epi8(&mut r as *mut _ as *mut i8, 0b11111111, a);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cvtusepi16_storeu_epi8() {
        let a = _mm512_set1_epi16(i16::MAX);
        let mut r = _mm256_undefined_si256();
        _mm512_mask_cvtusepi16_storeu_epi8(
            &mut r as *mut _ as *mut i8,
            0b11111111_11111111_11111111_11111111,
            a,
        );
        let e = _mm256_set1_epi8(u8::MAX as i8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cvtusepi16_storeu_epi8() {
        let a = _mm256_set1_epi16(i16::MAX);
        let mut r = _mm_undefined_si128();
        _mm256_mask_cvtusepi16_storeu_epi8(&mut r as *mut _ as *mut i8, 0b11111111_11111111, a);
        let e = _mm_set1_epi8(u8::MAX as i8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cvtusepi16_storeu_epi8() {
        let a = _mm_set1_epi16(i16::MAX);
        let mut r = _mm_set1_epi8(0);
        _mm_mask_cvtusepi16_storeu_epi8(&mut r as *mut _ as *mut i8, 0b11111111, a);
        #[rustfmt::skip]
        let e = _mm_set_epi8(
            0, 0, 0, 0,
            0, 0, 0, 0,
            u8::MAX as i8, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8, 
            u8::MAX as i8, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8,
        );
        assert_eq_m128i(r, e);
    }
}
