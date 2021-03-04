use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem::{self, transmute},
    ptr,
};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_abs_epi16&expand=30)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub unsafe fn _mm512_abs_epi16(a: __m512i) -> __m512i {
    let a = a.as_i16x32();
    // all-0 is a properly initialized i16x32
    let zero: i16x32 = mem::zeroed();
    let sub = simd_sub(zero, a);
    let cmp: i16x32 = simd_gt(a, zero);
    transmute(simd_select(cmp, a, sub))
}

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_abs_epi16&expand=31)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub unsafe fn _mm512_mask_abs_epi16(src: __m512i, k: __mmask32, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi16(a).as_i16x32();
    transmute(simd_select_bitmask(k, abs, src.as_i16x32()))
}

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_abs_epi16&expand=32)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub unsafe fn _mm512_maskz_abs_epi16(k: __mmask32, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi16(a).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, abs, zero))
}

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_abs_epi16&expand=28)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub unsafe fn _mm256_mask_abs_epi16(src: __m256i, k: __mmask16, a: __m256i) -> __m256i {
    let abs = _mm256_abs_epi16(a).as_i16x16();
    transmute(simd_select_bitmask(k, abs, src.as_i16x16()))
}

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_abs_epi16&expand=29)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub unsafe fn _mm256_maskz_abs_epi16(k: __mmask16, a: __m256i) -> __m256i {
    let abs = _mm256_abs_epi16(a).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, abs, zero))
}

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_abs_epi16&expand=25)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub unsafe fn _mm_mask_abs_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    let abs = _mm_abs_epi16(a).as_i16x8();
    transmute(simd_select_bitmask(k, abs, src.as_i16x8()))
}

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_abs_epi16&expand=26)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub unsafe fn _mm_maskz_abs_epi16(k: __mmask8, a: __m128i) -> __m128i {
    let abs = _mm_abs_epi16(a).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, abs, zero))
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_abs_epi8&expand=57)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub unsafe fn _mm512_abs_epi8(a: __m512i) -> __m512i {
    let a = a.as_i8x64();
    // all-0 is a properly initialized i8x64
    let zero: i8x64 = mem::zeroed();
    let sub = simd_sub(zero, a);
    let cmp: i8x64 = simd_gt(a, zero);
    transmute(simd_select(cmp, a, sub))
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_abs_epi8&expand=58)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub unsafe fn _mm512_mask_abs_epi8(src: __m512i, k: __mmask64, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi8(a).as_i8x64();
    transmute(simd_select_bitmask(k, abs, src.as_i8x64()))
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_abs_epi8&expand=59)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub unsafe fn _mm512_maskz_abs_epi8(k: __mmask64, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi8(a).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, abs, zero))
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_abs_epi8&expand=55)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub unsafe fn _mm256_mask_abs_epi8(src: __m256i, k: __mmask32, a: __m256i) -> __m256i {
    let abs = _mm256_abs_epi8(a).as_i8x32();
    transmute(simd_select_bitmask(k, abs, src.as_i8x32()))
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_abs_epi8&expand=56)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub unsafe fn _mm256_maskz_abs_epi8(k: __mmask32, a: __m256i) -> __m256i {
    let abs = _mm256_abs_epi8(a).as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, abs, zero))
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set)
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_abs_epi8&expand=52)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub unsafe fn _mm_mask_abs_epi8(src: __m128i, k: __mmask16, a: __m128i) -> __m128i {
    let abs = _mm_abs_epi8(a).as_i8x16();
    transmute(simd_select_bitmask(k, abs, src.as_i8x16()))
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_abs_epi8&expand=53)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub unsafe fn _mm_maskz_abs_epi8(k: __mmask16, a: __m128i) -> __m128i {
    let abs = _mm_abs_epi8(a).as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, abs, zero))
}

/// Add packed 16-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_add_epi16&expand=91)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub unsafe fn _mm512_add_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_add(a.as_i16x32(), b.as_i16x32()))
}

/// Add packed 16-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_add_epi16&expand=92)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub unsafe fn _mm512_mask_add_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, add, src.as_i16x32()))
}

/// Add packed 16-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_add_epi16&expand=93)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub unsafe fn _mm512_maskz_add_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Add packed 16-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_add_epi&expand=89)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub unsafe fn _mm256_mask_add_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let add = _mm256_add_epi16(a, b).as_i16x16();
    transmute(simd_select_bitmask(k, add, src.as_i16x16()))
}

/// Add packed 16-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_add_epi16&expand=90)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub unsafe fn _mm256_maskz_add_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let add = _mm256_add_epi16(a, b).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Add packed 16-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_add_epi16&expand=86)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub unsafe fn _mm_mask_add_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let add = _mm_add_epi16(a, b).as_i16x8();
    transmute(simd_select_bitmask(k, add, src.as_i16x8()))
}

/// Add packed 16-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_add_epi16&expand=87)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub unsafe fn _mm_maskz_add_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let add = _mm_add_epi16(a, b).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Add packed 8-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_add_epi8&expand=118)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub unsafe fn _mm512_add_epi8(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_add(a.as_i8x64(), b.as_i8x64()))
}

/// Add packed 8-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_add_epi8&expand=119)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub unsafe fn _mm512_mask_add_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi8(a, b).as_i8x64();
    transmute(simd_select_bitmask(k, add, src.as_i8x64()))
}

/// Add packed 8-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_add_epi8&expand=120)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub unsafe fn _mm512_maskz_add_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi8(a, b).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Add packed 8-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_add_epi8&expand=116)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub unsafe fn _mm256_mask_add_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let add = _mm256_add_epi8(a, b).as_i8x32();
    transmute(simd_select_bitmask(k, add, src.as_i8x32()))
}

/// Add packed 8-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_add_epi8&expand=117)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub unsafe fn _mm256_maskz_add_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let add = _mm256_add_epi8(a, b).as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Add packed 8-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_add_epi8&expand=113)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub unsafe fn _mm_mask_add_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let add = _mm_add_epi8(a, b).as_i8x16();
    transmute(simd_select_bitmask(k, add, src.as_i8x16()))
}

/// Add packed 8-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_add_epi8&expand=114)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub unsafe fn _mm_maskz_add_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let add = _mm_add_epi8(a, b).as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_adds_epu16&expand=197)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub unsafe fn _mm512_adds_epu16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddusw(
        a.as_u16x32(),
        b.as_u16x32(),
        _mm512_setzero_si512().as_u16x32(),
        0b11111111_11111111_11111111_11111111,
    ))
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_adds_epu16&expand=198)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub unsafe fn _mm512_mask_adds_epu16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    transmute(vpaddusw(a.as_u16x32(), b.as_u16x32(), src.as_u16x32(), k))
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_adds_epu16&expand=199)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub unsafe fn _mm512_maskz_adds_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddusw(
        a.as_u16x32(),
        b.as_u16x32(),
        _mm512_setzero_si512().as_u16x32(),
        k,
    ))
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_adds_epu16&expand=195)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub unsafe fn _mm256_mask_adds_epu16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    transmute(vpaddusw256(
        a.as_u16x16(),
        b.as_u16x16(),
        src.as_u16x16(),
        k,
    ))
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_adds_epu16&expand=196)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub unsafe fn _mm256_maskz_adds_epu16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    transmute(vpaddusw256(
        a.as_u16x16(),
        b.as_u16x16(),
        _mm256_setzero_si256().as_u16x16(),
        k,
    ))
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_adds_epu16&expand=192)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub unsafe fn _mm_mask_adds_epu16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpaddusw128(a.as_u16x8(), b.as_u16x8(), src.as_u16x8(), k))
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_adds_epu16&expand=193)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub unsafe fn _mm_maskz_adds_epu16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpaddusw128(
        a.as_u16x8(),
        b.as_u16x8(),
        _mm_setzero_si128().as_u16x8(),
        k,
    ))
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_adds_epu8&expand=206)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub unsafe fn _mm512_adds_epu8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddusb(
        a.as_u8x64(),
        b.as_u8x64(),
        _mm512_setzero_si512().as_u8x64(),
        0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
    ))
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_adds_epu8&expand=207)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub unsafe fn _mm512_mask_adds_epu8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddusb(a.as_u8x64(), b.as_u8x64(), src.as_u8x64(), k))
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_adds_epu8&expand=208)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub unsafe fn _mm512_maskz_adds_epu8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddusb(
        a.as_u8x64(),
        b.as_u8x64(),
        _mm512_setzero_si512().as_u8x64(),
        k,
    ))
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_adds_epu8&expand=204)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub unsafe fn _mm256_mask_adds_epu8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    transmute(vpaddusb256(a.as_u8x32(), b.as_u8x32(), src.as_u8x32(), k))
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_adds_epu8&expand=205)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub unsafe fn _mm256_maskz_adds_epu8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    transmute(vpaddusb256(
        a.as_u8x32(),
        b.as_u8x32(),
        _mm256_setzero_si256().as_u8x32(),
        k,
    ))
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_adds_epu8&expand=201)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub unsafe fn _mm_mask_adds_epu8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpaddusb128(a.as_u8x16(), b.as_u8x16(), src.as_u8x16(), k))
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_adds_epu8&expand=202)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub unsafe fn _mm_maskz_adds_epu8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpaddusb128(
        a.as_u8x16(),
        b.as_u8x16(),
        _mm_setzero_si128().as_u8x16(),
        k,
    ))
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_adds_epi16&expand=179)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub unsafe fn _mm512_adds_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddsw(
        a.as_i16x32(),
        b.as_i16x32(),
        _mm512_setzero_si512().as_i16x32(),
        0b11111111_11111111_11111111_11111111,
    ))
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_adds_epi16&expand=180)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub unsafe fn _mm512_mask_adds_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    transmute(vpaddsw(a.as_i16x32(), b.as_i16x32(), src.as_i16x32(), k))
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_adds_epi16&expand=181)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub unsafe fn _mm512_maskz_adds_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddsw(
        a.as_i16x32(),
        b.as_i16x32(),
        _mm512_setzero_si512().as_i16x32(),
        k,
    ))
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_adds_epi16&expand=177)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub unsafe fn _mm256_mask_adds_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    transmute(vpaddsw256(a.as_i16x16(), b.as_i16x16(), src.as_i16x16(), k))
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_adds_epi16&expand=178)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub unsafe fn _mm256_maskz_adds_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    transmute(vpaddsw256(
        a.as_i16x16(),
        b.as_i16x16(),
        _mm256_setzero_si256().as_i16x16(),
        k,
    ))
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_adds_epi16&expand=174)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub unsafe fn _mm_mask_adds_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpaddsw128(a.as_i16x8(), b.as_i16x8(), src.as_i16x8(), k))
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_adds_epi16&expand=175)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub unsafe fn _mm_maskz_adds_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpaddsw128(
        a.as_i16x8(),
        b.as_i16x8(),
        _mm_setzero_si128().as_i16x8(),
        k,
    ))
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_adds_epi8&expand=188)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub unsafe fn _mm512_adds_epi8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddsb(
        a.as_i8x64(),
        b.as_i8x64(),
        _mm512_setzero_si512().as_i8x64(),
        0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
    ))
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_adds_epi8&expand=189)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub unsafe fn _mm512_mask_adds_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddsb(a.as_i8x64(), b.as_i8x64(), src.as_i8x64(), k))
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_adds_epi8&expand=190)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub unsafe fn _mm512_maskz_adds_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddsb(
        a.as_i8x64(),
        b.as_i8x64(),
        _mm512_setzero_si512().as_i8x64(),
        k,
    ))
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_adds_epi8&expand=186)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub unsafe fn _mm256_mask_adds_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    transmute(vpaddsb256(a.as_i8x32(), b.as_i8x32(), src.as_i8x32(), k))
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_adds_epi8&expand=187)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub unsafe fn _mm256_maskz_adds_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    transmute(vpaddsb256(
        a.as_i8x32(),
        b.as_i8x32(),
        _mm256_setzero_si256().as_i8x32(),
        k,
    ))
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_adds_epi8&expand=183)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub unsafe fn _mm_mask_adds_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpaddsb128(a.as_i8x16(), b.as_i8x16(), src.as_i8x16(), k))
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_adds_epi8&expand=184)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub unsafe fn _mm_maskz_adds_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpaddsb128(
        a.as_i8x16(),
        b.as_i8x16(),
        _mm_setzero_si128().as_i8x16(),
        k,
    ))
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_sub_epi16&expand=5685)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub unsafe fn _mm512_sub_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_sub(a.as_i16x32(), b.as_i16x32()))
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_sub_epi16&expand=5683)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub unsafe fn _mm512_mask_sub_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let sub = _mm512_sub_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, sub, src.as_i16x32()))
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_sub_epi16&expand=5684)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub unsafe fn _mm512_maskz_sub_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let sub = _mm512_sub_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, sub, zero))
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_sub_epi16&expand=5680)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub unsafe fn _mm256_mask_sub_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let sub = _mm256_sub_epi16(a, b).as_i16x16();
    transmute(simd_select_bitmask(k, sub, src.as_i16x16()))
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_sub_epi16&expand=5681)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub unsafe fn _mm256_maskz_sub_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let sub = _mm256_sub_epi16(a, b).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, sub, zero))
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_sub_epi16&expand=5677)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub unsafe fn _mm_mask_sub_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let sub = _mm_sub_epi16(a, b).as_i16x8();
    transmute(simd_select_bitmask(k, sub, src.as_i16x8()))
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_sub_epi16&expand=5678)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub unsafe fn _mm_maskz_sub_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let sub = _mm_sub_epi16(a, b).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, sub, zero))
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_sub_epi8&expand=5712)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub unsafe fn _mm512_sub_epi8(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_sub(a.as_i8x64(), b.as_i8x64()))
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_sub_epi8&expand=5710)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub unsafe fn _mm512_mask_sub_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let sub = _mm512_sub_epi8(a, b).as_i8x64();
    transmute(simd_select_bitmask(k, sub, src.as_i8x64()))
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_sub_epi8&expand=5711)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub unsafe fn _mm512_maskz_sub_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let sub = _mm512_sub_epi8(a, b).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, sub, zero))
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_sub_epi8&expand=5707)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub unsafe fn _mm256_mask_sub_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let sub = _mm256_sub_epi8(a, b).as_i8x32();
    transmute(simd_select_bitmask(k, sub, src.as_i8x32()))
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_sub_epi8&expand=5708)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub unsafe fn _mm256_maskz_sub_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let sub = _mm256_sub_epi8(a, b).as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, sub, zero))
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_sub_epi8&expand=5704)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub unsafe fn _mm_mask_sub_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let sub = _mm_sub_epi8(a, b).as_i8x16();
    transmute(simd_select_bitmask(k, sub, src.as_i8x16()))
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_sub_epi8&expand=5705)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub unsafe fn _mm_maskz_sub_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let sub = _mm_sub_epi8(a, b).as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, sub, zero))
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_subs_epu16&expand=5793)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub unsafe fn _mm512_subs_epu16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubusw(
        a.as_u16x32(),
        b.as_u16x32(),
        _mm512_setzero_si512().as_u16x32(),
        0b11111111_11111111_11111111_11111111,
    ))
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_subs_epu16&expand=5791)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub unsafe fn _mm512_mask_subs_epu16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    transmute(vpsubusw(a.as_u16x32(), b.as_u16x32(), src.as_u16x32(), k))
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_subs_epu16&expand=5792)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub unsafe fn _mm512_maskz_subs_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubusw(
        a.as_u16x32(),
        b.as_u16x32(),
        _mm512_setzero_si512().as_u16x32(),
        k,
    ))
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_subs_epu16&expand=5788)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub unsafe fn _mm256_mask_subs_epu16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    transmute(vpsubusw256(
        a.as_u16x16(),
        b.as_u16x16(),
        src.as_u16x16(),
        k,
    ))
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_subs_epu16&expand=5789)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub unsafe fn _mm256_maskz_subs_epu16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    transmute(vpsubusw256(
        a.as_u16x16(),
        b.as_u16x16(),
        _mm256_setzero_si256().as_u16x16(),
        k,
    ))
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_subs_epu16&expand=5785)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub unsafe fn _mm_mask_subs_epu16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpsubusw128(a.as_u16x8(), b.as_u16x8(), src.as_u16x8(), k))
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_subs_epu16&expand=5786)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub unsafe fn _mm_maskz_subs_epu16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpsubusw128(
        a.as_u16x8(),
        b.as_u16x8(),
        _mm_setzero_si128().as_u16x8(),
        k,
    ))
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_subs_epu8&expand=5802)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub unsafe fn _mm512_subs_epu8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubusb(
        a.as_u8x64(),
        b.as_u8x64(),
        _mm512_setzero_si512().as_u8x64(),
        0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
    ))
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_subs_epu8&expand=5800)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub unsafe fn _mm512_mask_subs_epu8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubusb(a.as_u8x64(), b.as_u8x64(), src.as_u8x64(), k))
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_subs_epu8&expand=5801)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub unsafe fn _mm512_maskz_subs_epu8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubusb(
        a.as_u8x64(),
        b.as_u8x64(),
        _mm512_setzero_si512().as_u8x64(),
        k,
    ))
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_subs_epu8&expand=5797)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub unsafe fn _mm256_mask_subs_epu8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    transmute(vpsubusb256(a.as_u8x32(), b.as_u8x32(), src.as_u8x32(), k))
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_subs_epu8&expand=5798)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub unsafe fn _mm256_maskz_subs_epu8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    transmute(vpsubusb256(
        a.as_u8x32(),
        b.as_u8x32(),
        _mm256_setzero_si256().as_u8x32(),
        k,
    ))
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_subs_epu8&expand=5794)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub unsafe fn _mm_mask_subs_epu8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpsubusb128(a.as_u8x16(), b.as_u8x16(), src.as_u8x16(), k))
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_subs_epu8&expand=5795)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub unsafe fn _mm_maskz_subs_epu8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpsubusb128(
        a.as_u8x16(),
        b.as_u8x16(),
        _mm_setzero_si128().as_u8x16(),
        k,
    ))
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_subs_epi16&expand=5775)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub unsafe fn _mm512_subs_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubsw(
        a.as_i16x32(),
        b.as_i16x32(),
        _mm512_setzero_si512().as_i16x32(),
        0b11111111_11111111_11111111_11111111,
    ))
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_subs_epi16&expand=5773)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub unsafe fn _mm512_mask_subs_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    transmute(vpsubsw(a.as_i16x32(), b.as_i16x32(), src.as_i16x32(), k))
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_subs_epi16&expand=5774)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub unsafe fn _mm512_maskz_subs_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubsw(
        a.as_i16x32(),
        b.as_i16x32(),
        _mm512_setzero_si512().as_i16x32(),
        k,
    ))
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_subs_epi16&expand=5770)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub unsafe fn _mm256_mask_subs_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    transmute(vpsubsw256(a.as_i16x16(), b.as_i16x16(), src.as_i16x16(), k))
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_subs_epi16&expand=5771)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub unsafe fn _mm256_maskz_subs_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    transmute(vpsubsw256(
        a.as_i16x16(),
        b.as_i16x16(),
        _mm256_setzero_si256().as_i16x16(),
        k,
    ))
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_subs_epi16&expand=5767)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub unsafe fn _mm_mask_subs_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpsubsw128(a.as_i16x8(), b.as_i16x8(), src.as_i16x8(), k))
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_subs_epi16&expand=5768)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub unsafe fn _mm_maskz_subs_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpsubsw128(
        a.as_i16x8(),
        b.as_i16x8(),
        _mm_setzero_si128().as_i16x8(),
        k,
    ))
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_subs_epi8&expand=5784)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub unsafe fn _mm512_subs_epi8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubsb(
        a.as_i8x64(),
        b.as_i8x64(),
        _mm512_setzero_si512().as_i8x64(),
        0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
    ))
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_subs_epi8&expand=5782)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub unsafe fn _mm512_mask_subs_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubsb(a.as_i8x64(), b.as_i8x64(), src.as_i8x64(), k))
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_subs_epi8&expand=5783)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub unsafe fn _mm512_maskz_subs_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubsb(
        a.as_i8x64(),
        b.as_i8x64(),
        _mm512_setzero_si512().as_i8x64(),
        k,
    ))
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_subs_epi8&expand=5779)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub unsafe fn _mm256_mask_subs_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    transmute(vpsubsb256(a.as_i8x32(), b.as_i8x32(), src.as_i8x32(), k))
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_subs_epi8&expand=5780)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub unsafe fn _mm256_maskz_subs_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    transmute(vpsubsb256(
        a.as_i8x32(),
        b.as_i8x32(),
        _mm256_setzero_si256().as_i8x32(),
        k,
    ))
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_subs_epi8&expand=5776)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub unsafe fn _mm_mask_subs_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpsubsb128(a.as_i8x16(), b.as_i8x16(), src.as_i8x16(), k))
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_subs_epi8&expand=5777)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub unsafe fn _mm_maskz_subs_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    transmute(vpsubsb128(
        a.as_i8x16(),
        b.as_i8x16(),
        _mm_setzero_si128().as_i8x16(),
        k,
    ))
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mulhi_epu16&expand=3973)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub unsafe fn _mm512_mulhi_epu16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmulhuw(a.as_u16x32(), b.as_u16x32()))
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_mulhi_epu16&expand=3971)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub unsafe fn _mm512_mask_mulhi_epu16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let mul = _mm512_mulhi_epu16(a, b).as_u16x32();
    transmute(simd_select_bitmask(k, mul, src.as_u16x32()))
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_mulhi_epu16&expand=3972)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub unsafe fn _mm512_maskz_mulhi_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mulhi_epu16(a, b).as_u16x32();
    let zero = _mm512_setzero_si512().as_u16x32();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_mulhi_epu16&expand=3968)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub unsafe fn _mm256_mask_mulhi_epu16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    let mul = _mm256_mulhi_epu16(a, b).as_u16x16();
    transmute(simd_select_bitmask(k, mul, src.as_u16x16()))
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_mulhi_epu16&expand=3969)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub unsafe fn _mm256_maskz_mulhi_epu16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let mul = _mm256_mulhi_epu16(a, b).as_u16x16();
    let zero = _mm256_setzero_si256().as_u16x16();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_mulhi_epu16&expand=3965)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub unsafe fn _mm_mask_mulhi_epu16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let mul = _mm_mulhi_epu16(a, b).as_u16x8();
    transmute(simd_select_bitmask(k, mul, src.as_u16x8()))
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_mulhi_epu16&expand=3966)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub unsafe fn _mm_maskz_mulhi_epu16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let mul = _mm_mulhi_epu16(a, b).as_u16x8();
    let zero = _mm_setzero_si128().as_u16x8();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mulhi_epi16&expand=3962)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub unsafe fn _mm512_mulhi_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmulhw(a.as_i16x32(), b.as_i16x32()))
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_mulhi_epi16&expand=3960)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub unsafe fn _mm512_mask_mulhi_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let mul = _mm512_mulhi_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, mul, src.as_i16x32()))
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_mulhi_epi16&expand=3961)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub unsafe fn _mm512_maskz_mulhi_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mulhi_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_mulhi_epi16&expand=3957)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub unsafe fn _mm256_mask_mulhi_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    let mul = _mm256_mulhi_epi16(a, b).as_i16x16();
    transmute(simd_select_bitmask(k, mul, src.as_i16x16()))
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_mulhi_epi16&expand=3958)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub unsafe fn _mm256_maskz_mulhi_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let mul = _mm256_mulhi_epi16(a, b).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_mulhi_epi16&expand=3954)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub unsafe fn _mm_mask_mulhi_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let mul = _mm_mulhi_epi16(a, b).as_i16x8();
    transmute(simd_select_bitmask(k, mul, src.as_i16x8()))
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_mulhi_epi16&expand=3955)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub unsafe fn _mm_maskz_mulhi_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let mul = _mm_mulhi_epi16(a, b).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mulhrs_epi16&expand=3986)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub unsafe fn _mm512_mulhrs_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmulhrsw(a.as_i16x32(), b.as_i16x32()))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_mulhrs_epi16&expand=3984)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub unsafe fn _mm512_mask_mulhrs_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let mul = _mm512_mulhrs_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, mul, src.as_i16x32()))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_mulhrs_epi16&expand=3985)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub unsafe fn _mm512_maskz_mulhrs_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mulhrs_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_mulhrs_epi16&expand=3981)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub unsafe fn _mm256_mask_mulhrs_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    let mul = _mm256_mulhrs_epi16(a, b).as_i16x16();
    transmute(simd_select_bitmask(k, mul, src.as_i16x16()))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_mulhrs_epi16&expand=3982)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub unsafe fn _mm256_maskz_mulhrs_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let mul = _mm256_mulhrs_epi16(a, b).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_mulhrs_epi16&expand=3978)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub unsafe fn _mm_mask_mulhrs_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let mul = _mm_mulhrs_epi16(a, b).as_i16x8();
    transmute(simd_select_bitmask(k, mul, src.as_i16x8()))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_mulhrs_epi16&expand=3979)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub unsafe fn _mm_maskz_mulhrs_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let mul = _mm_mulhrs_epi16(a, b).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mullo_epi16&expand=3996)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub unsafe fn _mm512_mullo_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_mul(a.as_i16x32(), b.as_i16x32()))
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_mullo_epi16&expand=3994)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub unsafe fn _mm512_mask_mullo_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let mul = _mm512_mullo_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, mul, src.as_i16x32()))
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_mullo_epi16&expand=3995)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub unsafe fn _mm512_maskz_mullo_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mullo_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_mullo_epi16&expand=3991)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub unsafe fn _mm256_mask_mullo_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    let mul = _mm256_mullo_epi16(a, b).as_i16x16();
    transmute(simd_select_bitmask(k, mul, src.as_i16x16()))
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_mullo_epi16&expand=3992)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub unsafe fn _mm256_maskz_mullo_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let mul = _mm256_mullo_epi16(a, b).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_mullo_epi16&expand=3988)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub unsafe fn _mm_mask_mullo_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let mul = _mm_mullo_epi16(a, b).as_i16x8();
    transmute(simd_select_bitmask(k, mul, src.as_i16x8()))
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_mullo_epi16&expand=3989)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub unsafe fn _mm_maskz_mullo_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let mul = _mm_mullo_epi16(a, b).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_max_epu16&expand=3609)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub unsafe fn _mm512_max_epu16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaxuw(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_max_epu16&expand=3607)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub unsafe fn _mm512_mask_max_epu16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epu16(a, b).as_u16x32();
    transmute(simd_select_bitmask(k, max, src.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_max_epu16&expand=3608)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub unsafe fn _mm512_maskz_max_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epu16(a, b).as_u16x32();
    let zero = _mm512_setzero_si512().as_u16x32();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_max_epu16&expand=3604)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub unsafe fn _mm256_mask_max_epu16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let max = _mm256_max_epu16(a, b).as_u16x16();
    transmute(simd_select_bitmask(k, max, src.as_u16x16()))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_max_epu16&expand=3605)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub unsafe fn _mm256_maskz_max_epu16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let max = _mm256_max_epu16(a, b).as_u16x16();
    let zero = _mm256_setzero_si256().as_u16x16();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_max_epu16&expand=3601)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub unsafe fn _mm_mask_max_epu16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let max = _mm_max_epu16(a, b).as_u16x8();
    transmute(simd_select_bitmask(k, max, src.as_u16x8()))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_max_epu16&expand=3602)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub unsafe fn _mm_maskz_max_epu16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let max = _mm_max_epu16(a, b).as_u16x8();
    let zero = _mm_setzero_si128().as_u16x8();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_max_epu8&expand=3636)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub unsafe fn _mm512_max_epu8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaxub(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_max_epu8&expand=3634)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub unsafe fn _mm512_mask_max_epu8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epu8(a, b).as_u8x64();
    transmute(simd_select_bitmask(k, max, src.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_max_epu8&expand=3635)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub unsafe fn _mm512_maskz_max_epu8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epu8(a, b).as_u8x64();
    let zero = _mm512_setzero_si512().as_u8x64();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_max_epu8&expand=3631)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub unsafe fn _mm256_mask_max_epu8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let max = _mm256_max_epu8(a, b).as_u8x32();
    transmute(simd_select_bitmask(k, max, src.as_u8x32()))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_max_epu8&expand=3632)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub unsafe fn _mm256_maskz_max_epu8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let max = _mm256_max_epu8(a, b).as_u8x32();
    let zero = _mm256_setzero_si256().as_u8x32();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_max_epu8&expand=3628)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub unsafe fn _mm_mask_max_epu8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let max = _mm_max_epu8(a, b).as_u8x16();
    transmute(simd_select_bitmask(k, max, src.as_u8x16()))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_max_epu8&expand=3629)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub unsafe fn _mm_maskz_max_epu8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let max = _mm_max_epu8(a, b).as_u8x16();
    let zero = _mm_setzero_si128().as_u8x16();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_max_epi16&expand=3573)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub unsafe fn _mm512_max_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaxsw(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_max_epi16&expand=3571)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub unsafe fn _mm512_mask_max_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, max, src.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_max_epi16&expand=3572)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub unsafe fn _mm512_maskz_max_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_max_epi16&expand=3568)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub unsafe fn _mm256_mask_max_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let max = _mm256_max_epi16(a, b).as_i16x16();
    transmute(simd_select_bitmask(k, max, src.as_i16x16()))
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_max_epi16&expand=3569)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub unsafe fn _mm256_maskz_max_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let max = _mm256_max_epi16(a, b).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_max_epi16&expand=3565)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub unsafe fn _mm_mask_max_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let max = _mm_max_epi16(a, b).as_i16x8();
    transmute(simd_select_bitmask(k, max, src.as_i16x8()))
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_max_epi16&expand=3566)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub unsafe fn _mm_maskz_max_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let max = _mm_max_epi16(a, b).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_max_epi8&expand=3600)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub unsafe fn _mm512_max_epi8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaxsb(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_max_epi8&expand=3598)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub unsafe fn _mm512_mask_max_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi8(a, b).as_i8x64();
    transmute(simd_select_bitmask(k, max, src.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_max_epi8&expand=3599)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub unsafe fn _mm512_maskz_max_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi8(a, b).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_max_epi8&expand=3595)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub unsafe fn _mm256_mask_max_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let max = _mm256_max_epi8(a, b).as_i8x32();
    transmute(simd_select_bitmask(k, max, src.as_i8x32()))
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_max_epi8&expand=3596)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub unsafe fn _mm256_maskz_max_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let max = _mm256_max_epi8(a, b).as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_max_epi8&expand=3592)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub unsafe fn _mm_mask_max_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let max = _mm_max_epi8(a, b).as_i8x16();
    transmute(simd_select_bitmask(k, max, src.as_i8x16()))
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_max_epi8&expand=3593)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub unsafe fn _mm_maskz_max_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let max = _mm_max_epi8(a, b).as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_min_epu16&expand=3723)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpminuw))]
pub unsafe fn _mm512_min_epu16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpminuw(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_min_epu16&expand=3721)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpminuw))]
pub unsafe fn _mm512_mask_min_epu16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let min = _mm512_min_epu16(a, b).as_u16x32();
    transmute(simd_select_bitmask(k, min, src.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_min_epu16&expand=3722)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpminuw))]
pub unsafe fn _mm512_maskz_min_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let min = _mm512_min_epu16(a, b).as_u16x32();
    let zero = _mm512_setzero_si512().as_u16x32();
    transmute(simd_select_bitmask(k, min, zero))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_min_epu16&expand=3718)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminuw))]
pub unsafe fn _mm256_mask_min_epu16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let min = _mm256_min_epu16(a, b).as_u16x16();
    transmute(simd_select_bitmask(k, min, src.as_u16x16()))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_min_epu16&expand=3719)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminuw))]
pub unsafe fn _mm256_maskz_min_epu16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let min = _mm256_min_epu16(a, b).as_u16x16();
    let zero = _mm256_setzero_si256().as_u16x16();
    transmute(simd_select_bitmask(k, min, zero))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_min_epu16&expand=3715)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminuw))]
pub unsafe fn _mm_mask_min_epu16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let min = _mm_min_epu16(a, b).as_u16x8();
    transmute(simd_select_bitmask(k, min, src.as_u16x8()))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_min_epu16&expand=3716)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminuw))]
pub unsafe fn _mm_maskz_min_epu16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let min = _mm_min_epu16(a, b).as_u16x8();
    let zero = _mm_setzero_si128().as_u16x8();
    transmute(simd_select_bitmask(k, min, zero))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_min_epu8&expand=3750)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpminub))]
pub unsafe fn _mm512_min_epu8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpminub(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_min_epu8&expand=3748)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpminub))]
pub unsafe fn _mm512_mask_min_epu8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let min = _mm512_min_epu8(a, b).as_u8x64();
    transmute(simd_select_bitmask(k, min, src.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_min_epu8&expand=3749)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpminub))]
pub unsafe fn _mm512_maskz_min_epu8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let min = _mm512_min_epu8(a, b).as_u8x64();
    let zero = _mm512_setzero_si512().as_u8x64();
    transmute(simd_select_bitmask(k, min, zero))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_min_epu8&expand=3745)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminub))]
pub unsafe fn _mm256_mask_min_epu8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let min = _mm256_min_epu8(a, b).as_u8x32();
    transmute(simd_select_bitmask(k, min, src.as_u8x32()))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_min_epu8&expand=3746)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminub))]
pub unsafe fn _mm256_maskz_min_epu8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let min = _mm256_min_epu8(a, b).as_u8x32();
    let zero = _mm256_setzero_si256().as_u8x32();
    transmute(simd_select_bitmask(k, min, zero))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_min_epu8&expand=3742)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminub))]
pub unsafe fn _mm_mask_min_epu8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let min = _mm_min_epu8(a, b).as_u8x16();
    transmute(simd_select_bitmask(k, min, src.as_u8x16()))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_min_epu8&expand=3743)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminub))]
pub unsafe fn _mm_maskz_min_epu8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let min = _mm_min_epu8(a, b).as_u8x16();
    let zero = _mm_setzero_si128().as_u8x16();
    transmute(simd_select_bitmask(k, min, zero))
}

/// Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_min_epi16&expand=3687)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpminsw))]
pub unsafe fn _mm512_min_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpminsw(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_min_epi16&expand=3685)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpminsw))]
pub unsafe fn _mm512_mask_min_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let min = _mm512_min_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, min, src.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_min_epi16&expand=3686)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpminsw))]
pub unsafe fn _mm512_maskz_min_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let min = _mm512_min_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, min, zero))
}

/// Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_min_epi16&expand=3682)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminsw))]
pub unsafe fn _mm256_mask_min_epi16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let min = _mm256_min_epi16(a, b).as_i16x16();
    transmute(simd_select_bitmask(k, min, src.as_i16x16()))
}

/// Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_min_epi16&expand=3683)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminsw))]
pub unsafe fn _mm256_maskz_min_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let min = _mm256_min_epi16(a, b).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, min, zero))
}

/// Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_min_epi16&expand=3679)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminsw))]
pub unsafe fn _mm_mask_min_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let min = _mm_min_epi16(a, b).as_i16x8();
    transmute(simd_select_bitmask(k, min, src.as_i16x8()))
}

/// Compare packed signed 16-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_min_epi16&expand=3680)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminsw))]
pub unsafe fn _mm_maskz_min_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let min = _mm_min_epi16(a, b).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, min, zero))
}

/// Compare packed signed 8-bit integers in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_min_epi8&expand=3714)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpminsb))]
pub unsafe fn _mm512_min_epi8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpminsb(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_min_epi8&expand=3712)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpminsb))]
pub unsafe fn _mm512_mask_min_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let min = _mm512_min_epi8(a, b).as_i8x64();
    transmute(simd_select_bitmask(k, min, src.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_min_epi8&expand=3713)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpminsb))]
pub unsafe fn _mm512_maskz_min_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let min = _mm512_min_epi8(a, b).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, min, zero))
}

/// Compare packed signed 8-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_min_epi8&expand=3709)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminsb))]
pub unsafe fn _mm256_mask_min_epi8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let min = _mm256_min_epi8(a, b).as_i8x32();
    transmute(simd_select_bitmask(k, min, src.as_i8x32()))
}

/// Compare packed signed 8-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_min_epi8&expand=3710)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminsb))]
pub unsafe fn _mm256_maskz_min_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let min = _mm256_min_epi8(a, b).as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, min, zero))
}

/// Compare packed signed 8-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_min_epi8&expand=3706)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminsb))]
pub unsafe fn _mm_mask_min_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let min = _mm_min_epi8(a, b).as_i8x16();
    transmute(simd_select_bitmask(k, min, src.as_i8x16()))
}

/// Compare packed signed 8-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_min_epi8&expand=3707)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpminsb))]
pub unsafe fn _mm_maskz_min_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let min = _mm_min_epi8(a, b).as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, min, zero))
}

/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cmplt_epu16_mask&expand=1050)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<u16x32, _>(simd_lt(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmplt_epu16_mask&expand=1051)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmplt_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cmplt_epu16_mask&expand=1050)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmplt_epu16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    simd_bitmask::<u16x16, _>(simd_lt(a.as_u16x16(), b.as_u16x16()))
}

/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmplt_epu16_mask&expand=1049)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmplt_epu16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_cmplt_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmplt_epi16_mask&expand=1018)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmplt_epu16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    simd_bitmask::<u16x8, _>(simd_lt(a.as_u16x8(), b.as_u16x8()))
}

/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmplt_epi16_mask&expand=1019)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmplt_epu16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_cmplt_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=mm512_cmplt_epu8_mask&expand=1068)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<u8x64, _>(simd_lt(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmplt_epu8_mask&expand=1069)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmplt_epu8_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmplt_epu8_mask&expand=1066)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmplt_epu8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    simd_bitmask::<u8x32, _>(simd_lt(a.as_u8x32(), b.as_u8x32()))
}

/// Compare packed unsigned 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmplt_epu8_mask&expand=1067)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmplt_epu8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_cmplt_epu8_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmplt_epu8_mask&expand=1064)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmplt_epu8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    simd_bitmask::<u8x16, _>(simd_lt(a.as_u8x16(), b.as_u8x16()))
}

/// Compare packed unsigned 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmplt_epu8_mask&expand=1065)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmplt_epu8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_cmplt_epu8_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmplt_epi16_mask&expand=1022)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<i16x32, _>(simd_lt(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmplt_epi16_mask&expand=1023)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmplt_epi16_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmplt_epi16_mask&expand=1020)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmplt_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    simd_bitmask::<i16x16, _>(simd_lt(a.as_i16x16(), b.as_i16x16()))
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmplt_epi16_mask&expand=1021)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmplt_epi16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_cmplt_epi16_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmplt_epi16_mask&expand=1018)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmplt_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    simd_bitmask::<i16x8, _>(simd_lt(a.as_i16x8(), b.as_i16x8()))
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmplt_epi16_mask&expand=1019)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmplt_epi16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_cmplt_epi16_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmplt_epi8_mask&expand=1044)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<i8x64, _>(simd_lt(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmplt_epi8_mask&expand=1045)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmplt_epi8_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmplt_epi8_mask&expand=1042)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmplt_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    simd_bitmask::<i8x32, _>(simd_lt(a.as_i8x32(), b.as_i8x32()))
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmplt_epi8_mask&expand=1043)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmplt_epi8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_cmplt_epi8_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmplt_epi8_mask&expand=1040)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmplt_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    simd_bitmask::<i8x16, _>(simd_lt(a.as_i8x16(), b.as_i8x16()))
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmplt_epi8_mask&expand=1041)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmplt_epi8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_cmplt_epi8_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpgt_epu16_mask&expand=927)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<u16x32, _>(simd_gt(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpgt_epu16_mask&expand=928)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpgt_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpgt_epu16_mask&expand=925)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpgt_epu16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    simd_bitmask::<u16x16, _>(simd_gt(a.as_u16x16(), b.as_u16x16()))
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpgt_epu16_mask&expand=926)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpgt_epu16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_cmpgt_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpgt_epu16_mask&expand=923)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpgt_epu16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    simd_bitmask::<u16x8, _>(simd_gt(a.as_u16x8(), b.as_u16x8()))
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpgt_epu16_mask&expand=924)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpgt_epu16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_cmpgt_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpgt_epu8_mask&expand=945)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<u8x64, _>(simd_gt(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpgt_epu8_mask&expand=946)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpgt_epu8_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpgt_epu8_mask&expand=943)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpgt_epu8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    simd_bitmask::<u8x32, _>(simd_gt(a.as_u8x32(), b.as_u8x32()))
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpgt_epu8_mask&expand=944)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpgt_epu8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_cmpgt_epu8_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpgt_epu8_mask&expand=941)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpgt_epu8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    simd_bitmask::<u8x16, _>(simd_gt(a.as_u8x16(), b.as_u8x16()))
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpgt_epu8_mask&expand=942)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpgt_epu8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_cmpgt_epu8_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpgt_epi16_mask&expand=897)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<i16x32, _>(simd_gt(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpgt_epi16_mask&expand=898)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpgt_epi16_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpgt_epi16_mask&expand=895)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpgt_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    simd_bitmask::<i16x16, _>(simd_gt(a.as_i16x16(), b.as_i16x16()))
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpgt_epi16_mask&expand=896)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpgt_epi16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_cmpgt_epi16_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpgt_epi16_mask&expand=893)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpgt_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    simd_bitmask::<i16x8, _>(simd_gt(a.as_i16x8(), b.as_i16x8()))
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpgt_epi16_mask&expand=894)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpgt_epi16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_cmpgt_epi16_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpgt_epi8_mask&expand=921)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<i8x64, _>(simd_gt(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpgt_epi8_mask&expand=922)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpgt_epi8_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpgt_epi8_mask&expand=919)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpgt_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    simd_bitmask::<i8x32, _>(simd_gt(a.as_i8x32(), b.as_i8x32()))
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpgt_epi8_mask&expand=920)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpgt_epi8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_cmpgt_epi8_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpgt_epi8_mask&expand=917)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpgt_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    simd_bitmask::<i8x16, _>(simd_gt(a.as_i8x16(), b.as_i8x16()))
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpgt_epi8_mask&expand=918)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpgt_epi8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_cmpgt_epi8_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmple_epu16_mask&expand=989)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<u16x32, _>(simd_le(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmple_epu16_mask&expand=990)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmple_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmple_epu16_mask&expand=987)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmple_epu16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    simd_bitmask::<u16x16, _>(simd_le(a.as_u16x16(), b.as_u16x16()))
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmple_epu16_mask&expand=988)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmple_epu16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_cmple_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmple_epu16_mask&expand=985)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmple_epu16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    simd_bitmask::<u16x8, _>(simd_le(a.as_u16x8(), b.as_u16x8()))
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmple_epu16_mask&expand=986)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmple_epu16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_cmple_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.   
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmple_epu8_mask&expand=1007)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<u8x64, _>(simd_le(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmple_epu8_mask&expand=1008)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmple_epu8_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.   
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmple_epu8_mask&expand=1005)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmple_epu8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    simd_bitmask::<u8x32, _>(simd_le(a.as_u8x32(), b.as_u8x32()))
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmple_epu8_mask&expand=1006)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmple_epu8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_cmple_epu8_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.   
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmple_epu8_mask&expand=1003)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmple_epu8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    simd_bitmask::<u8x16, _>(simd_le(a.as_u8x16(), b.as_u8x16()))
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmple_epu8_mask&expand=1004)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmple_epu8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_cmple_epu8_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmple_epi16_mask&expand=965)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<i16x32, _>(simd_le(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmple_epi16_mask&expand=966)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmple_epi16_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmple_epi16_mask&expand=963)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmple_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    simd_bitmask::<i16x16, _>(simd_le(a.as_i16x16(), b.as_i16x16()))
}

/// Compare packed signed 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmple_epi16_mask&expand=964)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmple_epi16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_cmple_epi16_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmple_epi16_mask&expand=961)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmple_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    simd_bitmask::<i16x8, _>(simd_le(a.as_i16x8(), b.as_i16x8()))
}

/// Compare packed signed 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmple_epi16_mask&expand=962)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmple_epi16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_cmple_epi16_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmple_epi8_mask&expand=983)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<i8x64, _>(simd_le(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmple_epi8_mask&expand=984)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmple_epi8_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmple_epi8_mask&expand=981)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmple_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    simd_bitmask::<i8x32, _>(simd_le(a.as_i8x32(), b.as_i8x32()))
}

/// Compare packed signed 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmple_epi8_mask&expand=982)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmple_epi8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_cmple_epi8_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmple_epi8_mask&expand=979)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmple_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    simd_bitmask::<i8x16, _>(simd_le(a.as_i8x16(), b.as_i8x16()))
}

/// Compare packed signed 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmple_epi8_mask&expand=980)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmple_epi8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_cmple_epi8_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpge_epu16_mask&expand=867)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<u16x32, _>(simd_ge(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpge_epu16_mask&expand=868)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpge_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpge_epu16_mask&expand=865)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpge_epu16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    simd_bitmask::<u16x16, _>(simd_ge(a.as_u16x16(), b.as_u16x16()))
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpge_epu16_mask&expand=866)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpge_epu16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_cmpge_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpge_epu16_mask&expand=863)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpge_epu16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    simd_bitmask::<u16x8, _>(simd_ge(a.as_u16x8(), b.as_u16x8()))
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpge_epu16_mask&expand=864)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpge_epu16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_cmpge_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpge_epu8_mask&expand=885)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<u8x64, _>(simd_ge(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpge_epu8_mask&expand=886)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpge_epu8_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpge_epu8_mask&expand=883)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpge_epu8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    simd_bitmask::<u8x32, _>(simd_ge(a.as_u8x32(), b.as_u8x32()))
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpge_epu8_mask&expand=884)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpge_epu8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_cmpge_epu8_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpge_epu8_mask&expand=881)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpge_epu8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    simd_bitmask::<u8x16, _>(simd_ge(a.as_u8x16(), b.as_u8x16()))
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpge_epu8_mask&expand=882)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpge_epu8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_cmpge_epu8_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpge_epi16_mask&expand=843)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<i16x32, _>(simd_ge(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpge_epi16_mask&expand=844)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpge_epi16_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpge_epi16_mask&expand=841)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpge_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    simd_bitmask::<i16x16, _>(simd_ge(a.as_i16x16(), b.as_i16x16()))
}

/// Compare packed signed 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpge_epi16_mask&expand=842)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpge_epi16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_cmpge_epi16_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpge_epi16_mask&expand=839)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpge_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    simd_bitmask::<i16x8, _>(simd_ge(a.as_i16x8(), b.as_i16x8()))
}

/// Compare packed signed 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpge_epi16_mask&expand=840)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpge_epi16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_cmpge_epi16_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpge_epi8_mask&expand=861)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<i8x64, _>(simd_ge(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpge_epi8_mask&expand=862)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpge_epi8_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpge_epi8_mask&expand=859)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpge_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    simd_bitmask::<i8x32, _>(simd_ge(a.as_i8x32(), b.as_i8x32()))
}

/// Compare packed signed 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpge_epi8_mask&expand=860)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpge_epi8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_cmpge_epi8_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpge_epi8_mask&expand=857)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpge_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    simd_bitmask::<i8x16, _>(simd_ge(a.as_i8x16(), b.as_i8x16()))
}

/// Compare packed signed 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpge_epi8_mask&expand=858)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpge_epi8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_cmpge_epi8_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpeq_epu16_mask&expand=801)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<u16x32, _>(simd_eq(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpeq_epu16_mask&expand=802)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpeq_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpeq_epu16_mask&expand=799)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpeq_epu16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    simd_bitmask::<u16x16, _>(simd_eq(a.as_u16x16(), b.as_u16x16()))
}

/// Compare packed unsigned 16-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpeq_epu16_mask&expand=800)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpeq_epu16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_cmpeq_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpeq_epu16_mask&expand=797)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpeq_epu16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    simd_bitmask::<u16x8, _>(simd_eq(a.as_u16x8(), b.as_u16x8()))
}

/// Compare packed unsigned 16-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpeq_epu16_mask&expand=798)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpeq_epu16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_cmpeq_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpeq_epu8_mask&expand=819)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<u8x64, _>(simd_eq(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpeq_epu8_mask&expand=820)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpeq_epu8_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpeq_epu8_mask&expand=817)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpeq_epu8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    simd_bitmask::<u8x32, _>(simd_eq(a.as_u8x32(), b.as_u8x32()))
}

/// Compare packed unsigned 8-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpeq_epu8_mask&expand=818)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpeq_epu8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_cmpeq_epu8_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpeq_epu8_mask&expand=815)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpeq_epu8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    simd_bitmask::<u8x16, _>(simd_eq(a.as_u8x16(), b.as_u8x16()))
}

/// Compare packed unsigned 8-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpeq_epu8_mask&expand=816)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpeq_epu8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_cmpeq_epu8_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpeq_epi16_mask&expand=771)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<i16x32, _>(simd_eq(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpeq_epi16_mask&expand=772)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpeq_epi16_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpeq_epi16_mask&expand=769)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpeq_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    simd_bitmask::<i16x16, _>(simd_eq(a.as_i16x16(), b.as_i16x16()))
}

/// Compare packed signed 16-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpeq_epi16_mask&expand=770)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpeq_epi16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_cmpeq_epi16_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpeq_epi16_mask&expand=767)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpeq_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    simd_bitmask::<i16x8, _>(simd_eq(a.as_i16x8(), b.as_i16x8()))
}

/// Compare packed signed 16-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpeq_epi16_mask&expand=768)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpeq_epi16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_cmpeq_epi16_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpeq_epi8_mask&expand=795)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<i8x64, _>(simd_eq(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpeq_epi8_mask&expand=796)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpeq_epi8_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpeq_epi8_mask&expand=793)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpeq_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    simd_bitmask::<i8x32, _>(simd_eq(a.as_i8x32(), b.as_i8x32()))
}

/// Compare packed signed 8-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpeq_epi8_mask&expand=794)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpeq_epi8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_cmpeq_epi8_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpeq_epi8_mask&expand=791)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpeq_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    simd_bitmask::<i8x16, _>(simd_eq(a.as_i8x16(), b.as_i8x16()))
}

/// Compare packed signed 8-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpeq_epi8_mask&expand=792)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpeq_epi8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_cmpeq_epi8_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpneq_epu16_mask&expand=1106)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<u16x32, _>(simd_ne(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpneq_epu16_mask&expand=1107)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpneq_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpneq_epu16_mask&expand=1104)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpneq_epu16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    simd_bitmask::<u16x16, _>(simd_ne(a.as_u16x16(), b.as_u16x16()))
}

/// Compare packed unsigned 16-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpneq_epu16_mask&expand=1105)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpneq_epu16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_cmpneq_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpneq_epu16_mask&expand=1102)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpneq_epu16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    simd_bitmask::<u16x8, _>(simd_ne(a.as_u16x8(), b.as_u16x8()))
}

/// Compare packed unsigned 16-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpneq_epu16_mask&expand=1103)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpneq_epu16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_cmpneq_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpneq_epu8_mask&expand=1124)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<u8x64, _>(simd_ne(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpneq_epu8_mask&expand=1125)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpneq_epu8_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpneq_epu8_mask&expand=1122)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpneq_epu8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    simd_bitmask::<u8x32, _>(simd_ne(a.as_u8x32(), b.as_u8x32()))
}

/// Compare packed unsigned 8-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpneq_epu8_mask&expand=1123)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpneq_epu8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_cmpneq_epu8_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpneq_epu8_mask&expand=1120)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpneq_epu8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    simd_bitmask::<u8x16, _>(simd_ne(a.as_u8x16(), b.as_u8x16()))
}

/// Compare packed unsigned 8-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpneq_epu8_mask&expand=1121)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpneq_epu8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_cmpneq_epu8_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpneq_epi16_mask&expand=1082)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<i16x32, _>(simd_ne(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpneq_epi16_mask&expand=1083)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpneq_epi16_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpneq_epi16_mask&expand=1080)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpneq_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    simd_bitmask::<i16x16, _>(simd_ne(a.as_i16x16(), b.as_i16x16()))
}

/// Compare packed signed 16-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpneq_epi16_mask&expand=1081)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpneq_epi16_mask(k1: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    _mm256_cmpneq_epi16_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpneq_epi16_mask&expand=1078)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpneq_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    simd_bitmask::<i16x8, _>(simd_ne(a.as_i16x8(), b.as_i16x8()))
}

/// Compare packed signed 16-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpneq_epi16_mask&expand=1079)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpneq_epi16_mask(k1: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    _mm_cmpneq_epi16_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpneq_epi8_mask&expand=1100)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<i8x64, _>(simd_ne(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpneq_epi8_mask&expand=1101)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpneq_epi8_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpneq_epi8_mask&expand=1098)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_cmpneq_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    simd_bitmask::<i8x32, _>(simd_ne(a.as_i8x32(), b.as_i8x32()))
}

/// Compare packed signed 8-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmpneq_epi8_mask&expand=1099)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm256_mask_cmpneq_epi8_mask(k1: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    _mm256_cmpneq_epi8_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpneq_epi8_mask&expand=1096)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_cmpneq_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    simd_bitmask::<i8x16, _>(simd_ne(a.as_i8x16(), b.as_i8x16()))
}

/// Compare packed signed 8-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmpneq_epi8_mask&expand=1097)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm_mask_cmpneq_epi8_mask(k1: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    _mm_cmpneq_epi8_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b based on the comparison operand specified by `IMM8`, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmp_epu16_mask&expand=715)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub unsafe fn _mm512_cmp_epu16_mask<const IMM8: i32>(a: __m512i, b: __m512i) -> __mmask32 {
    static_assert_imm3!(IMM8);
    let a = a.as_u16x32();
    let b = b.as_u16x32();
    let r = vpcmpuw(a, b, IMM8, 0b11111111_11111111_11111111_11111111);
    transmute(r)
}

/// Compare packed unsigned 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmp_epu16_mask&expand=716)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub unsafe fn _mm512_mask_cmp_epu16_mask<const IMM8: i32>(
    k1: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __mmask32 {
    static_assert_imm3!(IMM8);
    let a = a.as_u16x32();
    let b = b.as_u16x32();
    let r = vpcmpuw(a, b, IMM8, k1);
    transmute(r)
}

/// Compare packed unsigned 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmp_epu16_mask&expand=713)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub unsafe fn _mm256_cmp_epu16_mask<const IMM8: i32>(a: __m256i, b: __m256i) -> __mmask16 {
    static_assert_imm3!(IMM8);
    let a = a.as_u16x16();
    let b = b.as_u16x16();
    let r = vpcmpuw256(a, b, IMM8, 0b11111111_11111111);
    transmute(r)
}

/// Compare packed unsigned 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmp_epu16_mask&expand=714)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vpcmp, IMM8 = 0))]
pub unsafe fn _mm256_mask_cmp_epu16_mask<const IMM8: i32>(
    k1: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __mmask16 {
    static_assert_imm3!(IMM8);
    let a = a.as_u16x16();
    let b = b.as_u16x16();
    let r = vpcmpuw256(a, b, IMM8, k1);
    transmute(r)
}

/// Compare packed unsigned 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmp_epu16_mask&expand=711)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm_cmp_epu16_mask(a: __m128i, b: __m128i, imm8: i32) -> __mmask8 {
    let a = a.as_u16x8();
    let b = b.as_u16x8();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpuw128(a, b, $imm3, 0b11111111)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed unsigned 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmp_epu16_mask&expand=712)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm_mask_cmp_epu16_mask(k1: __mmask8, a: __m128i, b: __m128i, imm8: i32) -> __mmask8 {
    let a = a.as_u16x8();
    let b = b.as_u16x8();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpuw128(a, b, $imm3, k1)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed unsigned 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmp_epu8_mask&expand=733)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_cmp_epu8_mask(a: __m512i, b: __m512i, imm8: i32) -> __mmask64 {
    let a = a.as_u8x64();
    let b = b.as_u8x64();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpub(
                a,
                b,
                $imm3,
                0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            )
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed unsigned 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmp_epu8_mask&expand=734)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_mask_cmp_epu8_mask(
    k1: __mmask64,
    a: __m512i,
    b: __m512i,
    imm8: i32,
) -> __mmask64 {
    let a = a.as_u8x64();
    let b = b.as_u8x64();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpub(a, b, $imm3, k1)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed unsigned 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmp_epu8_mask&expand=731)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm256_cmp_epu8_mask(a: __m256i, b: __m256i, imm8: i32) -> __mmask32 {
    let a = a.as_u8x32();
    let b = b.as_u8x32();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpub256(a, b, $imm3, 0b11111111_11111111_11111111_11111111)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed unsigned 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmp_epu8_mask&expand=732)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm256_mask_cmp_epu8_mask(
    k1: __mmask32,
    a: __m256i,
    b: __m256i,
    imm8: i32,
) -> __mmask32 {
    let a = a.as_u8x32();
    let b = b.as_u8x32();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpub256(a, b, $imm3, k1)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed unsigned 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmp_epu8_mask&expand=729)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm_cmp_epu8_mask(a: __m128i, b: __m128i, imm8: i32) -> __mmask16 {
    let a = a.as_u8x16();
    let b = b.as_u8x16();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpub128(a, b, $imm3, 0b11111111_11111111)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed unsigned 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmp_epu8_mask&expand=730)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm_mask_cmp_epu8_mask(
    k1: __mmask16,
    a: __m128i,
    b: __m128i,
    imm8: i32,
) -> __mmask16 {
    let a = a.as_u8x16();
    let b = b.as_u8x16();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpub128(a, b, $imm3, k1)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmp_epi16_mask&expand=691)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_cmp_epi16_mask(a: __m512i, b: __m512i, imm8: i32) -> __mmask32 {
    let a = a.as_i16x32();
    let b = b.as_i16x32();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpw(a, b, $imm3, 0b11111111_11111111_11111111_11111111)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmp_epi16_mask&expand=692)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_mask_cmp_epi16_mask(
    k1: __mmask32,
    a: __m512i,
    b: __m512i,
    imm8: i32,
) -> __mmask32 {
    let a = a.as_i16x32();
    let b = b.as_i16x32();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpw(a, b, $imm3, k1)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmp_epi16_mask&expand=689)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm256_cmp_epi16_mask(a: __m256i, b: __m256i, imm8: i32) -> __mmask16 {
    let a = a.as_i16x16();
    let b = b.as_i16x16();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpw256(a, b, $imm3, 0b11111111_11111111)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmp_epi16_mask&expand=690)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm256_mask_cmp_epi16_mask(
    k1: __mmask16,
    a: __m256i,
    b: __m256i,
    imm8: i32,
) -> __mmask16 {
    let a = a.as_i16x16();
    let b = b.as_i16x16();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpw256(a, b, $imm3, k1)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmp_epi16_mask&expand=687)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm_cmp_epi16_mask(a: __m128i, b: __m128i, imm8: i32) -> __mmask8 {
    let a = a.as_i16x8();
    let b = b.as_i16x8();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpw128(a, b, $imm3, 0b11111111)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmp_epi16_mask&expand=688)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm_mask_cmp_epi16_mask(k1: __mmask8, a: __m128i, b: __m128i, imm8: i32) -> __mmask8 {
    let a = a.as_i16x8();
    let b = b.as_i16x8();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpw128(a, b, $imm3, k1)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmp_epi8_mask&expand=709)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_cmp_epi8_mask(a: __m512i, b: __m512i, imm8: i32) -> __mmask64 {
    let a = a.as_i8x64();
    let b = b.as_i8x64();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpb(
                a,
                b,
                $imm3,
                0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            )
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmp_epi8_mask&expand=710)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_mask_cmp_epi8_mask(
    k1: __mmask64,
    a: __m512i,
    b: __m512i,
    imm8: i32,
) -> __mmask64 {
    let a = a.as_i8x64();
    let b = b.as_i8x64();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpb(a, b, $imm3, k1)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmp_epi8_mask&expand=707)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm256_cmp_epi8_mask(a: __m256i, b: __m256i, imm8: i32) -> __mmask32 {
    let a = a.as_i8x32();
    let b = b.as_i8x32();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpb256(a, b, $imm3, 0b11111111_11111111_11111111_11111111)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cmp_epi8_mask&expand=708)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm256_mask_cmp_epi8_mask(
    k1: __mmask32,
    a: __m256i,
    b: __m256i,
    imm8: i32,
) -> __mmask32 {
    let a = a.as_i8x32();
    let b = b.as_i8x32();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpb256(a, b, $imm3, k1)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmp_epi8_mask&expand=705)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm_cmp_epi8_mask(a: __m128i, b: __m128i, imm8: i32) -> __mmask16 {
    let a = a.as_i8x16();
    let b = b.as_i8x16();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpb128(a, b, $imm3, 0b11111111_11111111)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cmp_epi8_mask&expand=706)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm_mask_cmp_epi8_mask(
    k1: __mmask16,
    a: __m128i,
    b: __m128i,
    imm8: i32,
) -> __mmask16 {
    let a = a.as_i8x16();
    let b = b.as_i8x16();
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpb128(a, b, $imm3, k1)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Load 512-bits (composed of 32 packed 16-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_loadu_epi16&expand=3368)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu16
pub unsafe fn _mm512_loadu_epi16(mem_addr: *const i16) -> __m512i {
    ptr::read_unaligned(mem_addr as *const __m512i)
}

/// Load 256-bits (composed of 16 packed 16-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_loadu_epi16&expand=3365)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu16
pub unsafe fn _mm256_loadu_epi16(mem_addr: *const i16) -> __m256i {
    ptr::read_unaligned(mem_addr as *const __m256i)
}

/// Load 128-bits (composed of 8 packed 16-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_loadu_epi16&expand=3362)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu16
pub unsafe fn _mm_loadu_epi16(mem_addr: *const i16) -> __m128i {
    ptr::read_unaligned(mem_addr as *const __m128i)
}

/// Load 512-bits (composed of 64 packed 8-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_loadu_epi8&expand=3395)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu8
pub unsafe fn _mm512_loadu_epi8(mem_addr: *const i8) -> __m512i {
    ptr::read_unaligned(mem_addr as *const __m512i)
}

/// Load 256-bits (composed of 32 packed 8-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_loadu_epi8&expand=3392)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu8
pub unsafe fn _mm256_loadu_epi8(mem_addr: *const i8) -> __m256i {
    ptr::read_unaligned(mem_addr as *const __m256i)
}

/// Load 128-bits (composed of 16 packed 8-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_loadu_epi8&expand=3389)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu8
pub unsafe fn _mm_loadu_epi8(mem_addr: *const i8) -> __m128i {
    ptr::read_unaligned(mem_addr as *const __m128i)
}

/// Store 512-bits (composed of 32 packed 16-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_storeu_epi16&expand=5622)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu16
pub unsafe fn _mm512_storeu_epi16(mem_addr: *mut i16, a: __m512i) {
    ptr::write_unaligned(mem_addr as *mut __m512i, a);
}

/// Store 256-bits (composed of 16 packed 16-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_storeu_epi16&expand=5620)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu16
pub unsafe fn _mm256_storeu_epi16(mem_addr: *mut i16, a: __m256i) {
    ptr::write_unaligned(mem_addr as *mut __m256i, a);
}

/// Store 128-bits (composed of 8 packed 16-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_storeu_epi16&expand=5618)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu16
pub unsafe fn _mm_storeu_epi16(mem_addr: *mut i16, a: __m128i) {
    ptr::write_unaligned(mem_addr as *mut __m128i, a);
}

/// Store 512-bits (composed of 64 packed 8-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_storeu_epi8&expand=5640)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu8
pub unsafe fn _mm512_storeu_epi8(mem_addr: *mut i8, a: __m512i) {
    ptr::write_unaligned(mem_addr as *mut __m512i, a);
}

/// Store 256-bits (composed of 32 packed 8-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_storeu_epi8&expand=5638)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu8
pub unsafe fn _mm256_storeu_epi8(mem_addr: *mut i8, a: __m256i) {
    ptr::write_unaligned(mem_addr as *mut __m256i, a);
}

/// Store 128-bits (composed of 16 packed 8-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_storeu_epi8&expand=5636)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu8
pub unsafe fn _mm_storeu_epi8(mem_addr: *mut i8, a: __m128i) {
    ptr::write_unaligned(mem_addr as *mut __m128i, a);
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_madd_epi16&expand=3511)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub unsafe fn _mm512_madd_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaddwd(a.as_i16x32(), b.as_i16x32()))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_madd_epi16&expand=3512)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub unsafe fn _mm512_mask_madd_epi16(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let madd = _mm512_madd_epi16(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, madd, src.as_i32x16()))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_madd_epi16&expand=3513)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub unsafe fn _mm512_maskz_madd_epi16(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let madd = _mm512_madd_epi16(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, madd, zero))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_madd_epi16&expand=3509)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub unsafe fn _mm256_mask_madd_epi16(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    let madd = _mm256_madd_epi16(a, b).as_i32x8();
    transmute(simd_select_bitmask(k, madd, src.as_i32x8()))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_madd_epi16&expand=3510)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub unsafe fn _mm256_maskz_madd_epi16(k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    let madd = _mm256_madd_epi16(a, b).as_i32x8();
    let zero = _mm256_setzero_si256().as_i32x8();
    transmute(simd_select_bitmask(k, madd, zero))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_madd_epi16&expand=3506)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub unsafe fn _mm_mask_madd_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let madd = _mm_madd_epi16(a, b).as_i32x4();
    transmute(simd_select_bitmask(k, madd, src.as_i32x4()))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers, and pack the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_madd_epi16&expand=3507)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub unsafe fn _mm_maskz_madd_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let madd = _mm_madd_epi16(a, b).as_i32x4();
    let zero = _mm_setzero_si128().as_i32x4();
    transmute(simd_select_bitmask(k, madd, zero))
}

/// Vertically multiply each unsigned 8-bit integer from a with the corresponding signed 8-bit integer from b, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers, and pack the saturated results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maddubs_epi16&expand=3539)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub unsafe fn _mm512_maddubs_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaddubsw(a.as_i8x64(), b.as_i8x64()))
}

/// Multiply packed unsigned 8-bit integers in a by packed signed 8-bit integers in b, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers, and pack the saturated results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_maddubs_epi16&expand=3540)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub unsafe fn _mm512_mask_maddubs_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let madd = _mm512_maddubs_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, madd, src.as_i16x32()))
}

/// Multiply packed unsigned 8-bit integers in a by packed signed 8-bit integers in b, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers, and pack the saturated results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_maddubs_epi16&expand=3541)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub unsafe fn _mm512_maskz_maddubs_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let madd = _mm512_maddubs_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, madd, zero))
}

/// Multiply packed unsigned 8-bit integers in a by packed signed 8-bit integers in b, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers, and pack the saturated results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_maddubs_epi16&expand=3537)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub unsafe fn _mm256_mask_maddubs_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    let madd = _mm256_maddubs_epi16(a, b).as_i16x16();
    transmute(simd_select_bitmask(k, madd, src.as_i16x16()))
}

/// Multiply packed unsigned 8-bit integers in a by packed signed 8-bit integers in b, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers, and pack the saturated results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_maddubs_epi16&expand=3538)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub unsafe fn _mm256_maskz_maddubs_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let madd = _mm256_maddubs_epi16(a, b).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, madd, zero))
}

/// Multiply packed unsigned 8-bit integers in a by packed signed 8-bit integers in b, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers, and pack the saturated results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_maddubs_epi16&expand=3534)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub unsafe fn _mm_mask_maddubs_epi16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let madd = _mm_maddubs_epi16(a, b).as_i16x8();
    transmute(simd_select_bitmask(k, madd, src.as_i16x8()))
}

/// Multiply packed unsigned 8-bit integers in a by packed signed 8-bit integers in b, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers, and pack the saturated results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_maddubs_epi16&expand=3535)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub unsafe fn _mm_maskz_maddubs_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let madd = _mm_maddubs_epi16(a, b).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, madd, zero))
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_packs_epi32&expand=4091)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub unsafe fn _mm512_packs_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpackssdw(a.as_i32x16(), b.as_i32x16()))
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_packs_epi32&expand=4089)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub unsafe fn _mm512_mask_packs_epi32(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let pack = _mm512_packs_epi32(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, pack, src.as_i16x32()))
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_packs_epi32&expand=4090)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub unsafe fn _mm512_maskz_packs_epi32(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let pack = _mm512_packs_epi32(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, pack, zero))
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_packs_epi32&expand=4086)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub unsafe fn _mm256_mask_packs_epi32(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    let pack = _mm256_packs_epi32(a, b).as_i16x16();
    transmute(simd_select_bitmask(k, pack, src.as_i16x16()))
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_packs_epi32&expand=4087)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub unsafe fn _mm256_maskz_packs_epi32(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let pack = _mm256_packs_epi32(a, b).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, pack, zero))
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_packs_epi32&expand=4083)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub unsafe fn _mm_mask_packs_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let pack = _mm_packs_epi32(a, b).as_i16x8();
    transmute(simd_select_bitmask(k, pack, src.as_i16x8()))
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_packs_epi32&expand=4084)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub unsafe fn _mm_maskz_packs_epi32(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let pack = _mm_packs_epi32(a, b).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, pack, zero))
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_packs_epi16&expand=4082)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub unsafe fn _mm512_packs_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpacksswb(a.as_i16x32(), b.as_i16x32()))
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_packs_epi16&expand=4080)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub unsafe fn _mm512_mask_packs_epi16(
    src: __m512i,
    k: __mmask64,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let pack = _mm512_packs_epi16(a, b).as_i8x64();
    transmute(simd_select_bitmask(k, pack, src.as_i8x64()))
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_packs_epi16&expand=4081)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub unsafe fn _mm512_maskz_packs_epi16(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let pack = _mm512_packs_epi16(a, b).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, pack, zero))
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_packs_epi16&expand=4077)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub unsafe fn _mm256_mask_packs_epi16(
    src: __m256i,
    k: __mmask32,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    let pack = _mm256_packs_epi16(a, b).as_i8x32();
    transmute(simd_select_bitmask(k, pack, src.as_i8x32()))
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=#text=_mm256_maskz_packs_epi16&expand=4078)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub unsafe fn _mm256_maskz_packs_epi16(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let pack = _mm256_packs_epi16(a, b).as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, pack, zero))
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_packs_epi16&expand=4074)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub unsafe fn _mm_mask_packs_epi16(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let pack = _mm_packs_epi16(a, b).as_i8x16();
    transmute(simd_select_bitmask(k, pack, src.as_i8x16()))
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_packs_epi16&expand=4075)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub unsafe fn _mm_maskz_packs_epi16(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let pack = _mm_packs_epi16(a, b).as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, pack, zero))
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_packus_epi32&expand=4130)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub unsafe fn _mm512_packus_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpackusdw(a.as_i32x16(), b.as_i32x16()))
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_packus_epi32&expand=4128)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub unsafe fn _mm512_mask_packus_epi32(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let pack = _mm512_packus_epi32(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, pack, src.as_i16x32()))
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_packus_epi32&expand=4129)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub unsafe fn _mm512_maskz_packus_epi32(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let pack = _mm512_packus_epi32(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, pack, zero))
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_packus_epi32&expand=4125)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub unsafe fn _mm256_mask_packus_epi32(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    let pack = _mm256_packus_epi32(a, b).as_i16x16();
    transmute(simd_select_bitmask(k, pack, src.as_i16x16()))
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_packus_epi32&expand=4126)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub unsafe fn _mm256_maskz_packus_epi32(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let pack = _mm256_packus_epi32(a, b).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, pack, zero))
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_packus_epi32&expand=4122)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub unsafe fn _mm_mask_packus_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let pack = _mm_packus_epi32(a, b).as_i16x8();
    transmute(simd_select_bitmask(k, pack, src.as_i16x8()))
}

/// Convert packed signed 32-bit integers from a and b to packed 16-bit integers using unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_packus_epi32&expand=4123)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub unsafe fn _mm_maskz_packus_epi32(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let pack = _mm_packus_epi32(a, b).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, pack, zero))
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_packus_epi16&expand=4121)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub unsafe fn _mm512_packus_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpackuswb(a.as_i16x32(), b.as_i16x32()))
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_packus_epi16&expand=4119)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub unsafe fn _mm512_mask_packus_epi16(
    src: __m512i,
    k: __mmask64,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let pack = _mm512_packus_epi16(a, b).as_i8x64();
    transmute(simd_select_bitmask(k, pack, src.as_i8x64()))
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_packus_epi16&expand=4120)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub unsafe fn _mm512_maskz_packus_epi16(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let pack = _mm512_packus_epi16(a, b).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, pack, zero))
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_packus_epi16&expand=4116)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub unsafe fn _mm256_mask_packus_epi16(
    src: __m256i,
    k: __mmask32,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    let pack = _mm256_packus_epi16(a, b).as_i8x32();
    transmute(simd_select_bitmask(k, pack, src.as_i8x32()))
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_packus_epi16&expand=4117)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub unsafe fn _mm256_maskz_packus_epi16(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let pack = _mm256_packus_epi16(a, b).as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, pack, zero))
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_packus_epi16&expand=4113)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub unsafe fn _mm_mask_packus_epi16(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let pack = _mm_packus_epi16(a, b).as_i8x16();
    transmute(simd_select_bitmask(k, pack, src.as_i8x16()))
}

/// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_packus_epi16&expand=4114)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub unsafe fn _mm_maskz_packus_epi16(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let pack = _mm_packus_epi16(a, b).as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, pack, zero))
}

/// Average packed unsigned 16-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_avg_epu16&expand=388)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpavgw))]
pub unsafe fn _mm512_avg_epu16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpavgw(a.as_u16x32(), b.as_u16x32()))
}

/// Average packed unsigned 16-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_avg_epu16&expand=389)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpavgw))]
pub unsafe fn _mm512_mask_avg_epu16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let avg = _mm512_avg_epu16(a, b).as_u16x32();
    transmute(simd_select_bitmask(k, avg, src.as_u16x32()))
}

/// Average packed unsigned 16-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_avg_epu16&expand=390)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpavgw))]
pub unsafe fn _mm512_maskz_avg_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let avg = _mm512_avg_epu16(a, b).as_u16x32();
    let zero = _mm512_setzero_si512().as_u16x32();
    transmute(simd_select_bitmask(k, avg, zero))
}

/// Average packed unsigned 16-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_avg_epu16&expand=386)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpavgw))]
pub unsafe fn _mm256_mask_avg_epu16(src: __m256i, k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let avg = _mm256_avg_epu16(a, b).as_u16x16();
    transmute(simd_select_bitmask(k, avg, src.as_u16x16()))
}

/// Average packed unsigned 16-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_avg_epu16&expand=387)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpavgw))]
pub unsafe fn _mm256_maskz_avg_epu16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let avg = _mm256_avg_epu16(a, b).as_u16x16();
    let zero = _mm256_setzero_si256().as_u16x16();
    transmute(simd_select_bitmask(k, avg, zero))
}

/// Average packed unsigned 16-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_avg_epu16&expand=383)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpavgw))]
pub unsafe fn _mm_mask_avg_epu16(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let avg = _mm_avg_epu16(a, b).as_u16x8();
    transmute(simd_select_bitmask(k, avg, src.as_u16x8()))
}

/// Average packed unsigned 16-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_avg_epu16&expand=384)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpavgw))]
pub unsafe fn _mm_maskz_avg_epu16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let avg = _mm_avg_epu16(a, b).as_u16x8();
    let zero = _mm_setzero_si128().as_u16x8();
    transmute(simd_select_bitmask(k, avg, zero))
}

/// Average packed unsigned 8-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_avg_epu8&expand=397)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpavgb))]
pub unsafe fn _mm512_avg_epu8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpavgb(a.as_u8x64(), b.as_u8x64()))
}

/// Average packed unsigned 8-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_avg_epu8&expand=398)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpavgb))]
pub unsafe fn _mm512_mask_avg_epu8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let avg = _mm512_avg_epu8(a, b).as_u8x64();
    transmute(simd_select_bitmask(k, avg, src.as_u8x64()))
}

/// Average packed unsigned 8-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_avg_epu8&expand=399)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpavgb))]
pub unsafe fn _mm512_maskz_avg_epu8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let avg = _mm512_avg_epu8(a, b).as_u8x64();
    let zero = _mm512_setzero_si512().as_u8x64();
    transmute(simd_select_bitmask(k, avg, zero))
}

/// Average packed unsigned 8-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_avg_epu8&expand=395)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpavgb))]
pub unsafe fn _mm256_mask_avg_epu8(src: __m256i, k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let avg = _mm256_avg_epu8(a, b).as_u8x32();
    transmute(simd_select_bitmask(k, avg, src.as_u8x32()))
}

/// Average packed unsigned 8-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_avg_epu8&expand=396)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpavgb))]
pub unsafe fn _mm256_maskz_avg_epu8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let avg = _mm256_avg_epu8(a, b).as_u8x32();
    let zero = _mm256_setzero_si256().as_u8x32();
    transmute(simd_select_bitmask(k, avg, zero))
}

/// Average packed unsigned 8-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_avg_epu8&expand=392)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpavgb))]
pub unsafe fn _mm_mask_avg_epu8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let avg = _mm_avg_epu8(a, b).as_u8x16();
    transmute(simd_select_bitmask(k, avg, src.as_u8x16()))
}

/// Average packed unsigned 8-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_avg_epu8&expand=393)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpavgb))]
pub unsafe fn _mm_maskz_avg_epu8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let avg = _mm_avg_epu8(a, b).as_u8x16();
    let zero = _mm_setzero_si128().as_u8x16();
    transmute(simd_select_bitmask(k, avg, zero))
}

/// Shift packed 16-bit integers in a left by count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_sll_epi16&expand=5271)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsllw))]
pub unsafe fn _mm512_sll_epi16(a: __m512i, count: __m128i) -> __m512i {
    transmute(vpsllw(a.as_i16x32(), count.as_i16x8()))
}

/// Shift packed 16-bit integers in a left by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_sll_epi16&expand=5269)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsllw))]
pub unsafe fn _mm512_mask_sll_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    count: __m128i,
) -> __m512i {
    let shf = _mm512_sll_epi16(a, count).as_i16x32();
    transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
}

/// Shift packed 16-bit integers in a left by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_sll_epi16&expand=5270)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsllw))]
pub unsafe fn _mm512_maskz_sll_epi16(k: __mmask32, a: __m512i, count: __m128i) -> __m512i {
    let shf = _mm512_sll_epi16(a, count).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a left by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_sll_epi16&expand=5266)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsllw))]
pub unsafe fn _mm256_mask_sll_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    count: __m128i,
) -> __m256i {
    let shf = _mm256_sll_epi16(a, count).as_i16x16();
    transmute(simd_select_bitmask(k, shf, src.as_i16x16()))
}

/// Shift packed 16-bit integers in a left by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_sll_epi16&expand=5267)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsllw))]
pub unsafe fn _mm256_maskz_sll_epi16(k: __mmask16, a: __m256i, count: __m128i) -> __m256i {
    let shf = _mm256_sll_epi16(a, count).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a left by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_sll_epi16&expand=5263)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsllw))]
pub unsafe fn _mm_mask_sll_epi16(src: __m128i, k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    let shf = _mm_sll_epi16(a, count).as_i16x8();
    transmute(simd_select_bitmask(k, shf, src.as_i16x8()))
}

/// Shift packed 16-bit integers in a left by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_sll_epi16&expand=5264)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsllw))]
pub unsafe fn _mm_maskz_sll_epi16(k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    let shf = _mm_sll_epi16(a, count).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_slli_epi16&expand=5301)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsllw, imm8 = 5))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_slli_epi16(a: __m512i, imm8: u32) -> __m512i {
    let a = a.as_i16x32();
    macro_rules! call {
        ($imm8:expr) => {
            vpslliw(a, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_slli_epi16&expand=5299)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsllw, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_slli_epi16(src: __m512i, k: __mmask32, a: __m512i, imm8: u32) -> __m512i {
    let a = a.as_i16x32();
    macro_rules! call {
        ($imm8:expr) => {
            vpslliw(a, $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
}

/// Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_slli_epi16&expand=5300)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsllw, imm8 = 5))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_slli_epi16(k: __mmask32, a: __m512i, imm8: u32) -> __m512i {
    let a = a.as_i16x32();
    macro_rules! call {
        ($imm8:expr) => {
            vpslliw(a, $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_slli_epi16&expand=5296)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsllw, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm256_mask_slli_epi16(src: __m256i, k: __mmask16, a: __m256i, imm8: u32) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm256_slli_epi16::<$imm8>(a)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shf.as_i16x16(), src.as_i16x16()))
}

/// Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_slli_epi16&expand=5297)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsllw, imm8 = 5))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm256_maskz_slli_epi16(k: __mmask16, a: __m256i, imm8: u32) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm256_slli_epi16::<$imm8>(a)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, shf.as_i16x16(), zero))
}

/// Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_slli_epi16&expand=5293)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsllw, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm_mask_slli_epi16(src: __m128i, k: __mmask8, a: __m128i, imm8: u32) -> __m128i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm_slli_epi16::<$imm8>(a)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shf.as_i16x8(), src.as_i16x8()))
}

/// Shift packed 16-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_slli_epi16&expand=5294)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsllw, imm8 = 5))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_maskz_slli_epi16(k: __mmask8, a: __m128i, imm8: u32) -> __m128i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm_slli_epi16::<$imm8>(a)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, shf.as_i16x8(), zero))
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_sllv_epi16&expand=5333)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub unsafe fn _mm512_sllv_epi16(a: __m512i, count: __m512i) -> __m512i {
    transmute(vpsllvw(a.as_i16x32(), count.as_i16x32()))
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_sllv_epi16&expand=5331)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub unsafe fn _mm512_mask_sllv_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    count: __m512i,
) -> __m512i {
    let shf = _mm512_sllv_epi16(a, count).as_i16x32();
    transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_sllv_epi16&expand=5332)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub unsafe fn _mm512_maskz_sllv_epi16(k: __mmask32, a: __m512i, count: __m512i) -> __m512i {
    let shf = _mm512_sllv_epi16(a, count).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sllv_epi16&expand=5330)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub unsafe fn _mm256_sllv_epi16(a: __m256i, count: __m256i) -> __m256i {
    transmute(vpsllvw256(a.as_i16x16(), count.as_i16x16()))
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_sllv_epi16&expand=5328)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub unsafe fn _mm256_mask_sllv_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    count: __m256i,
) -> __m256i {
    let shf = _mm256_sllv_epi16(a, count).as_i16x16();
    transmute(simd_select_bitmask(k, shf, src.as_i16x16()))
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_sllv_epi16&expand=5329)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub unsafe fn _mm256_maskz_sllv_epi16(k: __mmask16, a: __m256i, count: __m256i) -> __m256i {
    let shf = _mm256_sllv_epi16(a, count).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sllv_epi16&expand=5327)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub unsafe fn _mm_sllv_epi16(a: __m128i, count: __m128i) -> __m128i {
    transmute(vpsllvw128(a.as_i16x8(), count.as_i16x8()))
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_sllv_epi16&expand=5325)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub unsafe fn _mm_mask_sllv_epi16(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    count: __m128i,
) -> __m128i {
    let shf = _mm_sllv_epi16(a, count).as_i16x8();
    transmute(simd_select_bitmask(k, shf, src.as_i16x8()))
}

/// Shift packed 16-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_sllv_epi16&expand=5326)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsllvw))]
pub unsafe fn _mm_maskz_sllv_epi16(k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    let shf = _mm_sllv_epi16(a, count).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a right by count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_srl_epi16&expand=5483)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub unsafe fn _mm512_srl_epi16(a: __m512i, count: __m128i) -> __m512i {
    transmute(vpsrlw(a.as_i16x32(), count.as_i16x8()))
}

/// Shift packed 16-bit integers in a right by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_srl_epi16&expand=5481)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub unsafe fn _mm512_mask_srl_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    count: __m128i,
) -> __m512i {
    let shf = _mm512_srl_epi16(a, count).as_i16x32();
    transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
}

/// Shift packed 16-bit integers in a right by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_srl_epi16&expand=5482)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub unsafe fn _mm512_maskz_srl_epi16(k: __mmask32, a: __m512i, count: __m128i) -> __m512i {
    let shf = _mm512_srl_epi16(a, count).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a right by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_srl_epi16&expand=5478)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub unsafe fn _mm256_mask_srl_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    count: __m128i,
) -> __m256i {
    let shf = _mm256_srl_epi16(a, count).as_i16x16();
    transmute(simd_select_bitmask(k, shf, src.as_i16x16()))
}

/// Shift packed 16-bit integers in a right by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_srl_epi16&expand=5479)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub unsafe fn _mm256_maskz_srl_epi16(k: __mmask16, a: __m256i, count: __m128i) -> __m256i {
    let shf = _mm256_srl_epi16(a, count).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a right by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_srl_epi16&expand=5475)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub unsafe fn _mm_mask_srl_epi16(src: __m128i, k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    let shf = _mm_srl_epi16(a, count).as_i16x8();
    transmute(simd_select_bitmask(k, shf, src.as_i16x8()))
}

/// Shift packed 16-bit integers in a right by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_srl_epi16&expand=5476)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub unsafe fn _mm_maskz_srl_epi16(k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    let shf = _mm_srl_epi16(a, count).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_srli_epi16&expand=5513)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsrlw, imm8 = 5))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_srli_epi16(a: __m512i, imm8: u32) -> __m512i {
    let a = a.as_i16x32();
    macro_rules! call {
        ($imm8:expr) => {
            vpsrliw(a, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_srli_epi16&expand=5511)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsrlw, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_srli_epi16(src: __m512i, k: __mmask32, a: __m512i, imm8: u32) -> __m512i {
    let a = a.as_i16x32();
    macro_rules! call {
        ($imm8:expr) => {
            vpsrliw(a, $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_srli_epi16&expand=5512)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsrlw, imm8 = 5))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_srli_epi16(k: __mmask32, a: __m512i, imm8: i32) -> __m512i {
    //imm8 should be u32, it seems the document to verify is incorrect
    let a = a.as_i16x32();
    macro_rules! call {
        ($imm8:expr) => {
            vpsrliw(a, $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_srli_epi16&expand=5508)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsrlw, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm256_mask_srli_epi16(src: __m256i, k: __mmask16, a: __m256i, imm8: i32) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm256_srli_epi16::<$imm8>(a)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shf.as_i16x16(), src.as_i16x16()))
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_srli_epi16&expand=5509)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsrlw, imm8 = 5))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm256_maskz_srli_epi16(k: __mmask16, a: __m256i, imm8: i32) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm256_srli_epi16::<$imm8>(a)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, shf.as_i16x16(), zero))
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_srli_epi16&expand=5505)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsrlw, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm_mask_srli_epi16(src: __m128i, k: __mmask8, a: __m128i, imm8: i32) -> __m128i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm_srli_epi16::<$imm8>(a)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shf.as_i16x8(), src.as_i16x8()))
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_srli_epi16&expand=5506)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsrlw, imm8 = 5))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_maskz_srli_epi16(k: __mmask8, a: __m128i, imm8: i32) -> __m128i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm_srli_epi16::<$imm8>(a)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, shf.as_i16x8(), zero))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_srlv_epi16&expand=5545)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub unsafe fn _mm512_srlv_epi16(a: __m512i, count: __m512i) -> __m512i {
    transmute(vpsrlvw(a.as_i16x32(), count.as_i16x32()))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_srlv_epi16&expand=5543)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub unsafe fn _mm512_mask_srlv_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    count: __m512i,
) -> __m512i {
    let shf = _mm512_srlv_epi16(a, count).as_i16x32();
    transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_srlv_epi16&expand=5544)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub unsafe fn _mm512_maskz_srlv_epi16(k: __mmask32, a: __m512i, count: __m512i) -> __m512i {
    let shf = _mm512_srlv_epi16(a, count).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_srlv_epi16&expand=5542)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub unsafe fn _mm256_srlv_epi16(a: __m256i, count: __m256i) -> __m256i {
    transmute(vpsrlvw256(a.as_i16x16(), count.as_i16x16()))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_srlv_epi16&expand=5540)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub unsafe fn _mm256_mask_srlv_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    count: __m256i,
) -> __m256i {
    let shf = _mm256_srlv_epi16(a, count).as_i16x16();
    transmute(simd_select_bitmask(k, shf, src.as_i16x16()))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_srlv_epi16&expand=5541)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub unsafe fn _mm256_maskz_srlv_epi16(k: __mmask16, a: __m256i, count: __m256i) -> __m256i {
    let shf = _mm256_srlv_epi16(a, count).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_srlv_epi16&expand=5539)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub unsafe fn _mm_srlv_epi16(a: __m128i, count: __m128i) -> __m128i {
    transmute(vpsrlvw128(a.as_i16x8(), count.as_i16x8()))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_srlv_epi16&expand=5537)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub unsafe fn _mm_mask_srlv_epi16(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    count: __m128i,
) -> __m128i {
    let shf = _mm_srlv_epi16(a, count).as_i16x8();
    transmute(simd_select_bitmask(k, shf, src.as_i16x8()))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_srlv_epi16&expand=5538)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsrlvw))]
pub unsafe fn _mm_maskz_srlv_epi16(k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    let shf = _mm_srlv_epi16(a, count).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a right by count while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_sra_epi16&expand=5398)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsraw))]
pub unsafe fn _mm512_sra_epi16(a: __m512i, count: __m128i) -> __m512i {
    transmute(vpsraw(a.as_i16x32(), count.as_i16x8()))
}

/// Shift packed 16-bit integers in a right by count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_sra_epi16&expand=5396)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsraw))]
pub unsafe fn _mm512_mask_sra_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    count: __m128i,
) -> __m512i {
    let shf = _mm512_sra_epi16(a, count).as_i16x32();
    transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
}

/// Shift packed 16-bit integers in a right by count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_sra_epi16&expand=5397)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsraw))]
pub unsafe fn _mm512_maskz_sra_epi16(k: __mmask32, a: __m512i, count: __m128i) -> __m512i {
    let shf = _mm512_sra_epi16(a, count).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a right by count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_sra_epi16&expand=5393)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsraw))]
pub unsafe fn _mm256_mask_sra_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    count: __m128i,
) -> __m256i {
    let shf = _mm256_sra_epi16(a, count).as_i16x16();
    transmute(simd_select_bitmask(k, shf, src.as_i16x16()))
}

/// Shift packed 16-bit integers in a right by count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_sra_epi16&expand=5394)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsraw))]
pub unsafe fn _mm256_maskz_sra_epi16(k: __mmask16, a: __m256i, count: __m128i) -> __m256i {
    let shf = _mm256_sra_epi16(a, count).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a right by count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_sra_epi16&expand=5390)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsraw))]
pub unsafe fn _mm_mask_sra_epi16(src: __m128i, k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    let shf = _mm_sra_epi16(a, count).as_i16x8();
    transmute(simd_select_bitmask(k, shf, src.as_i16x8()))
}

/// Shift packed 16-bit integers in a right by count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_sra_epi16&expand=5391)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsraw))]
pub unsafe fn _mm_maskz_sra_epi16(k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    let shf = _mm_sra_epi16(a, count).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_srai_epi16&expand=5427)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsraw, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_srai_epi16(a: __m512i, imm8: u32) -> __m512i {
    let a = a.as_i16x32();
    macro_rules! call {
        ($imm8:expr) => {
            vpsraiw(a, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_srai_epi16&expand=5425)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsraw, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_srai_epi16(src: __m512i, k: __mmask32, a: __m512i, imm8: u32) -> __m512i {
    let a = a.as_i16x32();
    macro_rules! call {
        ($imm8:expr) => {
            vpsraiw(a, $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_srai_epi16&expand=5426)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsraw, imm8 = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_srai_epi16(k: __mmask32, a: __m512i, imm8: u32) -> __m512i {
    let a = a.as_i16x32();
    macro_rules! call {
        ($imm8:expr) => {
            vpsraiw(a, $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_srai_epi16&expand=5422)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsraw, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm256_mask_srai_epi16(src: __m256i, k: __mmask16, a: __m256i, imm8: u32) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm256_srai_epi16::<$imm8>(a)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shf.as_i16x16(), src.as_i16x16()))
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_srai_epi16&expand=5423)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsraw, imm8 = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm256_maskz_srai_epi16(k: __mmask16, a: __m256i, imm8: u32) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm256_srai_epi16::<$imm8>(a)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, shf.as_i16x16(), zero))
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_srai_epi16&expand=5419)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsraw, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm_mask_srai_epi16(src: __m128i, k: __mmask8, a: __m128i, imm8: u32) -> __m128i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm_srai_epi16::<$imm8>(a)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shf.as_i16x8(), src.as_i16x8()))
}

/// Shift packed 16-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_srai_epi16&expand=5420)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsraw, imm8 = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_maskz_srai_epi16(k: __mmask8, a: __m128i, imm8: u32) -> __m128i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm_srai_epi16::<$imm8>(a)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, shf.as_i16x8(), zero))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_srav_epi16&expand=5456)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub unsafe fn _mm512_srav_epi16(a: __m512i, count: __m512i) -> __m512i {
    transmute(vpsravw(a.as_i16x32(), count.as_i16x32()))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_srav_epi16&expand=5454)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub unsafe fn _mm512_mask_srav_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    count: __m512i,
) -> __m512i {
    let shf = _mm512_srav_epi16(a, count).as_i16x32();
    transmute(simd_select_bitmask(k, shf, src.as_i16x32()))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_srav_epi16&expand=5455)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub unsafe fn _mm512_maskz_srav_epi16(k: __mmask32, a: __m512i, count: __m512i) -> __m512i {
    let shf = _mm512_srav_epi16(a, count).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_srav_epi16&expand=5453)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub unsafe fn _mm256_srav_epi16(a: __m256i, count: __m256i) -> __m256i {
    transmute(vpsravw256(a.as_i16x16(), count.as_i16x16()))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_srav_epi16&expand=5451)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub unsafe fn _mm256_mask_srav_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    count: __m256i,
) -> __m256i {
    let shf = _mm256_srav_epi16(a, count).as_i16x16();
    transmute(simd_select_bitmask(k, shf, src.as_i16x16()))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_srav_epi16&expand=5452)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub unsafe fn _mm256_maskz_srav_epi16(k: __mmask16, a: __m256i, count: __m256i) -> __m256i {
    let shf = _mm256_srav_epi16(a, count).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_srav_epi16&expand=5450)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub unsafe fn _mm_srav_epi16(a: __m128i, count: __m128i) -> __m128i {
    transmute(vpsravw128(a.as_i16x8(), count.as_i16x8()))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_srav_epi16&expand=5448)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub unsafe fn _mm_mask_srav_epi16(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    count: __m128i,
) -> __m128i {
    let shf = _mm_srav_epi16(a, count).as_i16x8();
    transmute(simd_select_bitmask(k, shf, src.as_i16x8()))
}

/// Shift packed 16-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_srav_epi16&expand=5449)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpsravw))]
pub unsafe fn _mm_maskz_srav_epi16(k: __mmask8, a: __m128i, count: __m128i) -> __m128i {
    let shf = _mm_srav_epi16(a, count).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_permutex2var_epi16&expand=4226)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vperm))] //vpermi2w or vpermt2w
pub unsafe fn _mm512_permutex2var_epi16(a: __m512i, idx: __m512i, b: __m512i) -> __m512i {
    transmute(vpermi2w(a.as_i16x32(), idx.as_i16x32(), b.as_i16x32()))
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_permutex2var_epi16&expand=4223)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpermt2w))]
pub unsafe fn _mm512_mask_permutex2var_epi16(
    a: __m512i,
    k: __mmask32,
    idx: __m512i,
    b: __m512i,
) -> __m512i {
    let permute = _mm512_permutex2var_epi16(a, idx, b).as_i16x32();
    transmute(simd_select_bitmask(k, permute, a.as_i16x32()))
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_permutex2var_epi16&expand=4225)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vperm))] //vpermi2w or vpermt2w
pub unsafe fn _mm512_maskz_permutex2var_epi16(
    k: __mmask32,
    a: __m512i,
    idx: __m512i,
    b: __m512i,
) -> __m512i {
    let permute = _mm512_permutex2var_epi16(a, idx, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, permute, zero))
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from idx when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask2_permutex2var_epi16&expand=4224)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpermi2w))]
pub unsafe fn _mm512_mask2_permutex2var_epi16(
    a: __m512i,
    idx: __m512i,
    k: __mmask32,
    b: __m512i,
) -> __m512i {
    let permute = _mm512_permutex2var_epi16(a, idx, b).as_i16x32();
    transmute(simd_select_bitmask(k, permute, idx.as_i16x32()))
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permutex2var_epi16&expand=4222)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vperm))] //vpermi2w or vpermt2w
pub unsafe fn _mm256_permutex2var_epi16(a: __m256i, idx: __m256i, b: __m256i) -> __m256i {
    transmute(vpermi2w256(a.as_i16x16(), idx.as_i16x16(), b.as_i16x16()))
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_permutex2var_epi16&expand=4219)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpermt2w))]
pub unsafe fn _mm256_mask_permutex2var_epi16(
    a: __m256i,
    k: __mmask16,
    idx: __m256i,
    b: __m256i,
) -> __m256i {
    let permute = _mm256_permutex2var_epi16(a, idx, b).as_i16x16();
    transmute(simd_select_bitmask(k, permute, a.as_i16x16()))
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_permutex2var_epi16&expand=4221)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vperm))] //vpermi2w or vpermt2w
pub unsafe fn _mm256_maskz_permutex2var_epi16(
    k: __mmask16,
    a: __m256i,
    idx: __m256i,
    b: __m256i,
) -> __m256i {
    let permute = _mm256_permutex2var_epi16(a, idx, b).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, permute, zero))
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from idx when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask2_permutex2var_epi16&expand=4220)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpermi2w))]
pub unsafe fn _mm256_mask2_permutex2var_epi16(
    a: __m256i,
    idx: __m256i,
    k: __mmask16,
    b: __m256i,
) -> __m256i {
    let permute = _mm256_permutex2var_epi16(a, idx, b).as_i16x16();
    transmute(simd_select_bitmask(k, permute, idx.as_i16x16()))
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_permutex2var_epi16&expand=4218)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vperm))] //vpermi2w or vpermt2w
pub unsafe fn _mm_permutex2var_epi16(a: __m128i, idx: __m128i, b: __m128i) -> __m128i {
    transmute(vpermi2w128(a.as_i16x8(), idx.as_i16x8(), b.as_i16x8()))
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_permutex2var_epi16&expand=4215)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpermt2w))]
pub unsafe fn _mm_mask_permutex2var_epi16(
    a: __m128i,
    k: __mmask8,
    idx: __m128i,
    b: __m128i,
) -> __m128i {
    let permute = _mm_permutex2var_epi16(a, idx, b).as_i16x8();
    transmute(simd_select_bitmask(k, permute, a.as_i16x8()))
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_permutex2var_epi16&expand=4217)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vperm))] //vpermi2w or vpermt2w
pub unsafe fn _mm_maskz_permutex2var_epi16(
    k: __mmask8,
    a: __m128i,
    idx: __m128i,
    b: __m128i,
) -> __m128i {
    let permute = _mm_permutex2var_epi16(a, idx, b).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, permute, zero))
}

/// Shuffle 16-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from idx when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask2_permutex2var_epi16&expand=4216)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpermi2w))]
pub unsafe fn _mm_mask2_permutex2var_epi16(
    a: __m128i,
    idx: __m128i,
    k: __mmask8,
    b: __m128i,
) -> __m128i {
    let permute = _mm_permutex2var_epi16(a, idx, b).as_i16x8();
    transmute(simd_select_bitmask(k, permute, idx.as_i16x8()))
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_permutexvar_epi16&expand=4295)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpermw))]
pub unsafe fn _mm512_permutexvar_epi16(idx: __m512i, a: __m512i) -> __m512i {
    transmute(vpermw(a.as_i16x32(), idx.as_i16x32()))
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_permutexvar_epi16&expand=4293)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpermw))]
pub unsafe fn _mm512_mask_permutexvar_epi16(
    src: __m512i,
    k: __mmask32,
    idx: __m512i,
    a: __m512i,
) -> __m512i {
    let permute = _mm512_permutexvar_epi16(idx, a).as_i16x32();
    transmute(simd_select_bitmask(k, permute, src.as_i16x32()))
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_permutexvar_epi16&expand=4294)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpermw))]
pub unsafe fn _mm512_maskz_permutexvar_epi16(k: __mmask32, idx: __m512i, a: __m512i) -> __m512i {
    let permute = _mm512_permutexvar_epi16(idx, a).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, permute, zero))
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permutexvar_epi16&expand=4292)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpermw))]
pub unsafe fn _mm256_permutexvar_epi16(idx: __m256i, a: __m256i) -> __m256i {
    transmute(vpermw256(a.as_i16x16(), idx.as_i16x16()))
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_permutexvar_epi16&expand=4290)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpermw))]
pub unsafe fn _mm256_mask_permutexvar_epi16(
    src: __m256i,
    k: __mmask16,
    idx: __m256i,
    a: __m256i,
) -> __m256i {
    let permute = _mm256_permutexvar_epi16(idx, a).as_i16x16();
    transmute(simd_select_bitmask(k, permute, src.as_i16x16()))
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_permutexvar_epi16&expand=4291)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpermw))]
pub unsafe fn _mm256_maskz_permutexvar_epi16(k: __mmask16, idx: __m256i, a: __m256i) -> __m256i {
    let permute = _mm256_permutexvar_epi16(idx, a).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, permute, zero))
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_permutexvar_epi16&expand=4289)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpermw))]
pub unsafe fn _mm_permutexvar_epi16(idx: __m128i, a: __m128i) -> __m128i {
    transmute(vpermw128(a.as_i16x8(), idx.as_i16x8()))
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_permutexvar_epi16&expand=4287)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpermw))]
pub unsafe fn _mm_mask_permutexvar_epi16(
    src: __m128i,
    k: __mmask8,
    idx: __m128i,
    a: __m128i,
) -> __m128i {
    let permute = _mm_permutexvar_epi16(idx, a).as_i16x8();
    transmute(simd_select_bitmask(k, permute, src.as_i16x8()))
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_permutexvar_epi16&expand=4288)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpermw))]
pub unsafe fn _mm_maskz_permutexvar_epi16(k: __mmask8, idx: __m128i, a: __m128i) -> __m128i {
    let permute = _mm_permutexvar_epi16(idx, a).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, permute, zero))
}

/// Blend packed 16-bit integers from a and b using control mask k, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_blend_epi16&expand=430)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovdqu16))] //should be vpblendmw
pub unsafe fn _mm512_mask_blend_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_select_bitmask(k, b.as_i16x32(), a.as_i16x32()))
}

/// Blend packed 16-bit integers from a and b using control mask k, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_blend_epi16&expand=429)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu16))] //should be vpblendmw
pub unsafe fn _mm256_mask_blend_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_select_bitmask(k, b.as_i16x16(), a.as_i16x16()))
}

/// Blend packed 16-bit integers from a and b using control mask k, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_blend_epi16&expand=427)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu16))] //should be vpblendmw
pub unsafe fn _mm_mask_blend_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    transmute(simd_select_bitmask(k, b.as_i16x8(), a.as_i16x8()))
}

/// Blend packed 8-bit integers from a and b using control mask k, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_blend_epi8&expand=441)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovdqu8))] //should be vpblendmb
pub unsafe fn _mm512_mask_blend_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_select_bitmask(k, b.as_i8x64(), a.as_i8x64()))
}

/// Blend packed 8-bit integers from a and b using control mask k, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_blend_epi8&expand=440)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu8))] //should be vpblendmb
pub unsafe fn _mm256_mask_blend_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_select_bitmask(k, b.as_i8x32(), a.as_i8x32()))
}

/// Blend packed 8-bit integers from a and b using control mask k, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_blend_epi8&expand=439)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu8))] //should be vpblendmb
pub unsafe fn _mm_mask_blend_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    transmute(simd_select_bitmask(k, b.as_i8x16(), a.as_i8x16()))
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_broadcastw_epi16&expand=587)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm512_broadcastw_epi16(a: __m128i) -> __m512i {
    let a = _mm512_castsi128_si512(a).as_i16x32();
    let ret: i16x32 = simd_shuffle32(
        a,
        a,
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ],
    );
    transmute(ret)
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_broadcastw_epi16&expand=588)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm512_mask_broadcastw_epi16(src: __m512i, k: __mmask32, a: __m128i) -> __m512i {
    let broadcast = _mm512_broadcastw_epi16(a).as_i16x32();
    transmute(simd_select_bitmask(k, broadcast, src.as_i16x32()))
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_broadcastw_epi16&expand=589)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm512_maskz_broadcastw_epi16(k: __mmask32, a: __m128i) -> __m512i {
    let broadcast = _mm512_broadcastw_epi16(a).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, broadcast, zero))
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_broadcastw_epi16&expand=585)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm256_mask_broadcastw_epi16(src: __m256i, k: __mmask16, a: __m128i) -> __m256i {
    let broadcast = _mm256_broadcastw_epi16(a).as_i16x16();
    transmute(simd_select_bitmask(k, broadcast, src.as_i16x16()))
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_broadcastw_epi16&expand=586)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm256_maskz_broadcastw_epi16(k: __mmask16, a: __m128i) -> __m256i {
    let broadcast = _mm256_broadcastw_epi16(a).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, broadcast, zero))
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_broadcastw_epi16&expand=582)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm_mask_broadcastw_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    let broadcast = _mm_broadcastw_epi16(a).as_i16x8();
    transmute(simd_select_bitmask(k, broadcast, src.as_i16x8()))
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_broadcastw_epi16&expand=583)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm_maskz_broadcastw_epi16(k: __mmask8, a: __m128i) -> __m128i {
    let broadcast = _mm_broadcastw_epi16(a).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, broadcast, zero))
}

/// Broadcast the low packed 8-bit integer from a to all elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_broadcastb_epi8&expand=536)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm512_broadcastb_epi8(a: __m128i) -> __m512i {
    let a = _mm512_castsi128_si512(a).as_i8x64();
    let ret: i8x64 = simd_shuffle64(
        a,
        a,
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ],
    );
    transmute(ret)
}

/// Broadcast the low packed 8-bit integer from a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_broadcastb_epi8&expand=537)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm512_mask_broadcastb_epi8(src: __m512i, k: __mmask64, a: __m128i) -> __m512i {
    let broadcast = _mm512_broadcastb_epi8(a).as_i8x64();
    transmute(simd_select_bitmask(k, broadcast, src.as_i8x64()))
}

/// Broadcast the low packed 8-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_broadcastb_epi8&expand=538)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm512_maskz_broadcastb_epi8(k: __mmask64, a: __m128i) -> __m512i {
    let broadcast = _mm512_broadcastb_epi8(a).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, broadcast, zero))
}

/// Broadcast the low packed 8-bit integer from a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_broadcastb_epi8&expand=534)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm256_mask_broadcastb_epi8(src: __m256i, k: __mmask32, a: __m128i) -> __m256i {
    let broadcast = _mm256_broadcastb_epi8(a).as_i8x32();
    transmute(simd_select_bitmask(k, broadcast, src.as_i8x32()))
}

/// Broadcast the low packed 8-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_broadcastb_epi8&expand=535)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm256_maskz_broadcastb_epi8(k: __mmask32, a: __m128i) -> __m256i {
    let broadcast = _mm256_broadcastb_epi8(a).as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, broadcast, zero))
}

/// Broadcast the low packed 8-bit integer from a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_broadcastb_epi8&expand=531)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm_mask_broadcastb_epi8(src: __m128i, k: __mmask16, a: __m128i) -> __m128i {
    let broadcast = _mm_broadcastb_epi8(a).as_i8x16();
    transmute(simd_select_bitmask(k, broadcast, src.as_i8x16()))
}

/// Broadcast the low packed 8-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_broadcastb_epi8&expand=532)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm_maskz_broadcastb_epi8(k: __mmask16, a: __m128i) -> __m128i {
    let broadcast = _mm_broadcastb_epi8(a).as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, broadcast, zero))
}

/// Unpack and interleave 16-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_unpackhi_epi16&expand=6012)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub unsafe fn _mm512_unpackhi_epi16(a: __m512i, b: __m512i) -> __m512i {
    let a = a.as_i16x32();
    let b = b.as_i16x32();
    #[rustfmt::skip]
    let r: i16x32 = simd_shuffle32(
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

/// Unpack and interleave 16-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_unpackhi_epi16&expand=6010)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub unsafe fn _mm512_mask_unpackhi_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let unpackhi = _mm512_unpackhi_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, unpackhi, src.as_i16x32()))
}

/// Unpack and interleave 16-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_unpackhi_epi16&expand=6011)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub unsafe fn _mm512_maskz_unpackhi_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let unpackhi = _mm512_unpackhi_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, unpackhi, zero))
}

/// Unpack and interleave 16-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_unpackhi_epi16&expand=6007)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub unsafe fn _mm256_mask_unpackhi_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    let unpackhi = _mm256_unpackhi_epi16(a, b).as_i16x16();
    transmute(simd_select_bitmask(k, unpackhi, src.as_i16x16()))
}

/// Unpack and interleave 16-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_unpackhi_epi16&expand=6008)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub unsafe fn _mm256_maskz_unpackhi_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let unpackhi = _mm256_unpackhi_epi16(a, b).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, unpackhi, zero))
}

/// Unpack and interleave 16-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_unpackhi_epi16&expand=6004)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub unsafe fn _mm_mask_unpackhi_epi16(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    b: __m128i,
) -> __m128i {
    let unpackhi = _mm_unpackhi_epi16(a, b).as_i16x8();
    transmute(simd_select_bitmask(k, unpackhi, src.as_i16x8()))
}

/// Unpack and interleave 16-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_unpackhi_epi16&expand=6005)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub unsafe fn _mm_maskz_unpackhi_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let unpackhi = _mm_unpackhi_epi16(a, b).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, unpackhi, zero))
}

/// Unpack and interleave 8-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_unpackhi_epi8&expand=6039)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub unsafe fn _mm512_unpackhi_epi8(a: __m512i, b: __m512i) -> __m512i {
    let a = a.as_i8x64();
    let b = b.as_i8x64();
    #[rustfmt::skip]
    let r: i8x64 = simd_shuffle64(
        a,
        b,
        [
            8,  64+8,   9, 64+9,
            10, 64+10, 11, 64+11,
            12, 64+12, 13, 64+13,
            14, 64+14, 15, 64+15,
            24, 64+24, 25, 64+25,
            26, 64+26, 27, 64+27,
            28, 64+28, 29, 64+29,
            30, 64+30, 31, 64+31,
            40, 64+40, 41, 64+41,
            42, 64+42, 43, 64+43,
            44, 64+44, 45, 64+45,
            46, 64+46, 47, 64+47,
            56, 64+56, 57, 64+57,
            58, 64+58, 59, 64+59,
            60, 64+60, 61, 64+61,
            62, 64+62, 63, 64+63,
        ],
    );
    transmute(r)
}

/// Unpack and interleave 8-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_unpackhi_epi8&expand=6037)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub unsafe fn _mm512_mask_unpackhi_epi8(
    src: __m512i,
    k: __mmask64,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let unpackhi = _mm512_unpackhi_epi8(a, b).as_i8x64();
    transmute(simd_select_bitmask(k, unpackhi, src.as_i8x64()))
}

/// Unpack and interleave 8-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_unpackhi_epi8&expand=6038)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub unsafe fn _mm512_maskz_unpackhi_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let unpackhi = _mm512_unpackhi_epi8(a, b).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, unpackhi, zero))
}

/// Unpack and interleave 8-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_unpackhi_epi8&expand=6034)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub unsafe fn _mm256_mask_unpackhi_epi8(
    src: __m256i,
    k: __mmask32,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    let unpackhi = _mm256_unpackhi_epi8(a, b).as_i8x32();
    transmute(simd_select_bitmask(k, unpackhi, src.as_i8x32()))
}

/// Unpack and interleave 8-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_unpackhi_epi8&expand=6035)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub unsafe fn _mm256_maskz_unpackhi_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let unpackhi = _mm256_unpackhi_epi8(a, b).as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, unpackhi, zero))
}

/// Unpack and interleave 8-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_unpackhi_epi8&expand=6031)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub unsafe fn _mm_mask_unpackhi_epi8(
    src: __m128i,
    k: __mmask16,
    a: __m128i,
    b: __m128i,
) -> __m128i {
    let unpackhi = _mm_unpackhi_epi8(a, b).as_i8x16();
    transmute(simd_select_bitmask(k, unpackhi, src.as_i8x16()))
}

/// Unpack and interleave 8-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_unpackhi_epi8&expand=6032)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub unsafe fn _mm_maskz_unpackhi_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let unpackhi = _mm_unpackhi_epi8(a, b).as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, unpackhi, zero))
}

/// Unpack and interleave 16-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_unpacklo_epi16&expand=6069)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub unsafe fn _mm512_unpacklo_epi16(a: __m512i, b: __m512i) -> __m512i {
    let a = a.as_i16x32();
    let b = b.as_i16x32();
    #[rustfmt::skip]
    let r: i16x32 = simd_shuffle32(
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

/// Unpack and interleave 16-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_unpacklo_epi16&expand=6067)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub unsafe fn _mm512_mask_unpacklo_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let unpacklo = _mm512_unpacklo_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, unpacklo, src.as_i16x32()))
}

/// Unpack and interleave 16-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_unpacklo_epi16&expand=6068)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub unsafe fn _mm512_maskz_unpacklo_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let unpacklo = _mm512_unpacklo_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, unpacklo, zero))
}

/// Unpack and interleave 16-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_unpacklo_epi16&expand=6064)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub unsafe fn _mm256_mask_unpacklo_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    let unpacklo = _mm256_unpacklo_epi16(a, b).as_i16x16();
    transmute(simd_select_bitmask(k, unpacklo, src.as_i16x16()))
}

/// Unpack and interleave 16-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_unpacklo_epi16&expand=6065)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub unsafe fn _mm256_maskz_unpacklo_epi16(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    let unpacklo = _mm256_unpacklo_epi16(a, b).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, unpacklo, zero))
}

/// Unpack and interleave 16-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_unpacklo_epi16&expand=6061)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub unsafe fn _mm_mask_unpacklo_epi16(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    b: __m128i,
) -> __m128i {
    let unpacklo = _mm_unpacklo_epi16(a, b).as_i16x8();
    transmute(simd_select_bitmask(k, unpacklo, src.as_i16x8()))
}

/// Unpack and interleave 16-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_unpacklo_epi16&expand=6062)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub unsafe fn _mm_maskz_unpacklo_epi16(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    let unpacklo = _mm_unpacklo_epi16(a, b).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, unpacklo, zero))
}

/// Unpack and interleave 8-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_unpacklo_epi8&expand=6096)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub unsafe fn _mm512_unpacklo_epi8(a: __m512i, b: __m512i) -> __m512i {
    let a = a.as_i8x64();
    let b = b.as_i8x64();
    #[rustfmt::skip]
    let r: i8x64 = simd_shuffle64(
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

/// Unpack and interleave 8-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_unpacklo_epi8&expand=6094)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub unsafe fn _mm512_mask_unpacklo_epi8(
    src: __m512i,
    k: __mmask64,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let unpacklo = _mm512_unpacklo_epi8(a, b).as_i8x64();
    transmute(simd_select_bitmask(k, unpacklo, src.as_i8x64()))
}

/// Unpack and interleave 8-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_unpacklo_epi8&expand=6095)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub unsafe fn _mm512_maskz_unpacklo_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let unpacklo = _mm512_unpacklo_epi8(a, b).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, unpacklo, zero))
}

/// Unpack and interleave 8-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_unpacklo_epi8&expand=6091)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub unsafe fn _mm256_mask_unpacklo_epi8(
    src: __m256i,
    k: __mmask32,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    let unpacklo = _mm256_unpacklo_epi8(a, b).as_i8x32();
    transmute(simd_select_bitmask(k, unpacklo, src.as_i8x32()))
}

/// Unpack and interleave 8-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_unpacklo_epi8&expand=6092)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub unsafe fn _mm256_maskz_unpacklo_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let unpacklo = _mm256_unpacklo_epi8(a, b).as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, unpacklo, zero))
}

/// Unpack and interleave 8-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_unpacklo_epi8&expand=6088)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub unsafe fn _mm_mask_unpacklo_epi8(
    src: __m128i,
    k: __mmask16,
    a: __m128i,
    b: __m128i,
) -> __m128i {
    let unpacklo = _mm_unpacklo_epi8(a, b).as_i8x16();
    transmute(simd_select_bitmask(k, unpacklo, src.as_i8x16()))
}

/// Unpack and interleave 8-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_unpacklo_epi8&expand=6089)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub unsafe fn _mm_maskz_unpacklo_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let unpacklo = _mm_unpacklo_epi8(a, b).as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, unpacklo, zero))
}

/// Move packed 16-bit integers from a into dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_mov_epi16&expand=3795)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
pub unsafe fn _mm512_mask_mov_epi16(src: __m512i, k: __mmask32, a: __m512i) -> __m512i {
    let mov = a.as_i16x32();
    transmute(simd_select_bitmask(k, mov, src.as_i16x32()))
}

/// Move packed 16-bit integers from a into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_mov_epi16&expand=3796)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
pub unsafe fn _mm512_maskz_mov_epi16(k: __mmask32, a: __m512i) -> __m512i {
    let mov = a.as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, mov, zero))
}

/// Move packed 16-bit integers from a into dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_mov_epi16&expand=3793)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
pub unsafe fn _mm256_mask_mov_epi16(src: __m256i, k: __mmask16, a: __m256i) -> __m256i {
    let mov = a.as_i16x16();
    transmute(simd_select_bitmask(k, mov, src.as_i16x16()))
}

/// Move packed 16-bit integers from a into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_mov_epi16&expand=3794)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
pub unsafe fn _mm256_maskz_mov_epi16(k: __mmask16, a: __m256i) -> __m256i {
    let mov = a.as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, mov, zero))
}

/// Move packed 16-bit integers from a into dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_mov_epi16&expand=3791)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
pub unsafe fn _mm_mask_mov_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    let mov = a.as_i16x8();
    transmute(simd_select_bitmask(k, mov, src.as_i16x8()))
}

/// Move packed 16-bit integers from a into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_mov_epi16&expand=3792)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu16))]
pub unsafe fn _mm_maskz_mov_epi16(k: __mmask8, a: __m128i) -> __m128i {
    let mov = a.as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, mov, zero))
}

/// Move packed 8-bit integers from a into dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_mov_epi8&expand=3813)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
pub unsafe fn _mm512_mask_mov_epi8(src: __m512i, k: __mmask64, a: __m512i) -> __m512i {
    let mov = a.as_i8x64();
    transmute(simd_select_bitmask(k, mov, src.as_i8x64()))
}

/// Move packed 8-bit integers from a into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_mov_epi8&expand=3814)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
pub unsafe fn _mm512_maskz_mov_epi8(k: __mmask64, a: __m512i) -> __m512i {
    let mov = a.as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, mov, zero))
}

/// Move packed 8-bit integers from a into dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_mov_epi8&expand=3811)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
pub unsafe fn _mm256_mask_mov_epi8(src: __m256i, k: __mmask32, a: __m256i) -> __m256i {
    let mov = a.as_i8x32();
    transmute(simd_select_bitmask(k, mov, src.as_i8x32()))
}

/// Move packed 8-bit integers from a into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_mov_epi8&expand=3812)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
pub unsafe fn _mm256_maskz_mov_epi8(k: __mmask32, a: __m256i) -> __m256i {
    let mov = a.as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, mov, zero))
}

/// Move packed 8-bit integers from a into dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_mov_epi8&expand=3809)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
pub unsafe fn _mm_mask_mov_epi8(src: __m128i, k: __mmask16, a: __m128i) -> __m128i {
    let mov = a.as_i8x16();
    transmute(simd_select_bitmask(k, mov, src.as_i8x16()))
}

/// Move packed 8-bit integers from a into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_mov_epi8&expand=3810)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vmovdqu8))]
pub unsafe fn _mm_maskz_mov_epi8(k: __mmask16, a: __m128i) -> __m128i {
    let mov = a.as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, mov, zero))
}

/// Broadcast 16-bit integer a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_set1_epi16&expand=4942)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm512_mask_set1_epi16(src: __m512i, k: __mmask32, a: i16) -> __m512i {
    let r = _mm512_set1_epi16(a).as_i16x32();
    transmute(simd_select_bitmask(k, r, src.as_i16x32()))
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_set1_epi16&expand=4943)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm512_maskz_set1_epi16(k: __mmask32, a: i16) -> __m512i {
    let r = _mm512_set1_epi16(a).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, r, zero))
}

/// Broadcast 16-bit integer a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_set1_epi16&expand=4939)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm256_mask_set1_epi16(src: __m256i, k: __mmask16, a: i16) -> __m256i {
    let r = _mm256_set1_epi16(a).as_i16x16();
    transmute(simd_select_bitmask(k, r, src.as_i16x16()))
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_set1_epi16&expand=4940)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm256_maskz_set1_epi16(k: __mmask16, a: i16) -> __m256i {
    let r = _mm256_set1_epi16(a).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, r, zero))
}

/// Broadcast 16-bit integer a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_set1_epi16&expand=4936)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm_mask_set1_epi16(src: __m128i, k: __mmask8, a: i16) -> __m128i {
    let r = _mm_set1_epi16(a).as_i16x8();
    transmute(simd_select_bitmask(k, r, src.as_i16x8()))
}

/// Broadcast the low packed 16-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_set1_epi16&expand=4937)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm_maskz_set1_epi16(k: __mmask8, a: i16) -> __m128i {
    let r = _mm_set1_epi16(a).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, r, zero))
}

/// Broadcast 8-bit integer a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_set1_epi8&expand=4970)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm512_mask_set1_epi8(src: __m512i, k: __mmask64, a: i8) -> __m512i {
    let r = _mm512_set1_epi8(a).as_i8x64();
    transmute(simd_select_bitmask(k, r, src.as_i8x64()))
}

/// Broadcast 8-bit integer a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_set1_epi8&expand=4971)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm512_maskz_set1_epi8(k: __mmask64, a: i8) -> __m512i {
    let r = _mm512_set1_epi8(a).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, r, zero))
}

/// Broadcast 8-bit integer a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_set1_epi8&expand=4967)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm256_mask_set1_epi8(src: __m256i, k: __mmask32, a: i8) -> __m256i {
    let r = _mm256_set1_epi8(a).as_i8x32();
    transmute(simd_select_bitmask(k, r, src.as_i8x32()))
}

/// Broadcast 8-bit integer a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_set1_epi8&expand=4968)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm256_maskz_set1_epi8(k: __mmask32, a: i8) -> __m256i {
    let r = _mm256_set1_epi8(a).as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, r, zero))
}

/// Broadcast 8-bit integer a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_set1_epi8&expand=4964)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm_mask_set1_epi8(src: __m128i, k: __mmask16, a: i8) -> __m128i {
    let r = _mm_set1_epi8(a).as_i8x16();
    transmute(simd_select_bitmask(k, r, src.as_i8x16()))
}

/// Broadcast 8-bit integer a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_set1_epi8&expand=4965)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm_maskz_set1_epi8(k: __mmask16, a: i8) -> __m128i {
    let r = _mm_set1_epi8(a).as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, r, zero))
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the low 64 bits of 128-bit lanes of dst, with the high 64 bits of 128-bit lanes being copied from from a to dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_shufflelo_epi16&expand=5221)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpshuflw, imm8 = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_shufflelo_epi16(a: __m512i, imm8: i32) -> __m512i {
    let imm8 = (imm8 & 0xFF) as u8;
    let a = a.as_i16x32();
    macro_rules! shuffle_done {
        ($x01: expr, $x23: expr, $x45: expr, $x67: expr) => {
            #[rustfmt::skip]
                        simd_shuffle32(a, a, [
                            0+$x01, 0+$x23, 0+$x45, 0+$x67, 4, 5, 6, 7, 8+$x01, 8+$x23, 8+$x45, 8+$x67, 12, 13, 14, 15,
                            16+$x01, 16+$x23, 16+$x45, 16+$x67, 20, 21, 22, 23, 24+$x01, 24+$x23, 24+$x45, 24+$x67, 28, 29, 30, 31,
                        ])
        };
    }
    macro_rules! shuffle_x67 {
        ($x01:expr, $x23:expr, $x45:expr) => {
            match (imm8 >> 6) & 0b11 {
                0b00 => shuffle_done!($x01, $x23, $x45, 0),
                0b01 => shuffle_done!($x01, $x23, $x45, 1),
                0b10 => shuffle_done!($x01, $x23, $x45, 2),
                _ => shuffle_done!($x01, $x23, $x45, 3),
            }
        };
    }
    macro_rules! shuffle_x45 {
        ($x01:expr, $x23:expr) => {
            match (imm8 >> 4) & 0b11 {
                0b00 => shuffle_x67!($x01, $x23, 0),
                0b01 => shuffle_x67!($x01, $x23, 1),
                0b10 => shuffle_x67!($x01, $x23, 2),
                _ => shuffle_x67!($x01, $x23, 3),
            }
        };
    }
    macro_rules! shuffle_x23 {
        ($x01:expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => shuffle_x45!($x01, 0),
                0b01 => shuffle_x45!($x01, 1),
                0b10 => shuffle_x45!($x01, 2),
                _ => shuffle_x45!($x01, 3),
            }
        };
    }
    let r: i16x32 = match imm8 & 0b11 {
        0b00 => shuffle_x23!(0),
        0b01 => shuffle_x23!(1),
        0b10 => shuffle_x23!(2),
        _ => shuffle_x23!(3),
    };
    transmute(r)
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the low 64 bits of 128-bit lanes of dst, with the high 64 bits of 128-bit lanes being copied from from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_shufflelo_epi16&expand=5219)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpshuflw, imm8 = 0))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_shufflelo_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    imm8: i32,
) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm512_shufflelo_epi16(a, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, r.as_i16x32(), src.as_i16x32()))
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the low 64 bits of 128-bit lanes of dst, with the high 64 bits of 128-bit lanes being copied from from a to dst, using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_shufflelo_epi16&expand=5220)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpshuflw, imm8 = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_shufflelo_epi16(k: __mmask32, a: __m512i, imm8: i32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm512_shufflelo_epi16(a, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, r.as_i16x32(), zero))
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the low 64 bits of 128-bit lanes of dst, with the high 64 bits of 128-bit lanes being copied from from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_shufflelo_epi16&expand=5216)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpshuflw, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm256_mask_shufflelo_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    imm8: i32,
) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm256_shufflelo_epi16(a, $imm8)
        };
    }
    let shuffle = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shuffle.as_i16x16(), src.as_i16x16()))
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the low 64 bits of 128-bit lanes of dst, with the high 64 bits of 128-bit lanes being copied from from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_shufflelo_epi16&expand=5217)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpshuflw, imm8 = 5))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm256_maskz_shufflelo_epi16(k: __mmask16, a: __m256i, imm8: i32) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm256_shufflelo_epi16(a, $imm8)
        };
    }
    let shuffle = constify_imm8_sae!(imm8, call);
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, shuffle.as_i16x16(), zero))
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the low 64 bits of 128-bit lanes of dst, with the high 64 bits of 128-bit lanes being copied from from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_shufflelo_epi16&expand=5213)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpshuflw, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm_mask_shufflelo_epi16(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    imm8: i32,
) -> __m128i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm_shufflelo_epi16::<$imm8>(a)
        };
    }
    let shuffle = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shuffle.as_i16x8(), src.as_i16x8()))
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the low 64 bits of 128-bit lanes of dst, with the high 64 bits of 128-bit lanes being copied from from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_shufflelo_epi16&expand=5214)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpshuflw, imm8 = 5))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_maskz_shufflelo_epi16(k: __mmask8, a: __m128i, imm8: i32) -> __m128i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm_shufflelo_epi16::<$imm8>(a)
        };
    }
    let shuffle = constify_imm8_sae!(imm8, call);
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, shuffle.as_i16x8(), zero))
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the high 64 bits of 128-bit lanes of dst, with the low 64 bits of 128-bit lanes being copied from from a to dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_shufflehi_epi16&expand=5212)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpshufhw, imm8 = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_shufflehi_epi16(a: __m512i, imm8: i32) -> __m512i {
    let imm8 = (imm8 & 0xFF) as u8;
    let a = a.as_i16x32();
    macro_rules! shuffle_done {
        ($x01: expr, $x23: expr, $x45: expr, $x67: expr) => {
            #[rustfmt::skip]
                        simd_shuffle32(a, a, [
                            0, 1, 2, 3, 4+$x01, 4+$x23, 4+$x45, 4+$x67, 8, 9, 10, 11, 12+$x01, 12+$x23, 12+$x45, 12+$x67,
                            16, 17, 18, 19, 20+$x01, 20+$x23, 20+$x45, 20+$x67, 24, 25, 26, 27, 28+$x01, 28+$x23, 28+$x45, 28+$x67,
                        ])
        };
    }
    macro_rules! shuffle_x67 {
        ($x01:expr, $x23:expr, $x45:expr) => {
            match (imm8 >> 6) & 0b11 {
                0b00 => shuffle_done!($x01, $x23, $x45, 0),
                0b01 => shuffle_done!($x01, $x23, $x45, 1),
                0b10 => shuffle_done!($x01, $x23, $x45, 2),
                _ => shuffle_done!($x01, $x23, $x45, 3),
            }
        };
    }
    macro_rules! shuffle_x45 {
        ($x01:expr, $x23:expr) => {
            match (imm8 >> 4) & 0b11 {
                0b00 => shuffle_x67!($x01, $x23, 0),
                0b01 => shuffle_x67!($x01, $x23, 1),
                0b10 => shuffle_x67!($x01, $x23, 2),
                _ => shuffle_x67!($x01, $x23, 3),
            }
        };
    }
    macro_rules! shuffle_x23 {
        ($x01:expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => shuffle_x45!($x01, 0),
                0b01 => shuffle_x45!($x01, 1),
                0b10 => shuffle_x45!($x01, 2),
                _ => shuffle_x45!($x01, 3),
            }
        };
    }
    let r: i16x32 = match imm8 & 0b11 {
        0b00 => shuffle_x23!(0),
        0b01 => shuffle_x23!(1),
        0b10 => shuffle_x23!(2),
        _ => shuffle_x23!(3),
    };
    transmute(r)
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the high 64 bits of 128-bit lanes of dst, with the low 64 bits of 128-bit lanes being copied from from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_shufflehi_epi16&expand=5210)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpshufhw, imm8 = 0))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_shufflehi_epi16(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    imm8: i32,
) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm512_shufflehi_epi16(a, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, r.as_i16x32(), src.as_i16x32()))
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the high 64 bits of 128-bit lanes of dst, with the low 64 bits of 128-bit lanes being copied from from a to dst, using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_shufflehi_epi16&expand=5211)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpshufhw, imm8 = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_shufflehi_epi16(k: __mmask32, a: __m512i, imm8: i32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm512_shufflehi_epi16(a, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, r.as_i16x32(), zero))
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the high 64 bits of 128-bit lanes of dst, with the low 64 bits of 128-bit lanes being copied from from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_shufflehi_epi16&expand=5207)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpshufhw, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm256_mask_shufflehi_epi16(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    imm8: i32,
) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm256_shufflehi_epi16(a, $imm8)
        };
    }
    let shuffle = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shuffle.as_i16x16(), src.as_i16x16()))
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the high 64 bits of 128-bit lanes of dst, with the low 64 bits of 128-bit lanes being copied from from a to dst, using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_shufflehi_epi16&expand=5208)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpshufhw, imm8 = 5))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm256_maskz_shufflehi_epi16(k: __mmask16, a: __m256i, imm8: i32) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm256_shufflehi_epi16(a, $imm8)
        };
    }
    let shuffle = constify_imm8_sae!(imm8, call);
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, shuffle.as_i16x16(), zero))
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the high 64 bits of 128-bit lanes of dst, with the low 64 bits of 128-bit lanes being copied from from a to dst, using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_shufflehi_epi16&expand=5204)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpshufhw, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm_mask_shufflehi_epi16(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    imm8: i32,
) -> __m128i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm_shufflehi_epi16::<$imm8>(a)
        };
    }
    let shuffle = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shuffle.as_i16x8(), src.as_i16x8()))
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of a using the control in imm8. Store the results in the high 64 bits of 128-bit lanes of dst, with the low 64 bits of 128-bit lanes being copied from from a to dst, using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_shufflehi_epi16&expand=5205)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpshufhw, imm8 = 5))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_maskz_shufflehi_epi16(k: __mmask8, a: __m128i, imm8: i32) -> __m128i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm_shufflehi_epi16::<$imm8>(a)
        };
    }
    let shuffle = constify_imm8_sae!(imm8, call);
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, shuffle.as_i16x8(), zero))
}

/// Shuffle packed 8-bit integers in a according to shuffle control mask in the corresponding 8-bit element of b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_shuffle_epi8&expand=5159)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub unsafe fn _mm512_shuffle_epi8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpshufb(a.as_i8x64(), b.as_i8x64()))
}

/// Shuffle 8-bit integers in a within 128-bit lanes using the control in the corresponding 8-bit element of b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_shuffle_epi8&expand=5157)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub unsafe fn _mm512_mask_shuffle_epi8(
    src: __m512i,
    k: __mmask64,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let shuffle = _mm512_shuffle_epi8(a, b).as_i8x64();
    transmute(simd_select_bitmask(k, shuffle, src.as_i8x64()))
}

/// Shuffle packed 8-bit integers in a according to shuffle control mask in the corresponding 8-bit element of b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_shuffle_epi8&expand=5158)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub unsafe fn _mm512_maskz_shuffle_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let shuffle = _mm512_shuffle_epi8(a, b).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, shuffle, zero))
}

/// Shuffle 8-bit integers in a within 128-bit lanes using the control in the corresponding 8-bit element of b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_shuffle_epi8&expand=5154)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub unsafe fn _mm256_mask_shuffle_epi8(
    src: __m256i,
    k: __mmask32,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    let shuffle = _mm256_shuffle_epi8(a, b).as_i8x32();
    transmute(simd_select_bitmask(k, shuffle, src.as_i8x32()))
}

/// Shuffle packed 8-bit integers in a according to shuffle control mask in the corresponding 8-bit element of b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_shuffle_epi8&expand=5155)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub unsafe fn _mm256_maskz_shuffle_epi8(k: __mmask32, a: __m256i, b: __m256i) -> __m256i {
    let shuffle = _mm256_shuffle_epi8(a, b).as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, shuffle, zero))
}

/// Shuffle 8-bit integers in a within 128-bit lanes using the control in the corresponding 8-bit element of b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_shuffle_epi8&expand=5151)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub unsafe fn _mm_mask_shuffle_epi8(src: __m128i, k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let shuffle = _mm_shuffle_epi8(a, b).as_i8x16();
    transmute(simd_select_bitmask(k, shuffle, src.as_i8x16()))
}

/// Shuffle packed 8-bit integers in a according to shuffle control mask in the corresponding 8-bit element of b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_shuffle_epi8&expand=5152)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub unsafe fn _mm_maskz_shuffle_epi8(k: __mmask16, a: __m128i, b: __m128i) -> __m128i {
    let shuffle = _mm_shuffle_epi8(a, b).as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, shuffle, zero))
}

/// Compute the bitwise AND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_test_epi16_mask&expand=5884)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vptestmw))]
pub unsafe fn _mm512_test_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_cmpneq_epi16_mask(and, zero)
}

/// Compute the bitwise AND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is non-zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_test_epi16_mask&expand=5883)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vptestmw))]
pub unsafe fn _mm512_mask_test_epi16_mask(k: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_mask_cmpneq_epi16_mask(k, and, zero)
}

/// Compute the bitwise AND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
///
// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_test_epi16_mask&expand=5882)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestmw))]
pub unsafe fn _mm256_test_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_cmpneq_epi16_mask(and, zero)
}

/// Compute the bitwise AND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is non-zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_test_epi16_mask&expand=5881)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestmw))]
pub unsafe fn _mm256_mask_test_epi16_mask(k: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_mask_cmpneq_epi16_mask(k, and, zero)
}

/// Compute the bitwise AND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
///
// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_test_epi16_mask&expand=5880)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestmw))]
pub unsafe fn _mm_test_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_cmpneq_epi16_mask(and, zero)
}

/// Compute the bitwise AND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is non-zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_test_epi16_mask&expand=5879)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestmw))]
pub unsafe fn _mm_mask_test_epi16_mask(k: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_mask_cmpneq_epi16_mask(k, and, zero)
}

/// Compute the bitwise AND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_test_epi8_mask&expand=5902)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vptestmb))]
pub unsafe fn _mm512_test_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_cmpneq_epi8_mask(and, zero)
}

/// Compute the bitwise AND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is non-zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_test_epi8_mask&expand=5901)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vptestmb))]
pub unsafe fn _mm512_mask_test_epi8_mask(k: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_mask_cmpneq_epi8_mask(k, and, zero)
}

/// Compute the bitwise AND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_test_epi8_mask&expand=5900)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestmb))]
pub unsafe fn _mm256_test_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_cmpneq_epi8_mask(and, zero)
}

/// Compute the bitwise AND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is non-zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_test_epi8_mask&expand=5899)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestmb))]
pub unsafe fn _mm256_mask_test_epi8_mask(k: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_mask_cmpneq_epi8_mask(k, and, zero)
}

/// Compute the bitwise AND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_test_epi8_mask&expand=5898)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestmb))]
pub unsafe fn _mm_test_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_cmpneq_epi8_mask(and, zero)
}

/// Compute the bitwise AND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is non-zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_test_epi8_mask&expand=5897)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestmb))]
pub unsafe fn _mm_mask_test_epi8_mask(k: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_mask_cmpneq_epi8_mask(k, and, zero)
}

/// Compute the bitwise NAND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_testn_epi16_mask&expand=5915)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vptestnmw))]
pub unsafe fn _mm512_testn_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_cmpeq_epi16_mask(and, zero)
}

/// Compute the bitwise NAND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_testn_epi16&expand=5914)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vptestnmw))]
pub unsafe fn _mm512_mask_testn_epi16_mask(k: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_mask_cmpeq_epi16_mask(k, and, zero)
}

/// Compute the bitwise NAND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_testn_epi16_mask&expand=5913)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestnmw))]
pub unsafe fn _mm256_testn_epi16_mask(a: __m256i, b: __m256i) -> __mmask16 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_cmpeq_epi16_mask(and, zero)
}

/// Compute the bitwise NAND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_testn_epi16_mask&expand=5912)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestnmw))]
pub unsafe fn _mm256_mask_testn_epi16_mask(k: __mmask16, a: __m256i, b: __m256i) -> __mmask16 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_mask_cmpeq_epi16_mask(k, and, zero)
}

/// Compute the bitwise NAND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_testn_epi16_mask&expand=5911)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestnmw))]
pub unsafe fn _mm_testn_epi16_mask(a: __m128i, b: __m128i) -> __mmask8 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_cmpeq_epi16_mask(and, zero)
}

/// Compute the bitwise NAND of packed 16-bit integers in a and b, producing intermediate 16-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_testn_epi16_mask&expand=5910)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestnmw))]
pub unsafe fn _mm_mask_testn_epi16_mask(k: __mmask8, a: __m128i, b: __m128i) -> __mmask8 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_mask_cmpeq_epi16_mask(k, and, zero)
}

/// Compute the bitwise NAND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_testn_epi8_mask&expand=5933)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vptestnmb))]
pub unsafe fn _mm512_testn_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_cmpeq_epi8_mask(and, zero)
}

/// Compute the bitwise NAND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_testn_epi8_mask&expand=5932)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vptestnmb))]
pub unsafe fn _mm512_mask_testn_epi8_mask(k: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    let and = _mm512_and_si512(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_mask_cmpeq_epi8_mask(k, and, zero)
}

/// Compute the bitwise NAND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_testn_epi8_mask&expand=5931)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestnmb))]
pub unsafe fn _mm256_testn_epi8_mask(a: __m256i, b: __m256i) -> __mmask32 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_cmpeq_epi8_mask(and, zero)
}

/// Compute the bitwise NAND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_testn_epi8_mask&expand=5930)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestnmb))]
pub unsafe fn _mm256_mask_testn_epi8_mask(k: __mmask32, a: __m256i, b: __m256i) -> __mmask32 {
    let and = _mm256_and_si256(a, b);
    let zero = _mm256_setzero_si256();
    _mm256_mask_cmpeq_epi8_mask(k, and, zero)
}

/// Compute the bitwise NAND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_testn_epi8_mask&expand=5929)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestnmb))]
pub unsafe fn _mm_testn_epi8_mask(a: __m128i, b: __m128i) -> __mmask16 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_cmpeq_epi8_mask(and, zero)
}

/// Compute the bitwise NAND of packed 8-bit integers in a and b, producing intermediate 8-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_testn_epi8_mask&expand=5928)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vptestnmb))]
pub unsafe fn _mm_mask_testn_epi8_mask(k: __mmask16, a: __m128i, b: __m128i) -> __mmask16 {
    let and = _mm_and_si128(a, b);
    let zero = _mm_setzero_si128();
    _mm_mask_cmpeq_epi8_mask(k, and, zero)
}

/// Store 64-bit mask from a into memory.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_store_mask64&expand=5578)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(mov))] //should be kmovq
pub unsafe fn _store_mask64(mem_addr: *mut u64, a: __mmask64) {
    ptr::write(mem_addr as *mut __mmask64, a);
}

/// Store 32-bit mask from a into memory.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_store_mask32&expand=5577)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(mov))] //should be kmovd
pub unsafe fn _store_mask32(mem_addr: *mut u32, a: __mmask32) {
    ptr::write(mem_addr as *mut __mmask32, a);
}

/// Load 64-bit mask from memory into k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_load_mask64&expand=3318)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(mov))] //should be kmovq
pub unsafe fn _load_mask64(mem_addr: *const u64) -> __mmask64 {
    ptr::read(mem_addr as *const __mmask64)
}

/// Load 32-bit mask from memory into k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_load_mask32&expand=3317)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(mov))] //should be kmovd
pub unsafe fn _load_mask32(mem_addr: *const u32) -> __mmask32 {
    ptr::read(mem_addr as *const __mmask32)
}

/// Compute the absolute differences of packed unsigned 8-bit integers in a and b, then horizontally sum each consecutive 8 differences to produce eight unsigned 16-bit integers, and pack these unsigned 16-bit integers in the low 16 bits of 64-bit elements in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_sad_epu8&expand=4855)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsadbw))]
pub unsafe fn _mm512_sad_epu8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsadbw(a.as_u8x64(), b.as_u8x64()))
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst. Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_dbsad_epu8&expand=2114)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vdbpsadbw, imm8 = 0))]
pub unsafe fn _mm512_dbsad_epu8(a: __m512i, b: __m512i, imm8: i32) -> __m512i {
    let a = a.as_u8x64();
    let b = b.as_u8x64();
    macro_rules! call {
        ($imm8:expr) => {
            vdbpsadbw(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_dbsad_epu8&expand=2115)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(4)]
#[cfg_attr(test, assert_instr(vdbpsadbw, imm8 = 0))]
pub unsafe fn _mm512_mask_dbsad_epu8(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
    imm8: i32,
) -> __m512i {
    let a = a.as_u8x64();
    let b = b.as_u8x64();
    macro_rules! call {
        ($imm8:expr) => {
            vdbpsadbw(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, r, src.as_u16x32()))
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_dbsad_epu8&expand=2116)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vdbpsadbw, imm8 = 0))]
pub unsafe fn _mm512_maskz_dbsad_epu8(k: __mmask32, a: __m512i, b: __m512i, imm8: i32) -> __m512i {
    let a = a.as_u8x64();
    let b = b.as_u8x64();
    macro_rules! call {
        ($imm8:expr) => {
            vdbpsadbw(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(
        k,
        r,
        _mm512_setzero_si512().as_u16x32(),
    ))
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst. Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_dbsad_epu8&expand=2111)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vdbpsadbw, imm8 = 0))]
pub unsafe fn _mm256_dbsad_epu8(a: __m256i, b: __m256i, imm8: i32) -> __m256i {
    let a = a.as_u8x32();
    let b = b.as_u8x32();
    macro_rules! call {
        ($imm8:expr) => {
            vdbpsadbw256(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_dbsad_epu8&expand=2112)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(4)]
#[cfg_attr(test, assert_instr(vdbpsadbw, imm8 = 0))]
pub unsafe fn _mm256_mask_dbsad_epu8(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
    imm8: i32,
) -> __m256i {
    let a = a.as_u8x32();
    let b = b.as_u8x32();
    macro_rules! call {
        ($imm8:expr) => {
            vdbpsadbw256(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, r, src.as_u16x16()))
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_dbsad_epu8&expand=2113)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vdbpsadbw, imm8 = 0))]
pub unsafe fn _mm256_maskz_dbsad_epu8(k: __mmask16, a: __m256i, b: __m256i, imm8: i32) -> __m256i {
    let a = a.as_u8x32();
    let b = b.as_u8x32();
    macro_rules! call {
        ($imm8:expr) => {
            vdbpsadbw256(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(
        k,
        r,
        _mm256_setzero_si256().as_u16x16(),
    ))
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst. Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_dbsad_epu8&expand=2108)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vdbpsadbw, imm8 = 0))]
pub unsafe fn _mm_dbsad_epu8(a: __m128i, b: __m128i, imm8: i32) -> __m128i {
    let a = a.as_u8x16();
    let b = b.as_u8x16();
    macro_rules! call {
        ($imm8:expr) => {
            vdbpsadbw128(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_dbsad_epu8&expand=2109)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(4)]
#[cfg_attr(test, assert_instr(vdbpsadbw, imm8 = 0))]
pub unsafe fn _mm_mask_dbsad_epu8(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    b: __m128i,
    imm8: i32,
) -> __m128i {
    let a = a.as_u8x16();
    let b = b.as_u8x16();
    macro_rules! call {
        ($imm8:expr) => {
            vdbpsadbw128(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, r, src.as_u16x8()))
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in a compared to those in b, and store the 16-bit results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). Four SADs are performed on four 8-bit quadruplets for each 64-bit lane. The first two SADs use the lower 8-bit quadruplet of the lane from a, and the last two SADs use the uppper 8-bit quadruplet of the lane from a. Quadruplets from b are selected from within 128-bit lanes according to the control in imm8, and each SAD in each 64-bit lane uses the selected quadruplet at 8-bit offsets.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_dbsad_epu8&expand=2110)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vdbpsadbw, imm8 = 0))]
pub unsafe fn _mm_maskz_dbsad_epu8(k: __mmask8, a: __m128i, b: __m128i, imm8: i32) -> __m128i {
    let a = a.as_u8x16();
    let b = b.as_u8x16();
    macro_rules! call {
        ($imm8:expr) => {
            vdbpsadbw128(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, r, _mm_setzero_si128().as_u16x8()))
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 16-bit integer in a.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_movepi16_mask&expand=3873)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(mov))] // should be vpmovw2m but msvc does not generate it
pub unsafe fn _mm512_movepi16_mask(a: __m512i) -> __mmask32 {
    let filter = _mm512_set1_epi16(1 << 15);
    let a = _mm512_and_si512(a, filter);
    _mm512_cmpeq_epi16_mask(a, filter)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 16-bit integer in a.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_movepi16_mask&expand=3872)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(mov))] // should be vpmovw2m but msvc does not generate it
pub unsafe fn _mm256_movepi16_mask(a: __m256i) -> __mmask16 {
    let filter = _mm256_set1_epi16(1 << 15);
    let a = _mm256_and_si256(a, filter);
    _mm256_cmpeq_epi16_mask(a, filter)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 16-bit integer in a.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_movepi16_mask&expand=3871)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(mov))] // should be vpmovw2m but msvc does not generate it
pub unsafe fn _mm_movepi16_mask(a: __m128i) -> __mmask8 {
    let filter = _mm_set1_epi16(1 << 15);
    let a = _mm_and_si128(a, filter);
    _mm_cmpeq_epi16_mask(a, filter)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 8-bit integer in a.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_movepi8_mask&expand=3883)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(mov))] // should be vpmovb2m but msvc does not generate it
pub unsafe fn _mm512_movepi8_mask(a: __m512i) -> __mmask64 {
    let filter = _mm512_set1_epi8(1 << 7);
    let a = _mm512_and_si512(a, filter);
    _mm512_cmpeq_epi8_mask(a, filter)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 8-bit integer in a.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_movepi8_mask&expand=3882)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(mov))] // should be vpmovb2m but msvc does not generate it
pub unsafe fn _mm256_movepi8_mask(a: __m256i) -> __mmask32 {
    let filter = _mm256_set1_epi8(1 << 7);
    let a = _mm256_and_si256(a, filter);
    _mm256_cmpeq_epi8_mask(a, filter)
}

/// Set each bit of mask register k based on the most significant bit of the corresponding packed 8-bit integer in a.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_movepi8_mask&expand=3881)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(mov))] // should be vpmovb2m but msvc does not generate it
pub unsafe fn _mm_movepi8_mask(a: __m128i) -> __mmask16 {
    let filter = _mm_set1_epi8(1 << 7);
    let a = _mm_and_si128(a, filter);
    _mm_cmpeq_epi8_mask(a, filter)
}

/// Set each packed 16-bit integer in dst to all ones or all zeros based on the value of the corresponding bit in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_movm_epi16&expand=3886)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovm2w))]
pub unsafe fn _mm512_movm_epi16(k: __mmask32) -> __m512i {
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
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, one, zero))
}

/// Set each packed 16-bit integer in dst to all ones or all zeros based on the value of the corresponding bit in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_movm_epi16&expand=3885)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovm2w))]
pub unsafe fn _mm256_movm_epi16(k: __mmask16) -> __m256i {
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
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, one, zero))
}

/// Set each packed 16-bit integer in dst to all ones or all zeros based on the value of the corresponding bit in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_movm_epi16&expand=3884)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovm2w))]
pub unsafe fn _mm_movm_epi16(k: __mmask8) -> __m128i {
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
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, one, zero))
}

/// Set each packed 8-bit integer in dst to all ones or all zeros based on the value of the corresponding bit in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_movm_epi8&expand=3895)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovm2b))]
pub unsafe fn _mm512_movm_epi8(k: __mmask64) -> __m512i {
    let one =
        _mm512_set1_epi8(1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0)
            .as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, one, zero))
}

/// Set each packed 8-bit integer in dst to all ones or all zeros based on the value of the corresponding bit in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_movm_epi8&expand=3894)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovm2b))]
pub unsafe fn _mm256_movm_epi8(k: __mmask32) -> __m256i {
    let one =
        _mm256_set1_epi8(1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0)
            .as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, one, zero))
}

/// Set each packed 8-bit integer in dst to all ones or all zeros based on the value of the corresponding bit in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_movm_epi8&expand=3893)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovm2b))]
pub unsafe fn _mm_movm_epi8(k: __mmask16) -> __m128i {
    let one = _mm_set1_epi8(1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0)
        .as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, one, zero))
}

/// Add 32-bit masks in a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_kadd_mask32&expand=3207)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(mov))] // generate normal and code instead of kaddd
                                     //llvm.x86.avx512.kadd.d
pub unsafe fn _kadd_mask32(a: __mmask32, b: __mmask32) -> __mmask32 {
    transmute(a + b)
}

/// Add 64-bit masks in a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_kadd_mask64&expand=3208)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(mov))] // generate normal and code instead of kaddq
pub unsafe fn _kadd_mask64(a: __mmask64, b: __mmask64) -> __mmask64 {
    transmute(a + b)
}

/// Compute the bitwise AND of 32-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_kand_mask32&expand=3213)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(and))] // generate normal and code instead of kandd
pub unsafe fn _kand_mask32(a: __mmask32, b: __mmask32) -> __mmask32 {
    transmute(a & b)
}

/// Compute the bitwise AND of 64-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_kand_mask64&expand=3214)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(and))] // generate normal and code instead of kandq
pub unsafe fn _kand_mask64(a: __mmask64, b: __mmask64) -> __mmask64 {
    transmute(a & b)
}

/// Compute the bitwise NOT of 32-bit mask a, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_knot_mask32&expand=3234)
#[inline]
#[target_feature(enable = "avx512bw")]
pub unsafe fn _knot_mask32(a: __mmask32) -> __mmask32 {
    transmute(a ^ 0b11111111_11111111_11111111_11111111)
}

/// Compute the bitwise NOT of 64-bit mask a, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_knot_mask64&expand=3235)
#[inline]
#[target_feature(enable = "avx512bw")]
pub unsafe fn _knot_mask64(a: __mmask64) -> __mmask64 {
    transmute(a ^ 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111)
}

/// Compute the bitwise NOT of 32-bit masks a and then AND with b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_kandn_mask32&expand=3219)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(not))] // generate normal and code instead of kandnd
pub unsafe fn _kandn_mask32(a: __mmask32, b: __mmask32) -> __mmask32 {
    transmute(_knot_mask32(a) & b)
}

/// Compute the bitwise NOT of 64-bit masks a and then AND with b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_kandn_mask64&expand=3220)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(not))] // generate normal and code instead of kandnq
pub unsafe fn _kandn_mask64(a: __mmask64, b: __mmask64) -> __mmask64 {
    transmute(_knot_mask64(a) & b)
}

/// Compute the bitwise OR of 32-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_kor_mask32&expand=3240)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(or))] // generate normal and code instead of kord
pub unsafe fn _kor_mask32(a: __mmask32, b: __mmask32) -> __mmask32 {
    transmute(a | b)
}

/// Compute the bitwise OR of 64-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_kor_mask64&expand=3241)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(or))] // generate normal and code instead of korq
pub unsafe fn _kor_mask64(a: __mmask64, b: __mmask64) -> __mmask64 {
    transmute(a | b)
}

/// Compute the bitwise XOR of 32-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_kxor_mask32&expand=3292)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(xor))] // generate normal and code instead of kxord
pub unsafe fn _kxor_mask32(a: __mmask32, b: __mmask32) -> __mmask32 {
    transmute(a ^ b)
}

/// Compute the bitwise XOR of 64-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_kxor_mask64&expand=3293)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(xor))] // generate normal and code instead of kxorq
pub unsafe fn _kxor_mask64(a: __mmask64, b: __mmask64) -> __mmask64 {
    transmute(a ^ b)
}

/// Compute the bitwise XNOR of 32-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_kxnor_mask32&expand=3286)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(xor))] // generate normal and code instead of kxnord
pub unsafe fn _kxnor_mask32(a: __mmask32, b: __mmask32) -> __mmask32 {
    transmute(_knot_mask32(a ^ b))
}

/// Compute the bitwise XNOR of 64-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_kxnor_mask64&expand=3287)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(xor))] // generate normal and code instead of kxnorq
pub unsafe fn _kxnor_mask64(a: __mmask64, b: __mmask64) -> __mmask64 {
    transmute(_knot_mask64(a ^ b))
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cvtepi16_epi8&expand=1407)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm512_cvtepi16_epi8(a: __m512i) -> __m256i {
    let a = a.as_i16x32();
    transmute::<i8x32, _>(simd_cast(a))
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cvtepi16_epi8&expand=1408)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm512_mask_cvtepi16_epi8(src: __m256i, k: __mmask32, a: __m512i) -> __m256i {
    let convert = _mm512_cvtepi16_epi8(a).as_i8x32();
    transmute(simd_select_bitmask(k, convert, src.as_i8x32()))
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_cvtepi16_epi8&expand=1409)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm512_maskz_cvtepi16_epi8(k: __mmask32, a: __m512i) -> __m256i {
    let convert = _mm512_cvtepi16_epi8(a).as_i8x32();
    transmute(simd_select_bitmask(
        k,
        convert,
        _mm256_setzero_si256().as_i8x32(),
    ))
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepi16_epi8&expand=1404)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm256_cvtepi16_epi8(a: __m256i) -> __m128i {
    let a = a.as_i16x16();
    transmute::<i8x16, _>(simd_cast(a))
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cvtepi16_epi8&expand=1405)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm256_mask_cvtepi16_epi8(src: __m128i, k: __mmask16, a: __m256i) -> __m128i {
    let convert = _mm256_cvtepi16_epi8(a).as_i8x16();
    transmute(simd_select_bitmask(k, convert, src.as_i8x16()))
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_cvtepi16_epi8&expand=1406)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm256_maskz_cvtepi16_epi8(k: __mmask16, a: __m256i) -> __m128i {
    let convert = _mm256_cvtepi16_epi8(a).as_i8x16();
    transmute(simd_select_bitmask(
        k,
        convert,
        _mm_setzero_si128().as_i8x16(),
    ))
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtepi16_epi8&expand=1401)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm_cvtepi16_epi8(a: __m128i) -> __m128i {
    let a = a.as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    let v256: i16x16 = simd_shuffle16(a, zero, [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8]);
    transmute::<i8x16, _>(simd_cast(v256))
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cvtepi16_epi8&expand=1402)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm_mask_cvtepi16_epi8(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    let convert = _mm_cvtepi16_epi8(a).as_i8x16();
    let k: __mmask16 = 0b11111111_11111111 & k as __mmask16;
    transmute(simd_select_bitmask(k, convert, src.as_i8x16()))
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_cvtepi16_epi8&expand=1403)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm_maskz_cvtepi16_epi8(k: __mmask8, a: __m128i) -> __m128i {
    let convert = _mm_cvtepi16_epi8(a).as_i8x16();
    let k: __mmask16 = 0b11111111_11111111 & k as __mmask16;
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, convert, zero))
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cvtsepi16_epi8&expand=1807)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm512_cvtsepi16_epi8(a: __m512i) -> __m256i {
    transmute(vpmovswb(
        a.as_i16x32(),
        _mm256_setzero_si256().as_i8x32(),
        0b11111111_11111111_11111111_11111111,
    ))
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cvtsepi16_epi8&expand=1808)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm512_mask_cvtsepi16_epi8(src: __m256i, k: __mmask32, a: __m512i) -> __m256i {
    transmute(vpmovswb(a.as_i16x32(), src.as_i8x32(), k))
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_cvtsepi16_epi8&expand=1809)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm512_maskz_cvtsepi16_epi8(k: __mmask32, a: __m512i) -> __m256i {
    transmute(vpmovswb(
        a.as_i16x32(),
        _mm256_setzero_si256().as_i8x32(),
        k,
    ))
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtsepi16_epi8&expand=1804)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm256_cvtsepi16_epi8(a: __m256i) -> __m128i {
    transmute(vpmovswb256(
        a.as_i16x16(),
        _mm_setzero_si128().as_i8x16(),
        0b11111111_11111111,
    ))
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cvtsepi16_epi8&expand=1805)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm256_mask_cvtsepi16_epi8(src: __m128i, k: __mmask16, a: __m256i) -> __m128i {
    transmute(vpmovswb256(a.as_i16x16(), src.as_i8x16(), k))
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_cvtsepi16_epi8&expand=1806)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm256_maskz_cvtsepi16_epi8(k: __mmask16, a: __m256i) -> __m128i {
    transmute(vpmovswb256(
        a.as_i16x16(),
        _mm_setzero_si128().as_i8x16(),
        k,
    ))
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtsepi16_epi8&expand=1801)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm_cvtsepi16_epi8(a: __m128i) -> __m128i {
    transmute(vpmovswb128(
        a.as_i16x8(),
        _mm_setzero_si128().as_i8x16(),
        0b11111111,
    ))
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cvtsepi16_epi8&expand=1802)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm_mask_cvtsepi16_epi8(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    transmute(vpmovswb128(a.as_i16x8(), src.as_i8x16(), k))
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_cvtsepi16_epi8&expand=1803)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm_maskz_cvtsepi16_epi8(k: __mmask8, a: __m128i) -> __m128i {
    transmute(vpmovswb128(a.as_i16x8(), _mm_setzero_si128().as_i8x16(), k))
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cvtusepi16_epi8&expand=2042)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm512_cvtusepi16_epi8(a: __m512i) -> __m256i {
    transmute(vpmovuswb(
        a.as_u16x32(),
        _mm256_setzero_si256().as_u8x32(),
        0b11111111_11111111_11111111_11111111,
    ))
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cvtusepi16_epi8&expand=2043)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm512_mask_cvtusepi16_epi8(src: __m256i, k: __mmask32, a: __m512i) -> __m256i {
    transmute(vpmovuswb(a.as_u16x32(), src.as_u8x32(), k))
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_cvtusepi16_epi8&expand=2044)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm512_maskz_cvtusepi16_epi8(k: __mmask32, a: __m512i) -> __m256i {
    transmute(vpmovuswb(
        a.as_u16x32(),
        _mm256_setzero_si256().as_u8x32(),
        k,
    ))
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtusepi16_epi8&expand=2039)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm256_cvtusepi16_epi8(a: __m256i) -> __m128i {
    transmute(vpmovuswb256(
        a.as_u16x16(),
        _mm_setzero_si128().as_u8x16(),
        0b11111111_11111111,
    ))
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cvtusepi16_epi8&expand=2040)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm256_mask_cvtusepi16_epi8(src: __m128i, k: __mmask16, a: __m256i) -> __m128i {
    transmute(vpmovuswb256(a.as_u16x16(), src.as_u8x16(), k))
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_cvtusepi16_epi8&expand=2041)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm256_maskz_cvtusepi16_epi8(k: __mmask16, a: __m256i) -> __m128i {
    transmute(vpmovuswb256(
        a.as_u16x16(),
        _mm_setzero_si128().as_u8x16(),
        k,
    ))
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtusepi16_epi8&expand=2036)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm_cvtusepi16_epi8(a: __m128i) -> __m128i {
    transmute(vpmovuswb128(
        a.as_u16x8(),
        _mm_setzero_si128().as_u8x16(),
        0b11111111,
    ))
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cvtusepi16_epi8&expand=2037)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm_mask_cvtusepi16_epi8(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    transmute(vpmovuswb128(a.as_u16x8(), src.as_u8x16(), k))
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_cvtusepi16_epi8&expand=2038)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm_maskz_cvtusepi16_epi8(k: __mmask8, a: __m128i) -> __m128i {
    transmute(vpmovuswb128(
        a.as_u16x8(),
        _mm_setzero_si128().as_u8x16(),
        k,
    ))
}

/// Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cvtepi8_epi16&expand=1526)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub unsafe fn _mm512_cvtepi8_epi16(a: __m256i) -> __m512i {
    let a = a.as_i8x32();
    transmute::<i16x32, _>(simd_cast(a))
}

/// Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cvtepi8_epi16&expand=1527)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub unsafe fn _mm512_mask_cvtepi8_epi16(src: __m512i, k: __mmask32, a: __m256i) -> __m512i {
    let convert = _mm512_cvtepi8_epi16(a).as_i16x32();
    transmute(simd_select_bitmask(k, convert, src.as_i16x32()))
}

/// Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_cvtepi8_epi16&expand=1528)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub unsafe fn _mm512_maskz_cvtepi8_epi16(k: __mmask32, a: __m256i) -> __m512i {
    let convert = _mm512_cvtepi8_epi16(a).as_i16x32();
    transmute(simd_select_bitmask(
        k,
        convert,
        _mm512_setzero_si512().as_i16x32(),
    ))
}

/// Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cvtepi8_epi16&expand=1524)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub unsafe fn _mm256_mask_cvtepi8_epi16(src: __m256i, k: __mmask16, a: __m128i) -> __m256i {
    let convert = _mm256_cvtepi8_epi16(a).as_i16x16();
    transmute(simd_select_bitmask(k, convert, src.as_i16x16()))
}

/// Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_cvtepi8_epi16&expand=1525)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub unsafe fn _mm256_maskz_cvtepi8_epi16(k: __mmask16, a: __m128i) -> __m256i {
    let convert = _mm256_cvtepi8_epi16(a).as_i16x16();
    transmute(simd_select_bitmask(
        k,
        convert,
        _mm256_setzero_si256().as_i16x16(),
    ))
}

/// Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cvtepi8_epi16&expand=1521)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub unsafe fn _mm_mask_cvtepi8_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    let convert = _mm_cvtepi8_epi16(a).as_i16x8();
    transmute(simd_select_bitmask(k, convert, src.as_i16x8()))
}

/// Sign extend packed 8-bit integers in a to packed 16-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_cvtepi8_epi16&expand=1522)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub unsafe fn _mm_maskz_cvtepi8_epi16(k: __mmask8, a: __m128i) -> __m128i {
    let convert = _mm_cvtepi8_epi16(a).as_i16x8();
    transmute(simd_select_bitmask(
        k,
        convert,
        _mm_setzero_si128().as_i16x8(),
    ))
}

/// Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cvtepu8_epi16&expand=1612)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub unsafe fn _mm512_cvtepu8_epi16(a: __m256i) -> __m512i {
    let a = a.as_u8x32();
    transmute::<i16x32, _>(simd_cast(a))
}

/// Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cvtepu8_epi16&expand=1613)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub unsafe fn _mm512_mask_cvtepu8_epi16(src: __m512i, k: __mmask32, a: __m256i) -> __m512i {
    let convert = _mm512_cvtepu8_epi16(a).as_i16x32();
    transmute(simd_select_bitmask(k, convert, src.as_i16x32()))
}

/// Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_cvtepu8_epi16&expand=1614)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub unsafe fn _mm512_maskz_cvtepu8_epi16(k: __mmask32, a: __m256i) -> __m512i {
    let convert = _mm512_cvtepu8_epi16(a).as_i16x32();
    transmute(simd_select_bitmask(
        k,
        convert,
        _mm512_setzero_si512().as_i16x32(),
    ))
}

/// Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cvtepu8_epi16&expand=1610)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub unsafe fn _mm256_mask_cvtepu8_epi16(src: __m256i, k: __mmask16, a: __m128i) -> __m256i {
    let convert = _mm256_cvtepu8_epi16(a).as_i16x16();
    transmute(simd_select_bitmask(k, convert, src.as_i16x16()))
}

/// Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_cvtepu8_epi16&expand=1611)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub unsafe fn _mm256_maskz_cvtepu8_epi16(k: __mmask16, a: __m128i) -> __m256i {
    let convert = _mm256_cvtepu8_epi16(a).as_i16x16();
    transmute(simd_select_bitmask(
        k,
        convert,
        _mm256_setzero_si256().as_i16x16(),
    ))
}

/// Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cvtepu8_epi16&expand=1607)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub unsafe fn _mm_mask_cvtepu8_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    let convert = _mm_cvtepu8_epi16(a).as_i16x8();
    transmute(simd_select_bitmask(k, convert, src.as_i16x8()))
}

/// Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_cvtepu8_epi16&expand=1608)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub unsafe fn _mm_maskz_cvtepu8_epi16(k: __mmask8, a: __m128i) -> __m128i {
    let convert = _mm_cvtepu8_epi16(a).as_i16x8();
    transmute(simd_select_bitmask(
        k,
        convert,
        _mm_setzero_si128().as_i16x8(),
    ))
}

/// Shift 128-bit lanes in a left by imm8 bytes while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_bslli_epi128&expand=591)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(1)]
#[cfg_attr(test, assert_instr(vpslldq, imm8 = 3))]
pub unsafe fn _mm512_bslli_epi128(a: __m512i, imm8: i32) -> __m512i {
    let a = a.as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    #[rustfmt::skip]
    macro_rules! call {
        ($imm8:expr) => {
            simd_shuffle64 (
                zero,
                a,
                [
                    64 - $imm8, 65 - $imm8, 66 - $imm8, 67 - $imm8, 68 - $imm8, 69 - $imm8, 70 - $imm8, 71 - $imm8,
                    72 - $imm8, 73 - $imm8, 74 - $imm8, 75 - $imm8, 76 - $imm8, 77 - $imm8, 78 - $imm8, 79 - $imm8,
                    80 - ($imm8+16), 81 - ($imm8+16), 82 - ($imm8+16), 83 - ($imm8+16), 84 - ($imm8+16), 85 - ($imm8+16), 86 - ($imm8+16), 87 - ($imm8+16),
                    88 - ($imm8+16), 89 - ($imm8+16), 90 - ($imm8+16), 91 - ($imm8+16), 92 - ($imm8+16), 93 - ($imm8+16), 94 - ($imm8+16), 95 - ($imm8+16),
                    96 - ($imm8+32), 97 - ($imm8+32), 98 - ($imm8+32), 99 - ($imm8+32), 100 - ($imm8+32), 101 - ($imm8+32), 102 - ($imm8+32), 103 - ($imm8+32),
                    104 - ($imm8+32), 105 - ($imm8+32), 106 - ($imm8+32), 107 - ($imm8+32), 108 - ($imm8+32), 109 - ($imm8+32), 110 - ($imm8+32), 111 - ($imm8+32),
                    112 - ($imm8+48), 113 - ($imm8+48), 114 - ($imm8+48), 115 - ($imm8+48), 116 - ($imm8+48), 117 - ($imm8+48), 118 - ($imm8+48), 119 - ($imm8+48),
                    120 - ($imm8+48), 121 - ($imm8+48), 122 - ($imm8+48), 123 - ($imm8+48), 124 - ($imm8+48), 125 - ($imm8+48), 126 - ($imm8+48), 127 - ($imm8+48),
                ],
            )
        };
    }
    let r: i8x64 = match imm8 {
        0 => call!(0),
        1 => call!(1),
        2 => call!(2),
        3 => call!(3),
        4 => call!(4),
        5 => call!(5),
        6 => call!(6),
        7 => call!(7),
        8 => call!(8),
        9 => call!(9),
        10 => call!(10),
        11 => call!(11),
        12 => call!(12),
        13 => call!(13),
        14 => call!(14),
        15 => call!(15),
        _ => call!(16),
    };
    transmute(r)
}

/// Shift 128-bit lanes in a right by imm8 bytes while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_bsrli_epi128&expand=594)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(1)]
#[cfg_attr(test, assert_instr(vpsrldq, imm8 = 3))]
pub unsafe fn _mm512_bsrli_epi128(a: __m512i, imm8: i32) -> __m512i {
    let a = a.as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    #[rustfmt::skip]
    macro_rules! call {
        ($imm8:expr) => {
            simd_shuffle64 (
                a,
                zero,
                [
                    0 + ($imm8+48), 1 + ($imm8+48), 2 + ($imm8+48), 3 + ($imm8+48), 4 + ($imm8+48), 5 + ($imm8+48), 6 + ($imm8+48), 7 + ($imm8+48),
                    8 + ($imm8+48), 9 + ($imm8+48), 10 + ($imm8+48), 11 + ($imm8+48), 12 + ($imm8+48), 13 + ($imm8+48), 14 + ($imm8+48), 15 + ($imm8+48),
                    16 + ($imm8+32), 17 + ($imm8+32), 18 + ($imm8+32), 19 + ($imm8+32), 20 + ($imm8+32), 21 + ($imm8+32), 22 + ($imm8+32), 23 + ($imm8+32),
                    24 + ($imm8+32), 25 + ($imm8+32), 26 + ($imm8+32), 27 + ($imm8+32), 28 + ($imm8+32), 29 + ($imm8+32), 30 + ($imm8+32), 31 + ($imm8+32),
                    32 + ($imm8+16), 33 + ($imm8+16), 34 + ($imm8+16), 35 + ($imm8+16), 36 + ($imm8+16), 37 + ($imm8+16), 38 + ($imm8+16), 39 + ($imm8+16),
                    40 + ($imm8+16), 41 + ($imm8+16), 42 + ($imm8+16), 43 + ($imm8+16), 44 + ($imm8+16), 45 + ($imm8+16), 46 + ($imm8+16), 47 + ($imm8+16),
                    48 + $imm8, 49 + $imm8, 50 + $imm8, 51 + $imm8, 52 + $imm8, 53 + $imm8, 54 + $imm8, 55 + $imm8,
                    56 + $imm8, 57 + $imm8, 58 + $imm8, 59 + $imm8, 60 + $imm8, 61 + $imm8, 62 + $imm8, 63 + $imm8,
                ],
            )
        };
    }
    let r: i8x64 = match imm8 {
        0 => call!(0),
        1 => call!(1),
        2 => call!(2),
        3 => call!(3),
        4 => call!(4),
        5 => call!(5),
        6 => call!(6),
        7 => call!(7),
        8 => call!(8),
        9 => call!(9),
        10 => call!(10),
        11 => call!(11),
        12 => call!(12),
        13 => call!(13),
        14 => call!(14),
        15 => call!(15),
        _ => call!(16),
    };
    transmute(r)
}

/// Concatenate pairs of 16-byte blocks in a and b into a 32-byte temporary result, shift the result right by imm8 bytes, and store the low 16 bytes in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_alignr_epi8&expand=263)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpalignr, imm8 = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_alignr_epi8(a: __m512i, b: __m512i, imm8: i32) -> __m512i {
    // If palignr is shifting the pair of vectors more than the size of two
    // lanes, emit zero.
    if imm8 > 32 {
        return _mm512_set1_epi8(0);
    }
    // If palignr is shifting the pair of input vectors more than one lane,
    // but less than two lanes, convert to shifting in zeroes.
    let (a, b, imm8) = if imm8 > 16 {
        (_mm512_set1_epi8(0), a, imm8 - 16)
    } else {
        (a, b, imm8)
    };
    let a = a.as_i8x64();
    let b = b.as_i8x64();
    #[rustfmt::skip]
    macro_rules! shuffle {
        ($imm8:expr) => {
            simd_shuffle64(
                b,
                a,
                [
                    0 + ($imm8+48), 1 + ($imm8+48), 2 + ($imm8+48), 3 + ($imm8+48), 4 + ($imm8+48), 5 + ($imm8+48), 6 + ($imm8+48), 7 + ($imm8+48),
                    8 + ($imm8+48), 9 + ($imm8+48), 10 + ($imm8+48), 11 + ($imm8+48), 12 + ($imm8+48), 13 + ($imm8+48), 14 + ($imm8+48), 15 + ($imm8+48),
                    16 + ($imm8+32), 17 + ($imm8+32), 18 + ($imm8+32), 19 + ($imm8+32), 20 + ($imm8+32), 21 + ($imm8+32), 22 + ($imm8+32), 23 + ($imm8+32),
                    24 + ($imm8+32), 25 + ($imm8+32), 26 + ($imm8+32), 27 + ($imm8+32), 28 + ($imm8+32), 29 + ($imm8+32), 30 + ($imm8+32), 31 + ($imm8+32),
                    32 + ($imm8+16), 33 + ($imm8+16), 34 + ($imm8+16), 35 + ($imm8+16), 36 + ($imm8+16), 37 + ($imm8+16), 38 + ($imm8+16), 39 + ($imm8+16),
                    40 + ($imm8+16), 41 + ($imm8+16), 42 + ($imm8+16), 43 + ($imm8+16), 44 + ($imm8+16), 45 + ($imm8+16), 46 + ($imm8+16), 47 + ($imm8+16),
                    48 + $imm8, 49 + $imm8, 50 + $imm8, 51 + $imm8, 52 + $imm8, 53 + $imm8, 54 + $imm8, 55 + $imm8,
                    56 + $imm8, 57 + $imm8, 58 + $imm8, 59 + $imm8, 60 + $imm8, 61 + $imm8, 62 + $imm8, 63 + $imm8,
                ],
            )
        };
    }
    let r: i8x64 = match imm8 {
        0 => shuffle!(0),
        1 => shuffle!(1),
        2 => shuffle!(2),
        3 => shuffle!(3),
        4 => shuffle!(4),
        5 => shuffle!(5),
        6 => shuffle!(6),
        7 => shuffle!(7),
        8 => shuffle!(8),
        9 => shuffle!(9),
        10 => shuffle!(10),
        11 => shuffle!(11),
        12 => shuffle!(12),
        13 => shuffle!(13),
        14 => shuffle!(14),
        15 => shuffle!(15),
        _ => shuffle!(16),
    };
    transmute(r)
}

/// Concatenate pairs of 16-byte blocks in a and b into a 32-byte temporary result, shift the result right by imm8 bytes, and store the low 16 bytes in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_alignr_epi8&expand=264)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpalignr, imm8 = 1))]
#[rustc_args_required_const(4)]
pub unsafe fn _mm512_mask_alignr_epi8(
    src: __m512i,
    k: __mmask64,
    a: __m512i,
    b: __m512i,
    imm8: i32,
) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm512_alignr_epi8(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, r.as_i8x64(), src.as_i8x64()))
}

/// Concatenate pairs of 16-byte blocks in a and b into a 32-byte temporary result, shift the result right by imm8 bytes, and store the low 16 bytes in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_alignr_epi8&expand=265)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpalignr, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_maskz_alignr_epi8(k: __mmask64, a: __m512i, b: __m512i, imm8: i32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm512_alignr_epi8(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, r.as_i8x64(), zero))
}

/// Concatenate pairs of 16-byte blocks in a and b into a 32-byte temporary result, shift the result right by imm8 bytes, and store the low 16 bytes in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_alignr_epi8&expand=261)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(4)]
#[cfg_attr(test, assert_instr(vpalignr, imm8 = 5))]
pub unsafe fn _mm256_mask_alignr_epi8(
    src: __m256i,
    k: __mmask32,
    a: __m256i,
    b: __m256i,
    imm8: i32,
) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm256_alignr_epi8(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, r.as_i8x32(), src.as_i8x32()))
}

/// Concatenate pairs of 16-byte blocks in a and b into a 32-byte temporary result, shift the result right by imm8 bytes, and store the low 16 bytes in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_alignr_epi8&expand=262)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpalignr, imm8 = 5))]
pub unsafe fn _mm256_maskz_alignr_epi8(k: __mmask32, a: __m256i, b: __m256i, imm8: i32) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm256_alignr_epi8(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(
        k,
        r.as_i8x32(),
        _mm256_setzero_si256().as_i8x32(),
    ))
}

/// Concatenate pairs of 16-byte blocks in a and b into a 32-byte temporary result, shift the result right by imm8 bytes, and store the low 16 bytes in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_alignr_epi8&expand=258)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(4)]
#[cfg_attr(test, assert_instr(vpalignr, imm8 = 5))]
pub unsafe fn _mm_mask_alignr_epi8(
    src: __m128i,
    k: __mmask16,
    a: __m128i,
    b: __m128i,
    imm8: i32,
) -> __m128i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm_alignr_epi8(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, r.as_i8x16(), src.as_i8x16()))
}

/// Concatenate pairs of 16-byte blocks in a and b into a 32-byte temporary result, shift the result right by imm8 bytes, and store the low 16 bytes in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_alignr_epi8&expand=259)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpalignr, imm8 = 5))]
pub unsafe fn _mm_maskz_alignr_epi8(k: __mmask16, a: __m128i, b: __m128i, imm8: i32) -> __m128i {
    macro_rules! call {
        ($imm8:expr) => {
            _mm_alignr_epi8(a, b, $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, r.as_i8x16(), zero))
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cvtsepi16_storeu_epi8&expand=1812)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm512_mask_cvtsepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask32, a: __m512i) {
    vpmovswbmem(mem_addr as *mut i8, a.as_i16x32(), k);
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cvtsepi16_storeu_epi8&expand=1811)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm256_mask_cvtsepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask16, a: __m256i) {
    vpmovswbmem256(mem_addr as *mut i8, a.as_i16x16(), k);
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cvtsepi16_storeu_epi8&expand=1810)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovswb))]
pub unsafe fn _mm_mask_cvtsepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask8, a: __m128i) {
    vpmovswbmem128(mem_addr as *mut i8, a.as_i16x8(), k);
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cvtepi16_storeu_epi8&expand=1412)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm512_mask_cvtepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask32, a: __m512i) {
    vpmovwbmem(mem_addr as *mut i8, a.as_i16x32(), k);
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cvtepi16_storeu_epi8&expand=1411)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm256_mask_cvtepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask16, a: __m256i) {
    vpmovwbmem256(mem_addr as *mut i8, a.as_i16x16(), k);
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cvtepi16_storeu_epi8&expand=1410)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovwb))]
pub unsafe fn _mm_mask_cvtepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask8, a: __m128i) {
    vpmovwbmem128(mem_addr as *mut i8, a.as_i16x8(), k);
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cvtusepi16_storeu_epi8&expand=2047)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm512_mask_cvtusepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask32, a: __m512i) {
    vpmovuswbmem(mem_addr as *mut i8, a.as_i16x32(), k);
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cvtusepi16_storeu_epi8&expand=2046)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm256_mask_cvtusepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask16, a: __m256i) {
    vpmovuswbmem256(mem_addr as *mut i8, a.as_i16x16(), k);
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cvtusepi16_storeu_epi8&expand=2045)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
#[cfg_attr(test, assert_instr(vpmovuswb))]
pub unsafe fn _mm_mask_cvtusepi16_storeu_epi8(mem_addr: *mut i8, k: __mmask8, a: __m128i) {
    vpmovuswbmem128(mem_addr as *mut i8, a.as_i16x8(), k);
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.mask.paddus.w.512"]
    fn vpaddusw(a: u16x32, b: u16x32, src: u16x32, mask: u32) -> u16x32;
    #[link_name = "llvm.x86.avx512.mask.paddus.w.256"]
    fn vpaddusw256(a: u16x16, b: u16x16, src: u16x16, mask: u16) -> u16x16;
    #[link_name = "llvm.x86.avx512.mask.paddus.w.128"]
    fn vpaddusw128(a: u16x8, b: u16x8, src: u16x8, mask: u8) -> u16x8;

    #[link_name = "llvm.x86.avx512.mask.paddus.b.512"]
    fn vpaddusb(a: u8x64, b: u8x64, src: u8x64, mask: u64) -> u8x64;
    #[link_name = "llvm.x86.avx512.mask.paddus.b.256"]
    fn vpaddusb256(a: u8x32, b: u8x32, src: u8x32, mask: u32) -> u8x32;
    #[link_name = "llvm.x86.avx512.mask.paddus.b.128"]
    fn vpaddusb128(a: u8x16, b: u8x16, src: u8x16, mask: u16) -> u8x16;

    #[link_name = "llvm.x86.avx512.mask.padds.w.512"]
    fn vpaddsw(a: i16x32, b: i16x32, src: i16x32, mask: u32) -> i16x32;
    #[link_name = "llvm.x86.avx512.mask.padds.w.256"]
    fn vpaddsw256(a: i16x16, b: i16x16, src: i16x16, mask: u16) -> i16x16;
    #[link_name = "llvm.x86.avx512.mask.padds.w.128"]
    fn vpaddsw128(a: i16x8, b: i16x8, src: i16x8, mask: u8) -> i16x8;

    #[link_name = "llvm.x86.avx512.mask.padds.b.512"]
    fn vpaddsb(a: i8x64, b: i8x64, src: i8x64, mask: u64) -> i8x64;
    #[link_name = "llvm.x86.avx512.mask.padds.b.256"]
    fn vpaddsb256(a: i8x32, b: i8x32, src: i8x32, mask: u32) -> i8x32;
    #[link_name = "llvm.x86.avx512.mask.padds.b.128"]
    fn vpaddsb128(a: i8x16, b: i8x16, src: i8x16, mask: u16) -> i8x16;

    #[link_name = "llvm.x86.avx512.mask.psubus.w.512"]
    fn vpsubusw(a: u16x32, b: u16x32, src: u16x32, mask: u32) -> u16x32;
    #[link_name = "llvm.x86.avx512.mask.psubus.w.256"]
    fn vpsubusw256(a: u16x16, b: u16x16, src: u16x16, mask: u16) -> u16x16;
    #[link_name = "llvm.x86.avx512.mask.psubus.w.128"]
    fn vpsubusw128(a: u16x8, b: u16x8, src: u16x8, mask: u8) -> u16x8;

    #[link_name = "llvm.x86.avx512.mask.psubus.b.512"]
    fn vpsubusb(a: u8x64, b: u8x64, src: u8x64, mask: u64) -> u8x64;
    #[link_name = "llvm.x86.avx512.mask.psubus.b.256"]
    fn vpsubusb256(a: u8x32, b: u8x32, src: u8x32, mask: u32) -> u8x32;
    #[link_name = "llvm.x86.avx512.mask.psubus.b.128"]
    fn vpsubusb128(a: u8x16, b: u8x16, src: u8x16, mask: u16) -> u8x16;

    #[link_name = "llvm.x86.avx512.mask.psubs.w.512"]
    fn vpsubsw(a: i16x32, b: i16x32, src: i16x32, mask: u32) -> i16x32;
    #[link_name = "llvm.x86.avx512.mask.psubs.w.256"]
    fn vpsubsw256(a: i16x16, b: i16x16, src: i16x16, mask: u16) -> i16x16;
    #[link_name = "llvm.x86.avx512.mask.psubs.w.128"]
    fn vpsubsw128(a: i16x8, b: i16x8, src: i16x8, mask: u8) -> i16x8;

    #[link_name = "llvm.x86.avx512.mask.psubs.b.512"]
    fn vpsubsb(a: i8x64, b: i8x64, src: i8x64, mask: u64) -> i8x64;
    #[link_name = "llvm.x86.avx512.mask.psubs.b.256"]
    fn vpsubsb256(a: i8x32, b: i8x32, src: i8x32, mask: u32) -> i8x32;
    #[link_name = "llvm.x86.avx512.mask.psubs.b.128"]
    fn vpsubsb128(a: i8x16, b: i8x16, src: i8x16, mask: u16) -> i8x16;

    #[link_name = "llvm.x86.avx512.pmulhu.w.512"]
    fn vpmulhuw(a: u16x32, b: u16x32) -> u16x32;
    #[link_name = "llvm.x86.avx512.pmulh.w.512"]
    fn vpmulhw(a: i16x32, b: i16x32) -> i16x32;
    #[link_name = "llvm.x86.avx512.pmul.hr.sw.512"]
    fn vpmulhrsw(a: i16x32, b: i16x32) -> i16x32;

    #[link_name = "llvm.x86.avx512.mask.ucmp.w.512"]
    fn vpcmpuw(a: u16x32, b: u16x32, op: i32, mask: u32) -> u32;
    #[link_name = "llvm.x86.avx512.mask.ucmp.w.256"]
    fn vpcmpuw256(a: u16x16, b: u16x16, op: i32, mask: u16) -> u16;
    #[link_name = "llvm.x86.avx512.mask.ucmp.w.128"]
    fn vpcmpuw128(a: u16x8, b: u16x8, op: i32, mask: u8) -> u8;

    #[link_name = "llvm.x86.avx512.mask.ucmp.b.512"]
    fn vpcmpub(a: u8x64, b: u8x64, op: i32, mask: u64) -> u64;
    #[link_name = "llvm.x86.avx512.mask.ucmp.b.256"]
    fn vpcmpub256(a: u8x32, b: u8x32, op: i32, mask: u32) -> u32;
    #[link_name = "llvm.x86.avx512.mask.ucmp.b.128"]
    fn vpcmpub128(a: u8x16, b: u8x16, op: i32, mask: u16) -> u16;

    #[link_name = "llvm.x86.avx512.mask.cmp.w.512"]
    fn vpcmpw(a: i16x32, b: i16x32, op: i32, mask: u32) -> u32;
    #[link_name = "llvm.x86.avx512.mask.cmp.w.256"]
    fn vpcmpw256(a: i16x16, b: i16x16, op: i32, mask: u16) -> u16;
    #[link_name = "llvm.x86.avx512.mask.cmp.w.128"]
    fn vpcmpw128(a: i16x8, b: i16x8, op: i32, mask: u8) -> u8;

    #[link_name = "llvm.x86.avx512.mask.cmp.b.512"]
    fn vpcmpb(a: i8x64, b: i8x64, op: i32, mask: u64) -> u64;
    #[link_name = "llvm.x86.avx512.mask.cmp.b.256"]
    fn vpcmpb256(a: i8x32, b: i8x32, op: i32, mask: u32) -> u32;
    #[link_name = "llvm.x86.avx512.mask.cmp.b.128"]
    fn vpcmpb128(a: i8x16, b: i8x16, op: i32, mask: u16) -> u16;

    #[link_name = "llvm.x86.avx512.mask.pmaxu.w.512"]
    fn vpmaxuw(a: u16x32, b: u16x32) -> u16x32;
    #[link_name = "llvm.x86.avx512.mask.pmaxu.b.512"]
    fn vpmaxub(a: u8x64, b: u8x64) -> u8x64;
    #[link_name = "llvm.x86.avx512.mask.pmaxs.w.512"]
    fn vpmaxsw(a: i16x32, b: i16x32) -> i16x32;
    #[link_name = "llvm.x86.avx512.mask.pmaxs.b.512"]
    fn vpmaxsb(a: i8x64, b: i8x64) -> i8x64;

    #[link_name = "llvm.x86.avx512.mask.pminu.w.512"]
    fn vpminuw(a: u16x32, b: u16x32) -> u16x32;
    #[link_name = "llvm.x86.avx512.mask.pminu.b.512"]
    fn vpminub(a: u8x64, b: u8x64) -> u8x64;
    #[link_name = "llvm.x86.avx512.mask.pmins.w.512"]
    fn vpminsw(a: i16x32, b: i16x32) -> i16x32;
    #[link_name = "llvm.x86.avx512.mask.pmins.b.512"]
    fn vpminsb(a: i8x64, b: i8x64) -> i8x64;

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

    #[link_name = "llvm.x86.avx512.pavg.w.512"]
    fn vpavgw(a: u16x32, b: u16x32) -> u16x32;
    #[link_name = "llvm.x86.avx512.pavg.b.512"]
    fn vpavgb(a: u8x64, b: u8x64) -> u8x64;

    #[link_name = "llvm.x86.avx512.psll.w.512"]
    fn vpsllw(a: i16x32, count: i16x8) -> i16x32;
    #[link_name = "llvm.x86.avx512.pslli.w.512"]
    fn vpslliw(a: i16x32, imm8: u32) -> i16x32;

    #[link_name = "llvm.x86.avx512.psllv.w.512"]
    fn vpsllvw(a: i16x32, b: i16x32) -> i16x32;
    #[link_name = "llvm.x86.avx512.psllv.w.256"]
    fn vpsllvw256(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx512.psllv.w.128"]
    fn vpsllvw128(a: i16x8, b: i16x8) -> i16x8;

    #[link_name = "llvm.x86.avx512.psrl.w.512"]
    fn vpsrlw(a: i16x32, count: i16x8) -> i16x32;
    #[link_name = "llvm.x86.avx512.psrli.w.512"]
    fn vpsrliw(a: i16x32, imm8: u32) -> i16x32;

    #[link_name = "llvm.x86.avx512.psrlv.w.512"]
    fn vpsrlvw(a: i16x32, b: i16x32) -> i16x32;
    #[link_name = "llvm.x86.avx512.psrlv.w.256"]
    fn vpsrlvw256(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx512.psrlv.w.128"]
    fn vpsrlvw128(a: i16x8, b: i16x8) -> i16x8;

    #[link_name = "llvm.x86.avx512.psra.w.512"]
    fn vpsraw(a: i16x32, count: i16x8) -> i16x32;
    #[link_name = "llvm.x86.avx512.psrai.w.512"]
    fn vpsraiw(a: i16x32, imm8: u32) -> i16x32;

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
        let m = _mm_cmp_epu16_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmp_epu16_mask() {
        let a = _mm_set1_epi16(0);
        let b = _mm_set1_epi16(1);
        let mask = 0b01010101;
        let r = _mm_mask_cmp_epu16_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmp_epu8_mask() {
        let a = _mm512_set1_epi8(0);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmp_epu8_mask(a, b, _MM_CMPINT_LT);
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
        let r = _mm512_mask_cmp_epu8_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmp_epu8_mask() {
        let a = _mm256_set1_epi8(0);
        let b = _mm256_set1_epi8(1);
        let m = _mm256_cmp_epu8_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmp_epu8_mask() {
        let a = _mm256_set1_epi8(0);
        let b = _mm256_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmp_epu8_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmp_epu8_mask() {
        let a = _mm_set1_epi8(0);
        let b = _mm_set1_epi8(1);
        let m = _mm_cmp_epu8_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmp_epu8_mask() {
        let a = _mm_set1_epi8(0);
        let b = _mm_set1_epi8(1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmp_epu8_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmp_epi16_mask() {
        let a = _mm512_set1_epi16(0);
        let b = _mm512_set1_epi16(1);
        let m = _mm512_cmp_epi16_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmp_epi16_mask() {
        let a = _mm512_set1_epi16(0);
        let b = _mm512_set1_epi16(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmp_epi16_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmp_epi16_mask() {
        let a = _mm256_set1_epi16(0);
        let b = _mm256_set1_epi16(1);
        let m = _mm256_cmp_epi16_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmp_epi16_mask() {
        let a = _mm256_set1_epi16(0);
        let b = _mm256_set1_epi16(1);
        let mask = 0b01010101_01010101;
        let r = _mm256_mask_cmp_epi16_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmp_epi16_mask() {
        let a = _mm_set1_epi16(0);
        let b = _mm_set1_epi16(1);
        let m = _mm_cmp_epi16_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmp_epi16_mask() {
        let a = _mm_set1_epi16(0);
        let b = _mm_set1_epi16(1);
        let mask = 0b01010101;
        let r = _mm_mask_cmp_epi16_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmp_epi8_mask() {
        let a = _mm512_set1_epi8(0);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_LT);
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
        let r = _mm512_mask_cmp_epi8_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(
            r,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101
        );
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_cmp_epi8_mask() {
        let a = _mm256_set1_epi8(0);
        let b = _mm256_set1_epi8(1);
        let m = _mm256_cmp_epi8_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_cmp_epi8_mask() {
        let a = _mm256_set1_epi8(0);
        let b = _mm256_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm256_mask_cmp_epi8_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_cmp_epi8_mask() {
        let a = _mm_set1_epi8(0);
        let b = _mm_set1_epi8(1);
        let m = _mm_cmp_epi8_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11111111_11111111);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_cmp_epi8_mask() {
        let a = _mm_set1_epi8(0);
        let b = _mm_set1_epi8(1);
        let mask = 0b01010101_01010101;
        let r = _mm_mask_cmp_epi8_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01010101_01010101);
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
        let r = _mm512_slli_epi16(a, 1);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_slli_epi16() {
        let a = _mm512_set1_epi16(1 << 15);
        let r = _mm512_mask_slli_epi16(a, 0, a, 1);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_slli_epi16(a, 0b11111111_11111111_11111111_11111111, a, 1);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_slli_epi16() {
        let a = _mm512_set1_epi16(1 << 15);
        let r = _mm512_maskz_slli_epi16(0, a, 1);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_slli_epi16(0b11111111_11111111_11111111_11111111, a, 1);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_slli_epi16() {
        let a = _mm256_set1_epi16(1 << 15);
        let r = _mm256_mask_slli_epi16(a, 0, a, 1);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_slli_epi16(a, 0b11111111_11111111, a, 1);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_slli_epi16() {
        let a = _mm256_set1_epi16(1 << 15);
        let r = _mm256_maskz_slli_epi16(0, a, 1);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_slli_epi16(0b11111111_11111111, a, 1);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_slli_epi16() {
        let a = _mm_set1_epi16(1 << 15);
        let r = _mm_mask_slli_epi16(a, 0, a, 1);
        assert_eq_m128i(r, a);
        let r = _mm_mask_slli_epi16(a, 0b11111111, a, 1);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_slli_epi16() {
        let a = _mm_set1_epi16(1 << 15);
        let r = _mm_maskz_slli_epi16(0, a, 1);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_slli_epi16(0b11111111, a, 1);
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
        let r = _mm512_srli_epi16(a, 2);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_srli_epi16() {
        let a = _mm512_set1_epi16(1 << 1);
        let r = _mm512_mask_srli_epi16(a, 0, a, 2);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_srli_epi16(a, 0b11111111_11111111_11111111_11111111, a, 2);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_srli_epi16() {
        let a = _mm512_set1_epi16(1 << 1);
        let r = _mm512_maskz_srli_epi16(0, a, 2);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_srli_epi16(0b11111111_11111111_11111111_11111111, a, 2);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_srli_epi16() {
        let a = _mm256_set1_epi16(1 << 1);
        let r = _mm256_mask_srli_epi16(a, 0, a, 2);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_srli_epi16(a, 0b11111111_11111111, a, 2);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_srli_epi16() {
        let a = _mm256_set1_epi16(1 << 1);
        let r = _mm256_maskz_srli_epi16(0, a, 2);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_srli_epi16(0b11111111_11111111, a, 2);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_srli_epi16() {
        let a = _mm_set1_epi16(1 << 1);
        let r = _mm_mask_srli_epi16(a, 0, a, 2);
        assert_eq_m128i(r, a);
        let r = _mm_mask_srli_epi16(a, 0b11111111, a, 2);
        let e = _mm_set1_epi16(0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_srli_epi16() {
        let a = _mm_set1_epi16(1 << 1);
        let r = _mm_maskz_srli_epi16(0, a, 2);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_srli_epi16(0b11111111, a, 2);
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
        let r = _mm512_srai_epi16(a, 2);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_srai_epi16() {
        let a = _mm512_set1_epi16(8);
        let r = _mm512_mask_srai_epi16(a, 0, a, 2);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_srai_epi16(a, 0b11111111_11111111_11111111_11111111, a, 2);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_srai_epi16() {
        let a = _mm512_set1_epi16(8);
        let r = _mm512_maskz_srai_epi16(0, a, 2);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_srai_epi16(0b11111111_11111111_11111111_11111111, a, 2);
        let e = _mm512_set1_epi16(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_srai_epi16() {
        let a = _mm256_set1_epi16(8);
        let r = _mm256_mask_srai_epi16(a, 0, a, 2);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_srai_epi16(a, 0b11111111_11111111, a, 2);
        let e = _mm256_set1_epi16(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_srai_epi16() {
        let a = _mm256_set1_epi16(8);
        let r = _mm256_maskz_srai_epi16(0, a, 2);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_srai_epi16(0b11111111_11111111, a, 2);
        let e = _mm256_set1_epi16(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_srai_epi16() {
        let a = _mm_set1_epi16(8);
        let r = _mm_mask_srai_epi16(a, 0, a, 2);
        assert_eq_m128i(r, a);
        let r = _mm_mask_srai_epi16(a, 0b11111111, a, 2);
        let e = _mm_set1_epi16(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_srai_epi16() {
        let a = _mm_set1_epi16(8);
        let r = _mm_maskz_srai_epi16(0, a, 2);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_srai_epi16(0b11111111, a, 2);
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
        let r = _mm512_shufflelo_epi16(a, 0b00_01_01_11);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_shufflelo_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        let r = _mm512_mask_shufflelo_epi16(a, 0, a, 0b00_01_01_11);
        assert_eq_m512i(r, a);
        let r =
            _mm512_mask_shufflelo_epi16(a, 0b11111111_11111111_11111111_11111111, a, 0b00_01_01_11);
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
        let r = _mm512_maskz_shufflelo_epi16(0, a, 0b00_01_01_11);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r =
            _mm512_maskz_shufflelo_epi16(0b11111111_11111111_11111111_11111111, a, 0b00_01_01_11);
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
        let r = _mm256_mask_shufflelo_epi16(a, 0, a, 0b00_01_01_11);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shufflelo_epi16(a, 0b11111111_11111111, a, 0b00_01_01_11);
        let e = _mm256_set_epi16(0, 1, 2, 3, 7, 6, 6, 4, 8, 9, 10, 11, 15, 14, 14, 12);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_shufflelo_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_maskz_shufflelo_epi16(0, a, 0b00_01_01_11);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shufflelo_epi16(0b11111111_11111111, a, 0b00_01_01_11);
        let e = _mm256_set_epi16(0, 1, 2, 3, 7, 6, 6, 4, 8, 9, 10, 11, 15, 14, 14, 12);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_shufflelo_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_mask_shufflelo_epi16(a, 0, a, 0b00_01_01_11);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shufflelo_epi16(a, 0b11111111, a, 0b00_01_01_11);
        let e = _mm_set_epi16(0, 1, 2, 3, 7, 6, 6, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_shufflelo_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_maskz_shufflelo_epi16(0, a, 0b00_01_01_11);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shufflelo_epi16(0b11111111, a, 0b00_01_01_11);
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
        let r = _mm512_shufflehi_epi16(a, 0b00_01_01_11);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_shufflehi_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        let r = _mm512_mask_shufflehi_epi16(a, 0, a, 0b00_01_01_11);
        assert_eq_m512i(r, a);
        let r =
            _mm512_mask_shufflehi_epi16(a, 0b11111111_11111111_11111111_11111111, a, 0b00_01_01_11);
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
        let r = _mm512_maskz_shufflehi_epi16(0, a, 0b00_01_01_11);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r =
            _mm512_maskz_shufflehi_epi16(0b11111111_11111111_11111111_11111111, a, 0b00_01_01_11);
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
        let r = _mm256_mask_shufflehi_epi16(a, 0, a, 0b00_01_01_11);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_shufflehi_epi16(a, 0b11111111_11111111, a, 0b00_01_01_11);
        let e = _mm256_set_epi16(3, 2, 2, 0, 4, 5, 6, 7, 11, 10, 10, 8, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_shufflehi_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_maskz_shufflehi_epi16(0, a, 0b00_01_01_11);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_shufflehi_epi16(0b11111111_11111111, a, 0b00_01_01_11);
        let e = _mm256_set_epi16(3, 2, 2, 0, 4, 5, 6, 7, 11, 10, 10, 8, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_shufflehi_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_mask_shufflehi_epi16(a, 0, a, 0b00_01_01_11);
        assert_eq_m128i(r, a);
        let r = _mm_mask_shufflehi_epi16(a, 0b11111111, a, 0b00_01_01_11);
        let e = _mm_set_epi16(3, 2, 2, 0, 4, 5, 6, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_shufflehi_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_maskz_shufflehi_epi16(0, a, 0b00_01_01_11);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_shufflehi_epi16(0b11111111, a, 0b00_01_01_11);
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
        _store_mask64(&mut r as *mut _ as *mut u64, a);
        assert_eq!(r, a);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_store_mask32() {
        let a: __mmask32 = 0b11111111_00000000_11111111_00000000;
        let mut r = 0;
        _store_mask32(&mut r as *mut _ as *mut u32, a);
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
        let r = _mm512_dbsad_epu8(a, b, 0);
        let e = _mm512_set1_epi16(8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_dbsad_epu8() {
        let src = _mm512_set1_epi16(1);
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(4);
        let r = _mm512_mask_dbsad_epu8(src, 0, a, b, 0);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_dbsad_epu8(src, 0b11111111_11111111_11111111_11111111, a, b, 0);
        let e = _mm512_set1_epi16(8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_dbsad_epu8() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(4);
        let r = _mm512_maskz_dbsad_epu8(0, a, b, 0);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_dbsad_epu8(0b11111111_11111111_11111111_11111111, a, b, 0);
        let e = _mm512_set1_epi16(8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_dbsad_epu8() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_dbsad_epu8(a, b, 0);
        let e = _mm256_set1_epi16(8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_mask_dbsad_epu8() {
        let src = _mm256_set1_epi16(1);
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_mask_dbsad_epu8(src, 0, a, b, 0);
        assert_eq_m256i(r, src);
        let r = _mm256_mask_dbsad_epu8(src, 0b11111111_11111111, a, b, 0);
        let e = _mm256_set1_epi16(8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm256_maskz_dbsad_epu8() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_maskz_dbsad_epu8(0, a, b, 0);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_dbsad_epu8(0b11111111_11111111, a, b, 0);
        let e = _mm256_set1_epi16(8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_dbsad_epu8() {
        let a = _mm_set1_epi8(2);
        let b = _mm_set1_epi8(4);
        let r = _mm_dbsad_epu8(a, b, 0);
        let e = _mm_set1_epi16(8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_mask_dbsad_epu8() {
        let src = _mm_set1_epi16(1);
        let a = _mm_set1_epi8(2);
        let b = _mm_set1_epi8(4);
        let r = _mm_mask_dbsad_epu8(src, 0, a, b, 0);
        assert_eq_m128i(r, src);
        let r = _mm_mask_dbsad_epu8(src, 0b11111111, a, b, 0);
        let e = _mm_set1_epi16(8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_dbsad_epu8() {
        let a = _mm_set1_epi8(2);
        let b = _mm_set1_epi8(4);
        let r = _mm_maskz_dbsad_epu8(0, a, b, 0);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_dbsad_epu8(0b11111111, a, b, 0);
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
        let r = _mm512_bslli_epi128(a, 9);
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
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        );
        let r = _mm512_bsrli_epi128(a, 9);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
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
        let r = _mm512_alignr_epi8(a, b, 14);
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
        let r = _mm512_mask_alignr_epi8(a, 0, a, b, 14);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_alignr_epi8(
            a,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
            b,
            14,
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
        let r = _mm512_maskz_alignr_epi8(0, a, b, 14);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_alignr_epi8(
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
            b,
            14,
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
        let r = _mm256_mask_alignr_epi8(a, 0, a, b, 14);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_alignr_epi8(a, 0b11111111_11111111_11111111_11111111, a, b, 14);
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
        let r = _mm256_maskz_alignr_epi8(0, a, b, 14);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_alignr_epi8(0b11111111_11111111_11111111_11111111, a, b, 14);
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
        let r = _mm_mask_alignr_epi8(a, 0, a, b, 14);
        assert_eq_m128i(r, a);
        let r = _mm_mask_alignr_epi8(a, 0b11111111_11111111, a, b, 14);
        let e = _mm_set_epi8(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512bw,avx512vl")]
    unsafe fn test_mm_maskz_alignr_epi8() {
        let a = _mm_set_epi8(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0);
        let b = _mm_set1_epi8(1);
        let r = _mm_maskz_alignr_epi8(0, a, b, 14);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_alignr_epi8(0b11111111_11111111, a, b, 14);
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
