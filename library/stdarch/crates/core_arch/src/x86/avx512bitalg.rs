//! Bit-oriented Algorithms (BITALG)
//!
//! The intrinsics here correspond to those in the `immintrin.h` C header.
//!
//! The reference is [Intel 64 and IA-32 Architectures Software Developer's
//! Manual Volume 2: Instruction Set Reference, A-Z][intel64_ref].
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf

use crate::core_arch::simd::i8x16;
use crate::core_arch::simd::i8x32;
use crate::core_arch::simd::i8x64;
use crate::core_arch::simd::i16x8;
use crate::core_arch::simd::i16x16;
use crate::core_arch::simd::i16x32;
use crate::core_arch::x86::__m128i;
use crate::core_arch::x86::__m256i;
use crate::core_arch::x86::__m512i;
use crate::core_arch::x86::__mmask8;
use crate::core_arch::x86::__mmask16;
use crate::core_arch::x86::__mmask32;
use crate::core_arch::x86::__mmask64;
use crate::intrinsics::simd::{simd_ctpop, simd_select_bitmask};
use crate::mem::transmute;

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.x86.avx512.mask.vpshufbitqmb.512"]
    fn bitshuffle_512(data: i8x64, indices: i8x64, mask: __mmask64) -> __mmask64;
    #[link_name = "llvm.x86.avx512.mask.vpshufbitqmb.256"]
    fn bitshuffle_256(data: i8x32, indices: i8x32, mask: __mmask32) -> __mmask32;
    #[link_name = "llvm.x86.avx512.mask.vpshufbitqmb.128"]
    fn bitshuffle_128(data: i8x16, indices: i8x16, mask: __mmask16) -> __mmask16;
}

/// For each packed 16-bit integer maps the value to the number of logical 1 bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_popcnt_epi16)
#[inline]
#[target_feature(enable = "avx512bitalg")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntw))]
pub fn _mm512_popcnt_epi16(a: __m512i) -> __m512i {
    unsafe { transmute(simd_ctpop(a.as_i16x32())) }
}

/// For each packed 16-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_popcnt_epi16)
#[inline]
#[target_feature(enable = "avx512bitalg")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntw))]
pub fn _mm512_maskz_popcnt_epi16(k: __mmask32, a: __m512i) -> __m512i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i16x32()),
            i16x32::ZERO,
        ))
    }
}

/// For each packed 16-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_popcnt_epi16)
#[inline]
#[target_feature(enable = "avx512bitalg")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntw))]
pub fn _mm512_mask_popcnt_epi16(src: __m512i, k: __mmask32, a: __m512i) -> __m512i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i16x32()),
            src.as_i16x32(),
        ))
    }
}

/// For each packed 16-bit integer maps the value to the number of logical 1 bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_popcnt_epi16)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntw))]
pub fn _mm256_popcnt_epi16(a: __m256i) -> __m256i {
    unsafe { transmute(simd_ctpop(a.as_i16x16())) }
}

/// For each packed 16-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_popcnt_epi16)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntw))]
pub fn _mm256_maskz_popcnt_epi16(k: __mmask16, a: __m256i) -> __m256i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i16x16()),
            i16x16::ZERO,
        ))
    }
}

/// For each packed 16-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_popcnt_epi16)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntw))]
pub fn _mm256_mask_popcnt_epi16(src: __m256i, k: __mmask16, a: __m256i) -> __m256i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i16x16()),
            src.as_i16x16(),
        ))
    }
}

/// For each packed 16-bit integer maps the value to the number of logical 1 bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_popcnt_epi16)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntw))]
pub fn _mm_popcnt_epi16(a: __m128i) -> __m128i {
    unsafe { transmute(simd_ctpop(a.as_i16x8())) }
}

/// For each packed 16-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_popcnt_epi16)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntw))]
pub fn _mm_maskz_popcnt_epi16(k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i16x8()),
            i16x8::ZERO,
        ))
    }
}

/// For each packed 16-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_popcnt_epi16)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntw))]
pub fn _mm_mask_popcnt_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i16x8()),
            src.as_i16x8(),
        ))
    }
}

/// For each packed 8-bit integer maps the value to the number of logical 1 bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_popcnt_epi8)
#[inline]
#[target_feature(enable = "avx512bitalg")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntb))]
pub fn _mm512_popcnt_epi8(a: __m512i) -> __m512i {
    unsafe { transmute(simd_ctpop(a.as_i8x64())) }
}

/// For each packed 8-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_popcnt_epi8)
#[inline]
#[target_feature(enable = "avx512bitalg")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntb))]
pub fn _mm512_maskz_popcnt_epi8(k: __mmask64, a: __m512i) -> __m512i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i8x64()),
            i8x64::ZERO,
        ))
    }
}

/// For each packed 8-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_popcnt_epi8)
#[inline]
#[target_feature(enable = "avx512bitalg")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntb))]
pub fn _mm512_mask_popcnt_epi8(src: __m512i, k: __mmask64, a: __m512i) -> __m512i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i8x64()),
            src.as_i8x64(),
        ))
    }
}

/// For each packed 8-bit integer maps the value to the number of logical 1 bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_popcnt_epi8)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntb))]
pub fn _mm256_popcnt_epi8(a: __m256i) -> __m256i {
    unsafe { transmute(simd_ctpop(a.as_i8x32())) }
}

/// For each packed 8-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_popcnt_epi8)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntb))]
pub fn _mm256_maskz_popcnt_epi8(k: __mmask32, a: __m256i) -> __m256i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i8x32()),
            i8x32::ZERO,
        ))
    }
}

/// For each packed 8-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_popcnt_epi8)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntb))]
pub fn _mm256_mask_popcnt_epi8(src: __m256i, k: __mmask32, a: __m256i) -> __m256i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i8x32()),
            src.as_i8x32(),
        ))
    }
}

/// For each packed 8-bit integer maps the value to the number of logical 1 bits.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_popcnt_epi8)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntb))]
pub fn _mm_popcnt_epi8(a: __m128i) -> __m128i {
    unsafe { transmute(simd_ctpop(a.as_i8x16())) }
}

/// For each packed 8-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_popcnt_epi8)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntb))]
pub fn _mm_maskz_popcnt_epi8(k: __mmask16, a: __m128i) -> __m128i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i8x16()),
            i8x16::ZERO,
        ))
    }
}

/// For each packed 8-bit integer maps the value to the number of logical 1 bits.
///
/// Uses the writemask in k - elements are copied from src if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_popcnt_epi8)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpopcntb))]
pub fn _mm_mask_popcnt_epi8(src: __m128i, k: __mmask16, a: __m128i) -> __m128i {
    unsafe {
        transmute(simd_select_bitmask(
            k,
            simd_ctpop(a.as_i8x16()),
            src.as_i8x16(),
        ))
    }
}

/// Considers the input `b` as packed 64-bit integers and `c` as packed 8-bit integers.
/// Then groups 8 8-bit values from `c`as indices into the bits of the corresponding 64-bit integer.
/// It then selects these bits and packs them into the output.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_bitshuffle_epi64_mask)
#[inline]
#[target_feature(enable = "avx512bitalg")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufbitqmb))]
pub fn _mm512_bitshuffle_epi64_mask(b: __m512i, c: __m512i) -> __mmask64 {
    unsafe { bitshuffle_512(b.as_i8x64(), c.as_i8x64(), !0) }
}

/// Considers the input `b` as packed 64-bit integers and `c` as packed 8-bit integers.
/// Then groups 8 8-bit values from `c`as indices into the bits of the corresponding 64-bit integer.
/// It then selects these bits and packs them into the output.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_bitshuffle_epi64_mask)
#[inline]
#[target_feature(enable = "avx512bitalg")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufbitqmb))]
pub fn _mm512_mask_bitshuffle_epi64_mask(k: __mmask64, b: __m512i, c: __m512i) -> __mmask64 {
    unsafe { bitshuffle_512(b.as_i8x64(), c.as_i8x64(), k) }
}

/// Considers the input `b` as packed 64-bit integers and `c` as packed 8-bit integers.
/// Then groups 8 8-bit values from `c`as indices into the bits of the corresponding 64-bit integer.
/// It then selects these bits and packs them into the output.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_bitshuffle_epi64_mask)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufbitqmb))]
pub fn _mm256_bitshuffle_epi64_mask(b: __m256i, c: __m256i) -> __mmask32 {
    unsafe { bitshuffle_256(b.as_i8x32(), c.as_i8x32(), !0) }
}

/// Considers the input `b` as packed 64-bit integers and `c` as packed 8-bit integers.
/// Then groups 8 8-bit values from `c`as indices into the bits of the corresponding 64-bit integer.
/// It then selects these bits and packs them into the output.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_bitshuffle_epi64_mask)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufbitqmb))]
pub fn _mm256_mask_bitshuffle_epi64_mask(k: __mmask32, b: __m256i, c: __m256i) -> __mmask32 {
    unsafe { bitshuffle_256(b.as_i8x32(), c.as_i8x32(), k) }
}

/// Considers the input `b` as packed 64-bit integers and `c` as packed 8-bit integers.
/// Then groups 8 8-bit values from `c`as indices into the bits of the corresponding 64-bit integer.
/// It then selects these bits and packs them into the output.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_bitshuffle_epi64_mask)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufbitqmb))]
pub fn _mm_bitshuffle_epi64_mask(b: __m128i, c: __m128i) -> __mmask16 {
    unsafe { bitshuffle_128(b.as_i8x16(), c.as_i8x16(), !0) }
}

/// Considers the input `b` as packed 64-bit integers and `c` as packed 8-bit integers.
/// Then groups 8 8-bit values from `c`as indices into the bits of the corresponding 64-bit integer.
/// It then selects these bits and packs them into the output.
///
/// Uses the writemask in k - elements are zeroed in the result if the corresponding mask bit is not set.
/// Otherwise the computation result is written into the result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_bitshuffle_epi64_mask)
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpshufbitqmb))]
pub fn _mm_mask_bitshuffle_epi64_mask(k: __mmask16, b: __m128i, c: __m128i) -> __mmask16 {
    unsafe { bitshuffle_128(b.as_i8x16(), c.as_i8x16(), k) }
}

#[cfg(test)]
mod tests {
    // Some of the constants in the tests below are just bit patterns. They should not
    // be interpreted as integers; signedness does not make sense for them, but
    // __mXXXi happens to be defined in terms of signed integers.
    #![allow(overflowing_literals)]

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "avx512bitalg,avx512f")]
    unsafe fn test_mm512_popcnt_epi16() {
        let test_data = _mm512_set_epi16(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, 0x1_FF, 0x3_FF, 0x7_FF, 0xF_FF, 0x1F_FF,
            0x3F_FF, 0x7F_FF, 0xFF_FF, -1, -100, 255, 256, 2, 4, 8, 16, 32, 64, 128, 256, 512,
            1024, 2048,
        );
        let actual_result = _mm512_popcnt_epi16(test_data);
        let reference_result = _mm512_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 12, 8, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
        );
        assert_eq_m512i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f")]
    unsafe fn test_mm512_maskz_popcnt_epi16() {
        let test_data = _mm512_set_epi16(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, 0x1_FF, 0x3_FF, 0x7_FF, 0xF_FF, 0x1F_FF,
            0x3F_FF, 0x7F_FF, 0xFF_FF, -1, -100, 255, 256, 2, 4, 8, 16, 32, 64, 128, 256, 512,
            1024, 2048,
        );
        let mask = 0xFF_FF_00_00;
        let actual_result = _mm512_maskz_popcnt_epi16(mask, test_data);
        let reference_result = _mm512_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        );
        assert_eq_m512i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f")]
    unsafe fn test_mm512_mask_popcnt_epi16() {
        let test_data = _mm512_set_epi16(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, 0x1_FF, 0x3_FF, 0x7_FF, 0xF_FF, 0x1F_FF,
            0x3F_FF, 0x7F_FF, 0xFF_FF, -1, -100, 255, 256, 2, 4, 8, 16, 32, 64, 128, 256, 512,
            1024, 2048,
        );
        let mask = 0xFF_FF_00_00;
        let actual_result = _mm512_mask_popcnt_epi16(test_data, mask, test_data);
        let reference_result = _mm512_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0xFF_FF, -1, -100, 255, 256, 2,
            4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
        );
        assert_eq_m512i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm256_popcnt_epi16() {
        let test_data = _mm256_set_epi16(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, 0x1_FF, 0x3_FF, 0x7_FF, 0xF_FF, 0x1F_FF,
            0x3F_FF, 0x7F_FF,
        );
        let actual_result = _mm256_popcnt_epi16(test_data);
        let reference_result =
            _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m256i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm256_maskz_popcnt_epi16() {
        let test_data = _mm256_set_epi16(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, 0x1_FF, 0x3_FF, 0x7_FF, 0xF_FF, 0x1F_FF,
            0x3F_FF, 0x7F_FF,
        );
        let mask = 0xFF_00;
        let actual_result = _mm256_maskz_popcnt_epi16(mask, test_data);
        let reference_result = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m256i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm256_mask_popcnt_epi16() {
        let test_data = _mm256_set_epi16(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, 0x1_FF, 0x3_FF, 0x7_FF, 0xF_FF, 0x1F_FF,
            0x3F_FF, 0x7F_FF,
        );
        let mask = 0xFF_00;
        let actual_result = _mm256_mask_popcnt_epi16(test_data, mask, test_data);
        let reference_result = _mm256_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 0xFF, 0x1_FF, 0x3_FF, 0x7_FF, 0xF_FF, 0x1F_FF, 0x3F_FF, 0x7F_FF,
        );
        assert_eq_m256i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm_popcnt_epi16() {
        let test_data = _mm_set_epi16(0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F);
        let actual_result = _mm_popcnt_epi16(test_data);
        let reference_result = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq_m128i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm_maskz_popcnt_epi16() {
        let test_data = _mm_set_epi16(0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F);
        let mask = 0xF0;
        let actual_result = _mm_maskz_popcnt_epi16(mask, test_data);
        let reference_result = _mm_set_epi16(0, 1, 2, 3, 0, 0, 0, 0);
        assert_eq_m128i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm_mask_popcnt_epi16() {
        let test_data = _mm_set_epi16(0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F);
        let mask = 0xF0;
        let actual_result = _mm_mask_popcnt_epi16(test_data, mask, test_data);
        let reference_result = _mm_set_epi16(0, 1, 2, 3, 0xF, 0x1F, 0x3F, 0x7F);
        assert_eq_m128i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f")]
    unsafe fn test_mm512_popcnt_epi8() {
        let test_data = _mm512_set_epi8(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, -1, 2, 4, 8, 16, 32, 64, 128, 171, 206, 100,
            217, 109, 253, 190, 177, 254, 179, 215, 230, 68, 201, 172, 183, 154, 84, 56, 227, 189,
            140, 35, 117, 219, 169, 226, 170, 13, 22, 159, 251, 73, 121, 143, 145, 85, 91, 137, 90,
            225, 21, 249, 211, 155, 228, 70,
        );
        let actual_result = _mm512_popcnt_epi8(test_data);
        let reference_result = _mm512_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 1, 1, 1, 1, 1, 1, 1, 5, 5, 3, 5, 5, 7, 6, 4, 7, 5, 6, 5,
            2, 4, 4, 6, 4, 3, 3, 5, 6, 3, 3, 5, 6, 4, 4, 4, 3, 3, 6, 7, 3, 5, 5, 3, 4, 5, 3, 4, 4,
            3, 6, 5, 5, 4, 3,
        );
        assert_eq_m512i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f")]
    unsafe fn test_mm512_maskz_popcnt_epi8() {
        let test_data = _mm512_set_epi8(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, -1, 2, 4, 8, 16, 32, 64, 128, 171, 206, 100,
            217, 109, 253, 190, 177, 254, 179, 215, 230, 68, 201, 172, 183, 154, 84, 56, 227, 189,
            140, 35, 117, 219, 169, 226, 170, 13, 22, 159, 251, 73, 121, 143, 145, 85, 91, 137, 90,
            225, 21, 249, 211, 155, 228, 70,
        );
        let mask = 0xFF_FF_FF_FF_00_00_00_00;
        let actual_result = _mm512_maskz_popcnt_epi8(mask, test_data);
        let reference_result = _mm512_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 1, 1, 1, 1, 1, 1, 1, 5, 5, 3, 5, 5, 7, 6, 4, 7, 5, 6, 5,
            2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        );
        assert_eq_m512i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f")]
    unsafe fn test_mm512_mask_popcnt_epi8() {
        let test_data = _mm512_set_epi8(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, -1, 2, 4, 8, 16, 32, 64, 128, 171, 206, 100,
            217, 109, 253, 190, 177, 254, 179, 215, 230, 68, 201, 172, 183, 154, 84, 56, 227, 189,
            140, 35, 117, 219, 169, 226, 170, 13, 22, 159, 251, 73, 121, 143, 145, 85, 91, 137, 90,
            225, 21, 249, 211, 155, 228, 70,
        );
        let mask = 0xFF_FF_FF_FF_00_00_00_00;
        let actual_result = _mm512_mask_popcnt_epi8(test_data, mask, test_data);
        let reference_result = _mm512_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 1, 1, 1, 1, 1, 1, 1, 5, 5, 3, 5, 5, 7, 6, 4, 7, 5, 6, 5,
            2, 4, 4, 183, 154, 84, 56, 227, 189, 140, 35, 117, 219, 169, 226, 170, 13, 22, 159,
            251, 73, 121, 143, 145, 85, 91, 137, 90, 225, 21, 249, 211, 155, 228, 70,
        );
        assert_eq_m512i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm256_popcnt_epi8() {
        let test_data = _mm256_set_epi8(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, -1, 2, 4, 8, 16, 32, 64, 128, 171, 206, 100,
            217, 109, 253, 190, 177, 254, 179, 215, 230, 68, 201, 172,
        );
        let actual_result = _mm256_popcnt_epi8(test_data);
        let reference_result = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 1, 1, 1, 1, 1, 1, 1, 5, 5, 3, 5, 5, 7, 6, 4, 7, 5, 6, 5,
            2, 4, 4,
        );
        assert_eq_m256i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm256_maskz_popcnt_epi8() {
        let test_data = _mm256_set_epi8(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, -1, 2, 4, 8, 16, 32, 64, 251, 73, 121, 143,
            145, 85, 91, 137, 90, 225, 21, 249, 211, 155, 228, 70,
        );
        let mask = 0xFF_FF_00_00;
        let actual_result = _mm256_maskz_popcnt_epi8(mask, test_data);
        let reference_result = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        );
        assert_eq_m256i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm256_mask_popcnt_epi8() {
        let test_data = _mm256_set_epi8(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, -1, 2, 4, 8, 16, 32, 64, 251, 73, 121, 143,
            145, 85, 91, 137, 90, 225, 21, 249, 211, 155, 228, 70,
        );
        let mask = 0xFF_FF_00_00;
        let actual_result = _mm256_mask_popcnt_epi8(test_data, mask, test_data);
        let reference_result = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 1, 1, 1, 1, 1, 1, 251, 73, 121, 143, 145, 85, 91, 137,
            90, 225, 21, 249, 211, 155, 228, 70,
        );
        assert_eq_m256i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm_popcnt_epi8() {
        let test_data = _mm_set_epi8(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, -1, 2, 4, 8, 16, 32, 64,
        );
        let actual_result = _mm_popcnt_epi8(test_data);
        let reference_result = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 1, 1, 1, 1, 1, 1);
        assert_eq_m128i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm_maskz_popcnt_epi8() {
        let test_data = _mm_set_epi8(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 90, 225, 21, 249, 211, 155, 228, 70,
        );
        let mask = 0xFF_00;
        let actual_result = _mm_maskz_popcnt_epi8(mask, test_data);
        let reference_result = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm_mask_popcnt_epi8() {
        let test_data = _mm_set_epi8(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 90, 225, 21, 249, 211, 155, 228, 70,
        );
        let mask = 0xFF_00;
        let actual_result = _mm_mask_popcnt_epi8(test_data, mask, test_data);
        let reference_result =
            _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 90, 225, 21, 249, 211, 155, 228, 70);
        assert_eq_m128i(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f")]
    unsafe fn test_mm512_bitshuffle_epi64_mask() {
        let test_indices = _mm512_set_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 63, 62, 61, 60, 59, 58, 57, 56, 32, 32, 16, 16, 0, 0,
            8, 8, 56, 48, 40, 32, 24, 16, 8, 0, 63, 62, 61, 60, 59, 58, 57, 56, 63, 62, 61, 60, 59,
            58, 57, 56, 32, 32, 16, 16, 0, 0, 8, 8, 56, 48, 40, 32, 24, 16, 8, 0,
        );
        let test_data = _mm512_setr_epi64(
            0xFF_FF_FF_FF_00_00_00_00,
            0xFF_00_FF_00_FF_00_FF_00,
            0xFF_00_00_00_00_00_00_00,
            0xAC_00_00_00_00_00_00_00,
            0xFF_FF_FF_FF_00_00_00_00,
            0xFF_00_FF_00_FF_00_FF_00,
            0xFF_00_00_00_00_00_00_00,
            0xAC_00_00_00_00_00_00_00,
        );
        let actual_result = _mm512_bitshuffle_epi64_mask(test_data, test_indices);
        let reference_result = 0xF0 << 0
            | 0x03 << 8
            | 0xFF << 16
            | 0xAC << 24
            | 0xF0 << 32
            | 0x03 << 40
            | 0xFF << 48
            | 0xAC << 56;

        assert_eq!(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f")]
    unsafe fn test_mm512_mask_bitshuffle_epi64_mask() {
        let test_indices = _mm512_set_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 63, 62, 61, 60, 59, 58, 57, 56, 32, 32, 16, 16, 0, 0,
            8, 8, 56, 48, 40, 32, 24, 16, 8, 0, 63, 62, 61, 60, 59, 58, 57, 56, 63, 62, 61, 60, 59,
            58, 57, 56, 32, 32, 16, 16, 0, 0, 8, 8, 56, 48, 40, 32, 24, 16, 8, 0,
        );
        let test_data = _mm512_setr_epi64(
            0xFF_FF_FF_FF_00_00_00_00,
            0xFF_00_FF_00_FF_00_FF_00,
            0xFF_00_00_00_00_00_00_00,
            0xAC_00_00_00_00_00_00_00,
            0xFF_FF_FF_FF_00_00_00_00,
            0xFF_00_FF_00_FF_00_FF_00,
            0xFF_00_00_00_00_00_00_00,
            0xAC_00_00_00_00_00_00_00,
        );
        let mask = 0xFF_FF_FF_FF_00_00_00_00;
        let actual_result = _mm512_mask_bitshuffle_epi64_mask(mask, test_data, test_indices);
        let reference_result = 0x00 << 0
            | 0x00 << 8
            | 0x00 << 16
            | 0x00 << 24
            | 0xF0 << 32
            | 0x03 << 40
            | 0xFF << 48
            | 0xAC << 56;

        assert_eq!(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm256_bitshuffle_epi64_mask() {
        let test_indices = _mm256_set_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 63, 62, 61, 60, 59, 58, 57, 56, 32, 32, 16, 16, 0, 0,
            8, 8, 56, 48, 40, 32, 24, 16, 8, 0,
        );
        let test_data = _mm256_setr_epi64x(
            0xFF_FF_FF_FF_00_00_00_00,
            0xFF_00_FF_00_FF_00_FF_00,
            0xFF_00_00_00_00_00_00_00,
            0xAC_00_00_00_00_00_00_00,
        );
        let actual_result = _mm256_bitshuffle_epi64_mask(test_data, test_indices);
        let reference_result = 0xF0 << 0 | 0x03 << 8 | 0xFF << 16 | 0xAC << 24;

        assert_eq!(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm256_mask_bitshuffle_epi64_mask() {
        let test_indices = _mm256_set_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 63, 62, 61, 60, 59, 58, 57, 56, 32, 32, 16, 16, 0, 0,
            8, 8, 56, 48, 40, 32, 24, 16, 8, 0,
        );
        let test_data = _mm256_setr_epi64x(
            0xFF_FF_FF_FF_00_00_00_00,
            0xFF_00_FF_00_FF_00_FF_00,
            0xFF_00_00_00_00_00_00_00,
            0xAC_00_00_00_00_00_00_00,
        );
        let mask = 0xFF_FF_00_00;
        let actual_result = _mm256_mask_bitshuffle_epi64_mask(mask, test_data, test_indices);
        let reference_result = 0x00 << 0 | 0x00 << 8 | 0xFF << 16 | 0xAC << 24;

        assert_eq!(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm_bitshuffle_epi64_mask() {
        let test_indices = _mm_set_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 63, 62, 61, 60, 59, 58, 57, 56,
        );
        let test_data = _mm_setr_epi64x(0xFF_00_00_00_00_00_00_00, 0xAC_00_00_00_00_00_00_00);
        let actual_result = _mm_bitshuffle_epi64_mask(test_data, test_indices);
        let reference_result = 0xFF << 0 | 0xAC << 8;

        assert_eq!(actual_result, reference_result);
    }

    #[simd_test(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm_mask_bitshuffle_epi64_mask() {
        let test_indices = _mm_set_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 63, 62, 61, 60, 59, 58, 57, 56,
        );
        let test_data = _mm_setr_epi64x(0xFF_00_00_00_00_00_00_00, 0xAC_00_00_00_00_00_00_00);
        let mask = 0xFF_00;
        let actual_result = _mm_mask_bitshuffle_epi64_mask(mask, test_data, test_indices);
        let reference_result = 0x00 << 0 | 0xAC << 8;

        assert_eq!(actual_result, reference_result);
    }
}
