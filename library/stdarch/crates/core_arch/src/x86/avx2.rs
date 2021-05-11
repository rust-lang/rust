//! Advanced Vector Extensions 2 (AVX)
//!
//! AVX2 expands most AVX commands to 256-bit wide vector registers and
//! adds [FMA](https://en.wikipedia.org/wiki/Fused_multiply-accumulate).
//!
//! The references are:
//!
//! - [Intel 64 and IA-32 Architectures Software Developer's Manual Volume 2:
//!   Instruction Set Reference, A-Z][intel64_ref].
//! - [AMD64 Architecture Programmer's Manual, Volume 3: General-Purpose and
//!   System Instructions][amd64_ref].
//!
//! Wikipedia's [AVX][wiki_avx] and [FMA][wiki_fma] pages provide a quick
//! overview of the instructions available.
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
//! [amd64_ref]: http://support.amd.com/TechDocs/24594.pdf
//! [wiki_avx]: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions
//! [wiki_fma]: https://en.wikipedia.org/wiki/Fused_multiply-accumulate

use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem::transmute,
};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Computes the absolute values of packed 32-bit integers in `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_abs_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpabsd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_abs_epi32(a: __m256i) -> __m256i {
    transmute(pabsd(a.as_i32x8()))
}

/// Computes the absolute values of packed 16-bit integers in `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_abs_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpabsw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_abs_epi16(a: __m256i) -> __m256i {
    transmute(pabsw(a.as_i16x16()))
}

/// Computes the absolute values of packed 8-bit integers in `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_abs_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpabsb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_abs_epi8(a: __m256i) -> __m256i {
    transmute(pabsb(a.as_i8x32()))
}

/// Adds packed 64-bit integers in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_add_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpaddq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_add_epi64(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_add(a.as_i64x4(), b.as_i64x4()))
}

/// Adds packed 32-bit integers in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_add_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpaddd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_add_epi32(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_add(a.as_i32x8(), b.as_i32x8()))
}

/// Adds packed 16-bit integers in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_add_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpaddw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_add_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_add(a.as_i16x16(), b.as_i16x16()))
}

/// Adds packed 8-bit integers in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_add_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpaddb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_add_epi8(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_add(a.as_i8x32(), b.as_i8x32()))
}

/// Adds packed 8-bit integers in `a` and `b` using saturation.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_adds_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpaddsb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_adds_epi8(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_saturating_add(a.as_i8x32(), b.as_i8x32()))
}

/// Adds packed 16-bit integers in `a` and `b` using saturation.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_adds_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpaddsw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_adds_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_saturating_add(a.as_i16x16(), b.as_i16x16()))
}

/// Adds packed unsigned 8-bit integers in `a` and `b` using saturation.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_adds_epu8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpaddusb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_adds_epu8(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_saturating_add(a.as_u8x32(), b.as_u8x32()))
}

/// Adds packed unsigned 16-bit integers in `a` and `b` using saturation.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_adds_epu16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpaddusw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_adds_epu16(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_saturating_add(a.as_u16x16(), b.as_u16x16()))
}

/// Concatenates pairs of 16-byte blocks in `a` and `b` into a 32-byte temporary
/// result, shifts the result right by `n` bytes, and returns the low 16 bytes.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_alignr_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpalignr, IMM8 = 7))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_alignr_epi8<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    // If palignr is shifting the pair of vectors more than the size of two
    // lanes, emit zero.
    if IMM8 > 32 {
        return _mm256_set1_epi8(0);
    }
    // If palignr is shifting the pair of input vectors more than one lane,
    // but less than two lanes, convert to shifting in zeroes.
    let (a, b) = if IMM8 > 16 {
        (_mm256_set1_epi8(0), a)
    } else {
        (a, b)
    };

    let a = a.as_i8x32();
    let b = b.as_i8x32();

    let r: i8x32 = match IMM8 % 16 {
        0 => simd_shuffle32!(
            b,
            a,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31,
            ],
        ),
        1 => simd_shuffle32!(
            b,
            a,
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 48,
            ],
        ),
        2 => simd_shuffle32!(
            b,
            a,
            [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 28, 29, 30, 31, 48, 49,
            ],
        ),
        3 => simd_shuffle32!(
            b,
            a,
            [
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 28, 29, 30, 31, 48, 49, 50,
            ],
        ),
        4 => simd_shuffle32!(
            b,
            a,
            [
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 20, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30, 31, 48, 49, 50, 51,
            ],
        ),
        5 => simd_shuffle32!(
            b,
            a,
            [
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 21, 22, 23, 24, 25, 26,
                27, 28, 29, 30, 31, 48, 49, 50, 51, 52,
            ],
        ),
        6 => simd_shuffle32!(
            b,
            a,
            [
                6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 48, 49, 50, 51, 52, 53,
            ],
        ),
        7 => simd_shuffle32!(
            b,
            a,
            [
                7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54,
            ],
        ),
        8 => simd_shuffle32!(
            b,
            a,
            [
                8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27, 28,
                29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55,
            ],
        ),
        9 => simd_shuffle32!(
            b,
            a,
            [
                9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 25, 26, 27, 28, 29,
                30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56,
            ],
        ),
        10 => simd_shuffle32!(
            b,
            a,
            [
                10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 26, 27, 28, 29, 30,
                31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            ],
        ),
        11 => simd_shuffle32!(
            b,
            a,
            [
                11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 27, 28, 29, 30, 31,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            ],
        ),
        12 => simd_shuffle32!(
            b,
            a,
            [
                12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 28, 29, 30, 31, 48,
                49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            ],
        ),
        13 => simd_shuffle32!(
            b,
            a,
            [
                13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 29, 30, 31, 48, 49,
                50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            ],
        ),
        14 => simd_shuffle32!(
            b,
            a,
            [
                14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 30, 31, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            ],
        ),
        15 => simd_shuffle32!(
            b,
            a,
            [
                15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 31, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
            ],
        ),
        _ => b,
    };
    transmute(r)
}

/// Computes the bitwise AND of 256 bits (representing integer data)
/// in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_and_si256)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vandps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_and_si256(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_and(a.as_i64x4(), b.as_i64x4()))
}

/// Computes the bitwise NOT of 256 bits (representing integer data)
/// in `a` and then AND with `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_andnot_si256)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vandnps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_andnot_si256(a: __m256i, b: __m256i) -> __m256i {
    let all_ones = _mm256_set1_epi8(-1);
    transmute(simd_and(
        simd_xor(a.as_i64x4(), all_ones.as_i64x4()),
        b.as_i64x4(),
    ))
}

/// Averages packed unsigned 16-bit integers in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_avg_epu16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpavgw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_avg_epu16(a: __m256i, b: __m256i) -> __m256i {
    transmute(pavgw(a.as_u16x16(), b.as_u16x16()))
}

/// Averages packed unsigned 8-bit integers in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_avg_epu8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpavgb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_avg_epu8(a: __m256i, b: __m256i) -> __m256i {
    transmute(pavgb(a.as_u8x32(), b.as_u8x32()))
}

/// Blends packed 32-bit integers from `a` and `b` using control mask `IMM4`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_blend_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vblendps, IMM4 = 9))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_blend_epi32<const IMM4: i32>(a: __m128i, b: __m128i) -> __m128i {
    static_assert_imm4!(IMM4);
    let a = a.as_i32x4();
    let b = b.as_i32x4();
    let r: i32x4 = simd_shuffle4!(
        a,
        b,
        <const IMM4: i32> [
            [0, 4, 0, 4][IMM4 as usize & 0b11],
            [1, 1, 5, 5][IMM4 as usize & 0b11],
            [2, 6, 2, 6][(IMM4 as usize >> 2) & 0b11],
            [3, 3, 7, 7][(IMM4 as usize >> 2) & 0b11],
        ],
    );
    transmute(r)
}

/// Blends packed 32-bit integers from `a` and `b` using control mask `IMM8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_blend_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vblendps, IMM8 = 9))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_blend_epi32<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    let a = a.as_i32x8();
    let b = b.as_i32x8();
    let r: i32x8 = simd_shuffle8!(
        a,
        b,
        <const IMM8: i32> [
            [0, 8, 0, 8][IMM8 as usize & 0b11],
            [1, 1, 9, 9][IMM8 as usize & 0b11],
            [2, 10, 2, 10][(IMM8 as usize >> 2) & 0b11],
            [3, 3, 11, 11][(IMM8 as usize >> 2) & 0b11],
            [4, 12, 4, 12][(IMM8 as usize >> 4) & 0b11],
            [5, 5, 13, 13][(IMM8 as usize >> 4) & 0b11],
            [6, 14, 6, 14][(IMM8 as usize >> 6) & 0b11],
            [7, 7, 15, 15][(IMM8 as usize >> 6) & 0b11],
        ],
    );
    transmute(r)
}

/// Blends packed 16-bit integers from `a` and `b` using control mask `IMM8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_blend_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpblendw, IMM8 = 9))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_blend_epi16<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    let a = a.as_i16x16();
    let b = b.as_i16x16();

    let r: i16x16 = simd_shuffle16!(
        a,
        b,
        <const IMM8: i32> [
            [0, 16, 0, 16][IMM8 as usize & 0b11],
            [1, 1, 17, 17][IMM8 as usize & 0b11],
            [2, 18, 2, 18][(IMM8 as usize >> 2) & 0b11],
            [3, 3, 19, 19][(IMM8 as usize >> 2) & 0b11],
            [4, 20, 4, 20][(IMM8 as usize >> 4) & 0b11],
            [5, 5, 21, 21][(IMM8 as usize >> 4) & 0b11],
            [6, 22, 6, 22][(IMM8 as usize >> 6) & 0b11],
            [7, 7, 23, 23][(IMM8 as usize >> 6) & 0b11],
            [8, 24, 8, 24][IMM8 as usize & 0b11],
            [9, 9, 25, 25][IMM8 as usize & 0b11],
            [10, 26, 10, 26][(IMM8 as usize >> 2) & 0b11],
            [11, 11, 27, 27][(IMM8 as usize >> 2) & 0b11],
            [12, 28, 12, 28][(IMM8 as usize >> 4) & 0b11],
            [13, 13, 29, 29][(IMM8 as usize >> 4) & 0b11],
            [14, 30, 14, 30][(IMM8 as usize >> 6) & 0b11],
            [15, 15, 31, 31][(IMM8 as usize >> 6) & 0b11],
        ],
    );
    transmute(r)
}

/// Blends packed 8-bit integers from `a` and `b` using `mask`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_blendv_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpblendvb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_blendv_epi8(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
    transmute(pblendvb(a.as_i8x32(), b.as_i8x32(), mask.as_i8x32()))
}

/// Broadcasts the low packed 8-bit integer from `a` to all elements of
/// the 128-bit returned value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_broadcastb_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_broadcastb_epi8(a: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let ret = simd_shuffle16!(a.as_i8x16(), zero.as_i8x16(), [0_u32; 16]);
    transmute::<i8x16, _>(ret)
}

/// Broadcasts the low packed 8-bit integer from `a` to all elements of
/// the 256-bit returned value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_broadcastb_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_broadcastb_epi8(a: __m128i) -> __m256i {
    let zero = _mm_setzero_si128();
    let ret = simd_shuffle32!(a.as_i8x16(), zero.as_i8x16(), [0_u32; 32]);
    transmute::<i8x32, _>(ret)
}

// N.B., `simd_shuffle4` with integer data types for `a` and `b` is
// often compiled to `vbroadcastss`.
/// Broadcasts the low packed 32-bit integer from `a` to all elements of
/// the 128-bit returned value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_broadcastd_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vbroadcastss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_broadcastd_epi32(a: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let ret = simd_shuffle4!(a.as_i32x4(), zero.as_i32x4(), [0_u32; 4]);
    transmute::<i32x4, _>(ret)
}

// N.B., `simd_shuffle4`` with integer data types for `a` and `b` is
// often compiled to `vbroadcastss`.
/// Broadcasts the low packed 32-bit integer from `a` to all elements of
/// the 256-bit returned value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_broadcastd_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vbroadcastss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_broadcastd_epi32(a: __m128i) -> __m256i {
    let zero = _mm_setzero_si128();
    let ret = simd_shuffle8!(a.as_i32x4(), zero.as_i32x4(), [0_u32; 8]);
    transmute::<i32x8, _>(ret)
}

/// Broadcasts the low packed 64-bit integer from `a` to all elements of
/// the 128-bit returned value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_broadcastq_epi64)
#[inline]
#[target_feature(enable = "avx2")]
// FIXME: https://github.com/rust-lang/stdarch/issues/791
#[cfg_attr(test, assert_instr(vmovddup))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_broadcastq_epi64(a: __m128i) -> __m128i {
    let ret = simd_shuffle2!(a.as_i64x2(), a.as_i64x2(), [0_u32; 2]);
    transmute::<i64x2, _>(ret)
}

/// Broadcasts the low packed 64-bit integer from `a` to all elements of
/// the 256-bit returned value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_broadcastq_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vbroadcastsd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_broadcastq_epi64(a: __m128i) -> __m256i {
    let ret = simd_shuffle4!(a.as_i64x2(), a.as_i64x2(), [0_u32; 4]);
    transmute::<i64x4, _>(ret)
}

/// Broadcasts the low double-precision (64-bit) floating-point element
/// from `a` to all elements of the 128-bit returned value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_broadcastsd_pd)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vmovddup))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_broadcastsd_pd(a: __m128d) -> __m128d {
    simd_shuffle2!(a, _mm_setzero_pd(), [0_u32; 2])
}

/// Broadcasts the low double-precision (64-bit) floating-point element
/// from `a` to all elements of the 256-bit returned value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_broadcastsd_pd)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vbroadcastsd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_broadcastsd_pd(a: __m128d) -> __m256d {
    simd_shuffle4!(a, _mm_setzero_pd(), [0_u32; 4])
}

// N.B., `broadcastsi128_si256` is often compiled to `vinsertf128` or
// `vbroadcastf128`.
/// Broadcasts 128 bits of integer data from a to all 128-bit lanes in
/// the 256-bit returned value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_broadcastsi128_si256)
#[inline]
#[target_feature(enable = "avx2")]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_broadcastsi128_si256(a: __m128i) -> __m256i {
    let zero = _mm_setzero_si128();
    let ret = simd_shuffle4!(a.as_i64x2(), zero.as_i64x2(), [0, 1, 0, 1]);
    transmute::<i64x4, _>(ret)
}

/// Broadcasts the low single-precision (32-bit) floating-point element
/// from `a` to all elements of the 128-bit returned value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_broadcastss_ps)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vbroadcastss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_broadcastss_ps(a: __m128) -> __m128 {
    simd_shuffle4!(a, _mm_setzero_ps(), [0_u32; 4])
}

/// Broadcasts the low single-precision (32-bit) floating-point element
/// from `a` to all elements of the 256-bit returned value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_broadcastss_ps)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vbroadcastss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_broadcastss_ps(a: __m128) -> __m256 {
    simd_shuffle8!(a, _mm_setzero_ps(), [0_u32; 8])
}

/// Broadcasts the low packed 16-bit integer from a to all elements of
/// the 128-bit returned value
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_broadcastw_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_broadcastw_epi16(a: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let ret = simd_shuffle8!(a.as_i16x8(), zero.as_i16x8(), [0_u32; 8]);
    transmute::<i16x8, _>(ret)
}

/// Broadcasts the low packed 16-bit integer from a to all elements of
/// the 256-bit returned value
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_broadcastw_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_broadcastw_epi16(a: __m128i) -> __m256i {
    let zero = _mm_setzero_si128();
    let ret = simd_shuffle16!(a.as_i16x8(), zero.as_i16x8(), [0_u32; 16]);
    transmute::<i16x16, _>(ret)
}

/// Compares packed 64-bit integers in `a` and `b` for equality.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpeq_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpcmpeqq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cmpeq_epi64(a: __m256i, b: __m256i) -> __m256i {
    transmute::<i64x4, _>(simd_eq(a.as_i64x4(), b.as_i64x4()))
}

/// Compares packed 32-bit integers in `a` and `b` for equality.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpeq_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpcmpeqd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cmpeq_epi32(a: __m256i, b: __m256i) -> __m256i {
    transmute::<i32x8, _>(simd_eq(a.as_i32x8(), b.as_i32x8()))
}

/// Compares packed 16-bit integers in `a` and `b` for equality.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpeq_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpcmpeqw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cmpeq_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute::<i16x16, _>(simd_eq(a.as_i16x16(), b.as_i16x16()))
}

/// Compares packed 8-bit integers in `a` and `b` for equality.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpeq_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpcmpeqb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cmpeq_epi8(a: __m256i, b: __m256i) -> __m256i {
    transmute::<i8x32, _>(simd_eq(a.as_i8x32(), b.as_i8x32()))
}

/// Compares packed 64-bit integers in `a` and `b` for greater-than.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpgt_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpcmpgtq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cmpgt_epi64(a: __m256i, b: __m256i) -> __m256i {
    transmute::<i64x4, _>(simd_gt(a.as_i64x4(), b.as_i64x4()))
}

/// Compares packed 32-bit integers in `a` and `b` for greater-than.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpgt_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpcmpgtd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cmpgt_epi32(a: __m256i, b: __m256i) -> __m256i {
    transmute::<i32x8, _>(simd_gt(a.as_i32x8(), b.as_i32x8()))
}

/// Compares packed 16-bit integers in `a` and `b` for greater-than.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpgt_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpcmpgtw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cmpgt_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute::<i16x16, _>(simd_gt(a.as_i16x16(), b.as_i16x16()))
}

/// Compares packed 8-bit integers in `a` and `b` for greater-than.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmpgt_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpcmpgtb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cmpgt_epi8(a: __m256i, b: __m256i) -> __m256i {
    transmute::<i8x32, _>(simd_gt(a.as_i8x32(), b.as_i8x32()))
}

/// Sign-extend 16-bit integers to 32-bit integers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepi16_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmovsxwd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtepi16_epi32(a: __m128i) -> __m256i {
    transmute::<i32x8, _>(simd_cast(a.as_i16x8()))
}

/// Sign-extend 16-bit integers to 64-bit integers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepi16_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmovsxwq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtepi16_epi64(a: __m128i) -> __m256i {
    let a = a.as_i16x8();
    let v64: i16x4 = simd_shuffle4!(a, a, [0, 1, 2, 3]);
    transmute::<i64x4, _>(simd_cast(v64))
}

/// Sign-extend 32-bit integers to 64-bit integers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepi32_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmovsxdq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtepi32_epi64(a: __m128i) -> __m256i {
    transmute::<i64x4, _>(simd_cast(a.as_i32x4()))
}

/// Sign-extend 8-bit integers to 16-bit integers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepi8_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtepi8_epi16(a: __m128i) -> __m256i {
    transmute::<i16x16, _>(simd_cast(a.as_i8x16()))
}

/// Sign-extend 8-bit integers to 32-bit integers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepi8_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmovsxbd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtepi8_epi32(a: __m128i) -> __m256i {
    let a = a.as_i8x16();
    let v64: i8x8 = simd_shuffle8!(a, a, [0, 1, 2, 3, 4, 5, 6, 7]);
    transmute::<i32x8, _>(simd_cast(v64))
}

/// Sign-extend 8-bit integers to 64-bit integers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepi8_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmovsxbq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtepi8_epi64(a: __m128i) -> __m256i {
    let a = a.as_i8x16();
    let v32: i8x4 = simd_shuffle4!(a, a, [0, 1, 2, 3]);
    transmute::<i64x4, _>(simd_cast(v32))
}

/// Zeroes extend packed unsigned 16-bit integers in `a` to packed 32-bit
/// integers, and stores the results in `dst`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepu16_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmovzxwd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtepu16_epi32(a: __m128i) -> __m256i {
    transmute::<i32x8, _>(simd_cast(a.as_u16x8()))
}

/// Zero-extend the lower four unsigned 16-bit integers in `a` to 64-bit
/// integers. The upper four elements of `a` are unused.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepu16_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmovzxwq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtepu16_epi64(a: __m128i) -> __m256i {
    let a = a.as_u16x8();
    let v64: u16x4 = simd_shuffle4!(a, a, [0, 1, 2, 3]);
    transmute::<i64x4, _>(simd_cast(v64))
}

/// Zero-extend unsigned 32-bit integers in `a` to 64-bit integers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepu32_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmovzxdq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtepu32_epi64(a: __m128i) -> __m256i {
    transmute::<i64x4, _>(simd_cast(a.as_u32x4()))
}

/// Zero-extend unsigned 8-bit integers in `a` to 16-bit integers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepu8_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtepu8_epi16(a: __m128i) -> __m256i {
    transmute::<i16x16, _>(simd_cast(a.as_u8x16()))
}

/// Zero-extend the lower eight unsigned 8-bit integers in `a` to 32-bit
/// integers. The upper eight elements of `a` are unused.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepu8_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmovzxbd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtepu8_epi32(a: __m128i) -> __m256i {
    let a = a.as_u8x16();
    let v64: u8x8 = simd_shuffle8!(a, a, [0, 1, 2, 3, 4, 5, 6, 7]);
    transmute::<i32x8, _>(simd_cast(v64))
}

/// Zero-extend the lower four unsigned 8-bit integers in `a` to 64-bit
/// integers. The upper twelve elements of `a` are unused.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepu8_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmovzxbq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtepu8_epi64(a: __m128i) -> __m256i {
    let a = a.as_u8x16();
    let v32: u8x4 = simd_shuffle4!(a, a, [0, 1, 2, 3]);
    transmute::<i64x4, _>(simd_cast(v32))
}

/// Extracts 128 bits (of integer data) from `a` selected with `IMM1`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_extracti128_si256)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(
    all(test, not(target_os = "windows")),
    assert_instr(vextractf128, IMM1 = 1)
)]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_extracti128_si256<const IMM1: i32>(a: __m256i) -> __m128i {
    static_assert_imm1!(IMM1);
    let a = a.as_i64x4();
    let b = _mm256_undefined_si256().as_i64x4();
    let dst: i64x2 = simd_shuffle2!(a, b, <const IMM1: i32> [[0, 1], [2, 3]][IMM1 as usize]);
    transmute(dst)
}

/// Horizontally adds adjacent pairs of 16-bit integers in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_hadd_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vphaddw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_hadd_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(phaddw(a.as_i16x16(), b.as_i16x16()))
}

/// Horizontally adds adjacent pairs of 32-bit integers in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_hadd_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vphaddd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_hadd_epi32(a: __m256i, b: __m256i) -> __m256i {
    transmute(phaddd(a.as_i32x8(), b.as_i32x8()))
}

/// Horizontally adds adjacent pairs of 16-bit integers in `a` and `b`
/// using saturation.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_hadds_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vphaddsw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_hadds_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(phaddsw(a.as_i16x16(), b.as_i16x16()))
}

/// Horizontally subtract adjacent pairs of 16-bit integers in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_hsub_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vphsubw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_hsub_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(phsubw(a.as_i16x16(), b.as_i16x16()))
}

/// Horizontally subtract adjacent pairs of 32-bit integers in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_hsub_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vphsubd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_hsub_epi32(a: __m256i, b: __m256i) -> __m256i {
    transmute(phsubd(a.as_i32x8(), b.as_i32x8()))
}

/// Horizontally subtract adjacent pairs of 16-bit integers in `a` and `b`
/// using saturation.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_hsubs_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vphsubsw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_hsubs_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(phsubsw(a.as_i16x16(), b.as_i16x16()))
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_i32gather_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherdd, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_i32gather_epi32<const SCALE: i32>(
    slice: *const i32,
    offsets: __m128i,
) -> __m128i {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm_setzero_si128().as_i32x4();
    let neg_one = _mm_set1_epi32(-1).as_i32x4();
    let offsets = offsets.as_i32x4();
    let slice = slice as *const i8;
    let r = pgatherdd(zero, slice, offsets, neg_one, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_i32gather_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherdd, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_mask_i32gather_epi32<const SCALE: i32>(
    src: __m128i,
    slice: *const i32,
    offsets: __m128i,
    mask: __m128i,
) -> __m128i {
    static_assert_imm8_scale!(SCALE);
    let src = src.as_i32x4();
    let mask = mask.as_i32x4();
    let offsets = offsets.as_i32x4();
    let slice = slice as *const i8;
    let r = pgatherdd(src, slice, offsets, mask, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_i32gather_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherdd, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_i32gather_epi32<const SCALE: i32>(
    slice: *const i32,
    offsets: __m256i,
) -> __m256i {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm256_setzero_si256().as_i32x8();
    let neg_one = _mm256_set1_epi32(-1).as_i32x8();
    let offsets = offsets.as_i32x8();
    let slice = slice as *const i8;
    let r = vpgatherdd(zero, slice, offsets, neg_one, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_i32gather_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherdd, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mask_i32gather_epi32<const SCALE: i32>(
    src: __m256i,
    slice: *const i32,
    offsets: __m256i,
    mask: __m256i,
) -> __m256i {
    static_assert_imm8_scale!(SCALE);
    let src = src.as_i32x8();
    let mask = mask.as_i32x8();
    let offsets = offsets.as_i32x8();
    let slice = slice as *const i8;
    let r = vpgatherdd(src, slice, offsets, mask, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_i32gather_ps)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherdps, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_i32gather_ps<const SCALE: i32>(slice: *const f32, offsets: __m128i) -> __m128 {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm_setzero_ps();
    let neg_one = _mm_set1_ps(-1.0);
    let offsets = offsets.as_i32x4();
    let slice = slice as *const i8;
    pgatherdps(zero, slice, offsets, neg_one, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_i32gather_ps)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherdps, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_mask_i32gather_ps<const SCALE: i32>(
    src: __m128,
    slice: *const f32,
    offsets: __m128i,
    mask: __m128,
) -> __m128 {
    static_assert_imm8_scale!(SCALE);
    let offsets = offsets.as_i32x4();
    let slice = slice as *const i8;
    pgatherdps(src, slice, offsets, mask, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_i32gather_ps)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherdps, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_i32gather_ps<const SCALE: i32>(slice: *const f32, offsets: __m256i) -> __m256 {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm256_setzero_ps();
    let neg_one = _mm256_set1_ps(-1.0);
    let offsets = offsets.as_i32x8();
    let slice = slice as *const i8;
    vpgatherdps(zero, slice, offsets, neg_one, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_i32gather_ps)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherdps, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mask_i32gather_ps<const SCALE: i32>(
    src: __m256,
    slice: *const f32,
    offsets: __m256i,
    mask: __m256,
) -> __m256 {
    static_assert_imm8_scale!(SCALE);
    let offsets = offsets.as_i32x8();
    let slice = slice as *const i8;
    vpgatherdps(src, slice, offsets, mask, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_i32gather_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherdq, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_i32gather_epi64<const SCALE: i32>(
    slice: *const i64,
    offsets: __m128i,
) -> __m128i {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm_setzero_si128().as_i64x2();
    let neg_one = _mm_set1_epi64x(-1).as_i64x2();
    let offsets = offsets.as_i32x4();
    let slice = slice as *const i8;
    let r = pgatherdq(zero, slice, offsets, neg_one, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_i32gather_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherdq, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_mask_i32gather_epi64<const SCALE: i32>(
    src: __m128i,
    slice: *const i64,
    offsets: __m128i,
    mask: __m128i,
) -> __m128i {
    static_assert_imm8_scale!(SCALE);
    let src = src.as_i64x2();
    let mask = mask.as_i64x2();
    let offsets = offsets.as_i32x4();
    let slice = slice as *const i8;
    let r = pgatherdq(src, slice, offsets, mask, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 and 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_i32gather_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherdq, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_i32gather_epi64<const SCALE: i32>(
    slice: *const i64,
    offsets: __m128i,
) -> __m256i {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm256_setzero_si256().as_i64x4();
    let neg_one = _mm256_set1_epi64x(-1).as_i64x4();
    let offsets = offsets.as_i32x4();
    let slice = slice as *const i8;
    let r = vpgatherdq(zero, slice, offsets, neg_one, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_i32gather_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherdq, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mask_i32gather_epi64<const SCALE: i32>(
    src: __m256i,
    slice: *const i64,
    offsets: __m128i,
    mask: __m256i,
) -> __m256i {
    static_assert_imm8_scale!(SCALE);
    let src = src.as_i64x4();
    let mask = mask.as_i64x4();
    let offsets = offsets.as_i32x4();
    let slice = slice as *const i8;
    let r = vpgatherdq(src, slice, offsets, mask, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_i32gather_pd)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherdpd, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_i32gather_pd<const SCALE: i32>(slice: *const f64, offsets: __m128i) -> __m128d {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm_setzero_pd();
    let neg_one = _mm_set1_pd(-1.0);
    let offsets = offsets.as_i32x4();
    let slice = slice as *const i8;
    pgatherdpd(zero, slice, offsets, neg_one, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_i32gather_pd)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherdpd, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_mask_i32gather_pd<const SCALE: i32>(
    src: __m128d,
    slice: *const f64,
    offsets: __m128i,
    mask: __m128d,
) -> __m128d {
    static_assert_imm8_scale!(SCALE);
    let offsets = offsets.as_i32x4();
    let slice = slice as *const i8;
    pgatherdpd(src, slice, offsets, mask, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_i32gather_pd)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherdpd, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_i32gather_pd<const SCALE: i32>(
    slice: *const f64,
    offsets: __m128i,
) -> __m256d {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm256_setzero_pd();
    let neg_one = _mm256_set1_pd(-1.0);
    let offsets = offsets.as_i32x4();
    let slice = slice as *const i8;
    vpgatherdpd(zero, slice, offsets, neg_one, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_i32gather_pd)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherdpd, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mask_i32gather_pd<const SCALE: i32>(
    src: __m256d,
    slice: *const f64,
    offsets: __m128i,
    mask: __m256d,
) -> __m256d {
    static_assert_imm8_scale!(SCALE);
    let offsets = offsets.as_i32x4();
    let slice = slice as *const i8;
    vpgatherdpd(src, slice, offsets, mask, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_i64gather_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherqd, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_i64gather_epi32<const SCALE: i32>(
    slice: *const i32,
    offsets: __m128i,
) -> __m128i {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm_setzero_si128().as_i32x4();
    let neg_one = _mm_set1_epi64x(-1).as_i32x4();
    let offsets = offsets.as_i64x2();
    let slice = slice as *const i8;
    let r = pgatherqd(zero, slice, offsets, neg_one, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_i64gather_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherqd, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_mask_i64gather_epi32<const SCALE: i32>(
    src: __m128i,
    slice: *const i32,
    offsets: __m128i,
    mask: __m128i,
) -> __m128i {
    static_assert_imm8_scale!(SCALE);
    let src = src.as_i32x4();
    let mask = mask.as_i32x4();
    let offsets = offsets.as_i64x2();
    let slice = slice as *const i8;
    let r = pgatherqd(src, slice, offsets, mask, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_i64gather_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherqd, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_i64gather_epi32<const SCALE: i32>(
    slice: *const i32,
    offsets: __m256i,
) -> __m128i {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm_setzero_si128().as_i32x4();
    let neg_one = _mm_set1_epi64x(-1).as_i32x4();
    let offsets = offsets.as_i64x4();
    let slice = slice as *const i8;
    let r = vpgatherqd(zero, slice, offsets, neg_one, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_i64gather_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherqd, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mask_i64gather_epi32<const SCALE: i32>(
    src: __m128i,
    slice: *const i32,
    offsets: __m256i,
    mask: __m128i,
) -> __m128i {
    static_assert_imm8_scale!(SCALE);
    let src = src.as_i32x4();
    let mask = mask.as_i32x4();
    let offsets = offsets.as_i64x4();
    let slice = slice as *const i8;
    let r = vpgatherqd(src, slice, offsets, mask, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_i64gather_ps)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherqps, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_i64gather_ps<const SCALE: i32>(slice: *const f32, offsets: __m128i) -> __m128 {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm_setzero_ps();
    let neg_one = _mm_set1_ps(-1.0);
    let offsets = offsets.as_i64x2();
    let slice = slice as *const i8;
    pgatherqps(zero, slice, offsets, neg_one, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_i64gather_ps)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherqps, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_mask_i64gather_ps<const SCALE: i32>(
    src: __m128,
    slice: *const f32,
    offsets: __m128i,
    mask: __m128,
) -> __m128 {
    static_assert_imm8_scale!(SCALE);
    let offsets = offsets.as_i64x2();
    let slice = slice as *const i8;
    pgatherqps(src, slice, offsets, mask, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_i64gather_ps)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherqps, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_i64gather_ps<const SCALE: i32>(slice: *const f32, offsets: __m256i) -> __m128 {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm_setzero_ps();
    let neg_one = _mm_set1_ps(-1.0);
    let offsets = offsets.as_i64x4();
    let slice = slice as *const i8;
    vpgatherqps(zero, slice, offsets, neg_one, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_i64gather_ps)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherqps, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mask_i64gather_ps<const SCALE: i32>(
    src: __m128,
    slice: *const f32,
    offsets: __m256i,
    mask: __m128,
) -> __m128 {
    static_assert_imm8_scale!(SCALE);
    let offsets = offsets.as_i64x4();
    let slice = slice as *const i8;
    vpgatherqps(src, slice, offsets, mask, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_i64gather_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherqq, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_i64gather_epi64<const SCALE: i32>(
    slice: *const i64,
    offsets: __m128i,
) -> __m128i {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm_setzero_si128().as_i64x2();
    let neg_one = _mm_set1_epi64x(-1).as_i64x2();
    let slice = slice as *const i8;
    let offsets = offsets.as_i64x2();
    let r = pgatherqq(zero, slice, offsets, neg_one, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_i64gather_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherqq, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_mask_i64gather_epi64<const SCALE: i32>(
    src: __m128i,
    slice: *const i64,
    offsets: __m128i,
    mask: __m128i,
) -> __m128i {
    static_assert_imm8_scale!(SCALE);
    let src = src.as_i64x2();
    let mask = mask.as_i64x2();
    let offsets = offsets.as_i64x2();
    let slice = slice as *const i8;
    let r = pgatherqq(src, slice, offsets, mask, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_i64gather_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherqq, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_i64gather_epi64<const SCALE: i32>(
    slice: *const i64,
    offsets: __m256i,
) -> __m256i {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm256_setzero_si256().as_i64x4();
    let neg_one = _mm256_set1_epi64x(-1).as_i64x4();
    let slice = slice as *const i8;
    let offsets = offsets.as_i64x4();
    let r = vpgatherqq(zero, slice, offsets, neg_one, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_i64gather_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpgatherqq, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mask_i64gather_epi64<const SCALE: i32>(
    src: __m256i,
    slice: *const i64,
    offsets: __m256i,
    mask: __m256i,
) -> __m256i {
    static_assert_imm8_scale!(SCALE);
    let src = src.as_i64x4();
    let mask = mask.as_i64x4();
    let offsets = offsets.as_i64x4();
    let slice = slice as *const i8;
    let r = vpgatherqq(src, slice, offsets, mask, SCALE as i8);
    transmute(r)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_i64gather_pd)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherqpd, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_i64gather_pd<const SCALE: i32>(slice: *const f64, offsets: __m128i) -> __m128d {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm_setzero_pd();
    let neg_one = _mm_set1_pd(-1.0);
    let slice = slice as *const i8;
    let offsets = offsets.as_i64x2();
    pgatherqpd(zero, slice, offsets, neg_one, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_i64gather_pd)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherqpd, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_mask_i64gather_pd<const SCALE: i32>(
    src: __m128d,
    slice: *const f64,
    offsets: __m128i,
    mask: __m128d,
) -> __m128d {
    static_assert_imm8_scale!(SCALE);
    let slice = slice as *const i8;
    let offsets = offsets.as_i64x2();
    pgatherqpd(src, slice, offsets, mask, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_i64gather_pd)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherqpd, SCALE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_i64gather_pd<const SCALE: i32>(
    slice: *const f64,
    offsets: __m256i,
) -> __m256d {
    static_assert_imm8_scale!(SCALE);
    let zero = _mm256_setzero_pd();
    let neg_one = _mm256_set1_pd(-1.0);
    let slice = slice as *const i8;
    let offsets = offsets.as_i64x4();
    vpgatherqpd(zero, slice, offsets, neg_one, SCALE as i8)
}

/// Returns values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in
/// that position instead.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_i64gather_pd)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vgatherqpd, SCALE = 1))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mask_i64gather_pd<const SCALE: i32>(
    src: __m256d,
    slice: *const f64,
    offsets: __m256i,
    mask: __m256d,
) -> __m256d {
    static_assert_imm8_scale!(SCALE);
    let slice = slice as *const i8;
    let offsets = offsets.as_i64x4();
    vpgatherqpd(src, slice, offsets, mask, SCALE as i8)
}

/// Copies `a` to `dst`, then insert 128 bits (of integer data) from `b` at the
/// location specified by `IMM1`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_inserti128_si256)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(
    all(test, not(target_os = "windows")),
    assert_instr(vinsertf128, IMM1 = 1)
)]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_inserti128_si256<const IMM1: i32>(a: __m256i, b: __m128i) -> __m256i {
    static_assert_imm1!(IMM1);
    let a = a.as_i64x4();
    let b = _mm256_castsi128_si256(b).as_i64x4();
    let dst: i64x4 =
        simd_shuffle4!(a, b, <const IMM1: i32> [[4, 5, 2, 3], [0, 1, 4, 5]][IMM1 as usize]);
    transmute(dst)
}

/// Multiplies packed signed 16-bit integers in `a` and `b`, producing
/// intermediate signed 32-bit integers. Horizontally add adjacent pairs
/// of intermediate 32-bit integers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_madd_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaddwd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_madd_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(pmaddwd(a.as_i16x16(), b.as_i16x16()))
}

/// Vertically multiplies each unsigned 8-bit integer from `a` with the
/// corresponding signed 8-bit integer from `b`, producing intermediate
/// signed 16-bit integers. Horizontally add adjacent pairs of intermediate
/// signed 16-bit integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maddubs_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_maddubs_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(pmaddubsw(a.as_u8x32(), b.as_u8x32()))
}

/// Loads packed 32-bit integers from memory pointed by `mem_addr` using `mask`
/// (elements are zeroed out when the highest bit is not set in the
/// corresponding element).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskload_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaskmovd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_maskload_epi32(mem_addr: *const i32, mask: __m128i) -> __m128i {
    transmute(maskloadd(mem_addr as *const i8, mask.as_i32x4()))
}

/// Loads packed 32-bit integers from memory pointed by `mem_addr` using `mask`
/// (elements are zeroed out when the highest bit is not set in the
/// corresponding element).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskload_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaskmovd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_maskload_epi32(mem_addr: *const i32, mask: __m256i) -> __m256i {
    transmute(maskloadd256(mem_addr as *const i8, mask.as_i32x8()))
}

/// Loads packed 64-bit integers from memory pointed by `mem_addr` using `mask`
/// (elements are zeroed out when the highest bit is not set in the
/// corresponding element).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskload_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaskmovq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_maskload_epi64(mem_addr: *const i64, mask: __m128i) -> __m128i {
    transmute(maskloadq(mem_addr as *const i8, mask.as_i64x2()))
}

/// Loads packed 64-bit integers from memory pointed by `mem_addr` using `mask`
/// (elements are zeroed out when the highest bit is not set in the
/// corresponding element).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskload_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaskmovq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_maskload_epi64(mem_addr: *const i64, mask: __m256i) -> __m256i {
    transmute(maskloadq256(mem_addr as *const i8, mask.as_i64x4()))
}

/// Stores packed 32-bit integers from `a` into memory pointed by `mem_addr`
/// using `mask` (elements are not stored when the highest bit is not set
/// in the corresponding element).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskstore_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaskmovd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_maskstore_epi32(mem_addr: *mut i32, mask: __m128i, a: __m128i) {
    maskstored(mem_addr as *mut i8, mask.as_i32x4(), a.as_i32x4())
}

/// Stores packed 32-bit integers from `a` into memory pointed by `mem_addr`
/// using `mask` (elements are not stored when the highest bit is not set
/// in the corresponding element).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskstore_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaskmovd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_maskstore_epi32(mem_addr: *mut i32, mask: __m256i, a: __m256i) {
    maskstored256(mem_addr as *mut i8, mask.as_i32x8(), a.as_i32x8())
}

/// Stores packed 64-bit integers from `a` into memory pointed by `mem_addr`
/// using `mask` (elements are not stored when the highest bit is not set
/// in the corresponding element).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskstore_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaskmovq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_maskstore_epi64(mem_addr: *mut i64, mask: __m128i, a: __m128i) {
    maskstoreq(mem_addr as *mut i8, mask.as_i64x2(), a.as_i64x2())
}

/// Stores packed 64-bit integers from `a` into memory pointed by `mem_addr`
/// using `mask` (elements are not stored when the highest bit is not set
/// in the corresponding element).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskstore_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaskmovq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_maskstore_epi64(mem_addr: *mut i64, mask: __m256i, a: __m256i) {
    maskstoreq256(mem_addr as *mut i8, mask.as_i64x4(), a.as_i64x4())
}

/// Compares packed 16-bit integers in `a` and `b`, and returns the packed
/// maximum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_max_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_max_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(pmaxsw(a.as_i16x16(), b.as_i16x16()))
}

/// Compares packed 32-bit integers in `a` and `b`, and returns the packed
/// maximum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_max_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaxsd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_max_epi32(a: __m256i, b: __m256i) -> __m256i {
    transmute(pmaxsd(a.as_i32x8(), b.as_i32x8()))
}

/// Compares packed 8-bit integers in `a` and `b`, and returns the packed
/// maximum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_max_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_max_epi8(a: __m256i, b: __m256i) -> __m256i {
    transmute(pmaxsb(a.as_i8x32(), b.as_i8x32()))
}

/// Compares packed unsigned 16-bit integers in `a` and `b`, and returns
/// the packed maximum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_max_epu16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_max_epu16(a: __m256i, b: __m256i) -> __m256i {
    transmute(pmaxuw(a.as_u16x16(), b.as_u16x16()))
}

/// Compares packed unsigned 32-bit integers in `a` and `b`, and returns
/// the packed maximum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_max_epu32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaxud))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_max_epu32(a: __m256i, b: __m256i) -> __m256i {
    transmute(pmaxud(a.as_u32x8(), b.as_u32x8()))
}

/// Compares packed unsigned 8-bit integers in `a` and `b`, and returns
/// the packed maximum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_max_epu8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmaxub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_max_epu8(a: __m256i, b: __m256i) -> __m256i {
    transmute(pmaxub(a.as_u8x32(), b.as_u8x32()))
}

/// Compares packed 16-bit integers in `a` and `b`, and returns the packed
/// minimum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_min_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpminsw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_min_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(pminsw(a.as_i16x16(), b.as_i16x16()))
}

/// Compares packed 32-bit integers in `a` and `b`, and returns the packed
/// minimum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_min_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpminsd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_min_epi32(a: __m256i, b: __m256i) -> __m256i {
    transmute(pminsd(a.as_i32x8(), b.as_i32x8()))
}

/// Compares packed 8-bit integers in `a` and `b`, and returns the packed
/// minimum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_min_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpminsb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_min_epi8(a: __m256i, b: __m256i) -> __m256i {
    transmute(pminsb(a.as_i8x32(), b.as_i8x32()))
}

/// Compares packed unsigned 16-bit integers in `a` and `b`, and returns
/// the packed minimum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_min_epu16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpminuw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_min_epu16(a: __m256i, b: __m256i) -> __m256i {
    transmute(pminuw(a.as_u16x16(), b.as_u16x16()))
}

/// Compares packed unsigned 32-bit integers in `a` and `b`, and returns
/// the packed minimum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_min_epu32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpminud))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_min_epu32(a: __m256i, b: __m256i) -> __m256i {
    transmute(pminud(a.as_u32x8(), b.as_u32x8()))
}

/// Compares packed unsigned 8-bit integers in `a` and `b`, and returns
/// the packed minimum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_min_epu8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpminub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_min_epu8(a: __m256i, b: __m256i) -> __m256i {
    transmute(pminub(a.as_u8x32(), b.as_u8x32()))
}

/// Creates mask from the most significant bit of each 8-bit element in `a`,
/// return the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_movemask_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmovmskb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_movemask_epi8(a: __m256i) -> i32 {
    pmovmskb(a.as_i8x32())
}

/// Computes the sum of absolute differences (SADs) of quadruplets of unsigned
/// 8-bit integers in `a` compared to those in `b`, and stores the 16-bit
/// results in dst. Eight SADs are performed for each 128-bit lane using one
/// quadruplet from `b` and eight quadruplets from `a`. One quadruplet is
/// selected from `b` starting at on the offset specified in `imm8`. Eight
/// quadruplets are formed from sequential 8-bit integers selected from `a`
/// starting at the offset specified in `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mpsadbw_epu8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vmpsadbw, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mpsadbw_epu8<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    transmute(mpsadbw(a.as_u8x32(), b.as_u8x32(), IMM8))
}

/// Multiplies the low 32-bit integers from each packed 64-bit element in
/// `a` and `b`
///
/// Returns the 64-bit results.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mul_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmuldq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mul_epi32(a: __m256i, b: __m256i) -> __m256i {
    transmute(pmuldq(a.as_i32x8(), b.as_i32x8()))
}

/// Multiplies the low unsigned 32-bit integers from each packed 64-bit
/// element in `a` and `b`
///
/// Returns the unsigned 64-bit results.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mul_epu32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmuludq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mul_epu32(a: __m256i, b: __m256i) -> __m256i {
    transmute(pmuludq(a.as_u32x8(), b.as_u32x8()))
}

/// Multiplies the packed 16-bit integers in `a` and `b`, producing
/// intermediate 32-bit integers and returning the high 16 bits of the
/// intermediate integers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mulhi_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmulhw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mulhi_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(pmulhw(a.as_i16x16(), b.as_i16x16()))
}

/// Multiplies the packed unsigned 16-bit integers in `a` and `b`, producing
/// intermediate 32-bit integers and returning the high 16 bits of the
/// intermediate integers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mulhi_epu16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mulhi_epu16(a: __m256i, b: __m256i) -> __m256i {
    transmute(pmulhuw(a.as_u16x16(), b.as_u16x16()))
}

/// Multiplies the packed 16-bit integers in `a` and `b`, producing
/// intermediate 32-bit integers, and returns the low 16 bits of the
/// intermediate integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mullo_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmullw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mullo_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_mul(a.as_i16x16(), b.as_i16x16()))
}

/// Multiplies the packed 32-bit integers in `a` and `b`, producing
/// intermediate 64-bit integers, and returns the low 32 bits of the
/// intermediate integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mullo_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmulld))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mullo_epi32(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_mul(a.as_i32x8(), b.as_i32x8()))
}

/// Multiplies packed 16-bit integers in `a` and `b`, producing
/// intermediate signed 32-bit integers. Truncate each intermediate
/// integer to the 18 most significant bits, round by adding 1, and
/// return bits `[16:1]`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mulhrs_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mulhrs_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(pmulhrsw(a.as_i16x16(), b.as_i16x16()))
}

/// Computes the bitwise OR of 256 bits (representing integer data) in `a`
/// and `b`
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_or_si256)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vorps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_or_si256(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_or(a.as_i32x8(), b.as_i32x8()))
}

/// Converts packed 16-bit integers from `a` and `b` to packed 8-bit integers
/// using signed saturation
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_packs_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpacksswb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_packs_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(packsswb(a.as_i16x16(), b.as_i16x16()))
}

/// Converts packed 32-bit integers from `a` and `b` to packed 16-bit integers
/// using signed saturation
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_packs_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpackssdw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_packs_epi32(a: __m256i, b: __m256i) -> __m256i {
    transmute(packssdw(a.as_i32x8(), b.as_i32x8()))
}

/// Converts packed 16-bit integers from `a` and `b` to packed 8-bit integers
/// using unsigned saturation
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_packus_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpackuswb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_packus_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(packuswb(a.as_i16x16(), b.as_i16x16()))
}

/// Converts packed 32-bit integers from `a` and `b` to packed 16-bit integers
/// using unsigned saturation
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_packus_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpackusdw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_packus_epi32(a: __m256i, b: __m256i) -> __m256i {
    transmute(packusdw(a.as_i32x8(), b.as_i32x8()))
}

/// Permutes packed 32-bit integers from `a` according to the content of `b`.
///
/// The last 3 bits of each integer of `b` are used as addresses into the 8
/// integers of `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permutevar8x32_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpermps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_permutevar8x32_epi32(a: __m256i, b: __m256i) -> __m256i {
    transmute(permd(a.as_u32x8(), b.as_u32x8()))
}

/// Permutes 64-bit integers from `a` using control mask `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permute4x64_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpermpd, IMM8 = 9))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_permute4x64_epi64<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    let zero = _mm256_setzero_si256().as_i64x4();
    let r: i64x4 = simd_shuffle4!(
        a.as_i64x4(),
        zero,
        <const IMM8: i32> [
            IMM8 as u32 & 0b11,
            (IMM8 as u32 >> 2) & 0b11,
            (IMM8 as u32 >> 4) & 0b11,
            (IMM8 as u32 >> 6) & 0b11,
        ],
    );
    transmute(r)
}

/// Shuffles 128-bits of integer data selected by `imm8` from `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permute2x128_si256)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vperm2f128, IMM8 = 9))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_permute2x128_si256<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    transmute(vperm2i128(a.as_i64x4(), b.as_i64x4(), IMM8 as i8))
}

/// Shuffles 64-bit floating-point elements in `a` across lanes using the
/// control in `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permute4x64_pd)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpermpd, IMM8 = 1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_permute4x64_pd<const IMM8: i32>(a: __m256d) -> __m256d {
    static_assert_imm8!(IMM8);
    simd_shuffle4!(
        a,
        _mm256_undefined_pd(),
        <const IMM8: i32> [
            IMM8 as u32 & 0b11,
            (IMM8 as u32 >> 2) & 0b11,
            (IMM8 as u32 >> 4) & 0b11,
            (IMM8 as u32 >> 6) & 0b11,
        ],
    )
}

/// Shuffles eight 32-bit foating-point elements in `a` across lanes using
/// the corresponding 32-bit integer index in `idx`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permutevar8x32_ps)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpermps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_permutevar8x32_ps(a: __m256, idx: __m256i) -> __m256 {
    permps(a, idx.as_i32x8())
}

/// Computes the absolute differences of packed unsigned 8-bit integers in `a`
/// and `b`, then horizontally sum each consecutive 8 differences to
/// produce four unsigned 16-bit integers, and pack these unsigned 16-bit
/// integers in the low 16 bits of the 64-bit return value
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sad_epu8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsadbw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sad_epu8(a: __m256i, b: __m256i) -> __m256i {
    transmute(psadbw(a.as_u8x32(), b.as_u8x32()))
}

/// Shuffles bytes from `a` according to the content of `b`.
///
/// For each of the 128-bit low and high halves of the vectors, the last
/// 4 bits of each byte of `b` are used as addresses into the respective
/// low or high 16 bytes of `a`. That is, the halves are shuffled separately.
///
/// In addition, if the highest significant bit of a byte of `b` is set, the
/// respective destination byte is set to 0.
///
/// Picturing `a` and `b` as `[u8; 32]`, `_mm256_shuffle_epi8` is logically
/// equivalent to:
///
/// ```
/// fn mm256_shuffle_epi8(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
///     let mut r = [0; 32];
///     for i in 0..16 {
///         // if the most significant bit of b is set,
///         // then the destination byte is set to 0.
///         if b[i] & 0x80 == 0u8 {
///             r[i] = a[(b[i] % 16) as usize];
///         }
///         if b[i + 16] & 0x80 == 0u8 {
///             r[i + 16] = a[(b[i + 16] % 16 + 16) as usize];
///         }
///     }
///     r
/// }
/// ```
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_shuffle_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpshufb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_shuffle_epi8(a: __m256i, b: __m256i) -> __m256i {
    transmute(pshufb(a.as_u8x32(), b.as_u8x32()))
}

/// Shuffles 32-bit integers in 128-bit lanes of `a` using the control in
/// `imm8`.
///
/// ```rust
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// #     if is_x86_feature_detected!("avx2") {
/// #         #[target_feature(enable = "avx2")]
/// #         unsafe fn worker() {
/// let a = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
///
/// let c1 = _mm256_shuffle_epi32(a, 0b00_11_10_01);
/// let c2 = _mm256_shuffle_epi32(a, 0b01_00_10_11);
///
/// let expected1 = _mm256_setr_epi32(1, 2, 3, 0, 5, 6, 7, 4);
/// let expected2 = _mm256_setr_epi32(3, 2, 0, 1, 7, 6, 4, 5);
///
/// assert_eq!(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c1, expected1)), !0);
/// assert_eq!(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c2, expected2)), !0);
/// #         }
/// #         unsafe { worker(); }
/// #     }
/// # }
/// ```
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_shuffle_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpermilps, MASK = 9))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_shuffle_epi32<const MASK: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(MASK);
    let r: i32x8 = simd_shuffle8!(
        a.as_i32x8(),
        a.as_i32x8(),
        <const MASK: i32> [
            MASK as u32 & 0b11,
            (MASK as u32 >> 2) & 0b11,
            (MASK as u32 >> 4) & 0b11,
            (MASK as u32 >> 6) & 0b11,
            (MASK as u32 & 0b11) + 4,
            ((MASK as u32 >> 2) & 0b11) + 4,
            ((MASK as u32 >> 4) & 0b11) + 4,
            ((MASK as u32 >> 6) & 0b11) + 4,
        ],
    );
    transmute(r)
}

/// Shuffles 16-bit integers in the high 64 bits of 128-bit lanes of `a` using
/// the control in `imm8`. The low 64 bits of 128-bit lanes of `a` are copied
/// to the output.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_shufflehi_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpshufhw, IMM8 = 9))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_shufflehi_epi16<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    let a = a.as_i16x16();
    let r: i16x16 = simd_shuffle16!(
        a,
        a,
        <const IMM8: i32> [
            0,
            1,
            2,
            3,
            4 + (IMM8 as u32 & 0b11),
            4 + ((IMM8 as u32 >> 2) & 0b11),
            4 + ((IMM8 as u32 >> 4) & 0b11),
            4 + ((IMM8 as u32 >> 6) & 0b11),
            8,
            9,
            10,
            11,
            12 + (IMM8 as u32 & 0b11),
            12 + ((IMM8 as u32 >> 2) & 0b11),
            12 + ((IMM8 as u32 >> 4) & 0b11),
            12 + ((IMM8 as u32 >> 6) & 0b11),
        ],
    );
    transmute(r)
}

/// Shuffles 16-bit integers in the low 64 bits of 128-bit lanes of `a` using
/// the control in `imm8`. The high 64 bits of 128-bit lanes of `a` are copied
/// to the output.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_shufflelo_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpshuflw, IMM8 = 9))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_shufflelo_epi16<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    let a = a.as_i16x16();
    let r: i16x16 = simd_shuffle16!(
        a,
        a,
        <const IMM8: i32> [
            0 + (IMM8 as u32 & 0b11),
            0 + ((IMM8 as u32 >> 2) & 0b11),
            0 + ((IMM8 as u32 >> 4) & 0b11),
            0 + ((IMM8 as u32 >> 6) & 0b11),
            4,
            5,
            6,
            7,
            8 + (IMM8 as u32 & 0b11),
            8 + ((IMM8 as u32 >> 2) & 0b11),
            8 + ((IMM8 as u32 >> 4) & 0b11),
            8 + ((IMM8 as u32 >> 6) & 0b11),
            12,
            13,
            14,
            15,
        ],
    );
    transmute(r)
}

/// Negates packed 16-bit integers in `a` when the corresponding signed
/// 16-bit integer in `b` is negative, and returns the results.
/// Results are zeroed out when the corresponding element in `b` is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sign_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsignw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sign_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(psignw(a.as_i16x16(), b.as_i16x16()))
}

/// Negates packed 32-bit integers in `a` when the corresponding signed
/// 32-bit integer in `b` is negative, and returns the results.
/// Results are zeroed out when the corresponding element in `b` is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sign_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsignd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sign_epi32(a: __m256i, b: __m256i) -> __m256i {
    transmute(psignd(a.as_i32x8(), b.as_i32x8()))
}

/// Negates packed 8-bit integers in `a` when the corresponding signed
/// 8-bit integer in `b` is negative, and returns the results.
/// Results are zeroed out when the corresponding element in `b` is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sign_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsignb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sign_epi8(a: __m256i, b: __m256i) -> __m256i {
    transmute(psignb(a.as_i8x32(), b.as_i8x32()))
}

/// Shifts packed 16-bit integers in `a` left by `count` while
/// shifting in zeros, and returns the result
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sll_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsllw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sll_epi16(a: __m256i, count: __m128i) -> __m256i {
    transmute(psllw(a.as_i16x16(), count.as_i16x8()))
}

/// Shifts packed 32-bit integers in `a` left by `count` while
/// shifting in zeros, and returns the result
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sll_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpslld))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sll_epi32(a: __m256i, count: __m128i) -> __m256i {
    transmute(pslld(a.as_i32x8(), count.as_i32x4()))
}

/// Shifts packed 64-bit integers in `a` left by `count` while
/// shifting in zeros, and returns the result
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sll_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsllq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sll_epi64(a: __m256i, count: __m128i) -> __m256i {
    transmute(psllq(a.as_i64x4(), count.as_i64x2()))
}

/// Shifts packed 16-bit integers in `a` left by `IMM8` while
/// shifting in zeros, return the results;
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_slli_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsllw, IMM8 = 7))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_slli_epi16<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    transmute(pslliw(a.as_i16x16(), IMM8))
}

/// Shifts packed 32-bit integers in `a` left by `IMM8` while
/// shifting in zeros, return the results;
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_slli_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpslld, IMM8 = 7))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_slli_epi32<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    transmute(psllid(a.as_i32x8(), IMM8))
}

/// Shifts packed 64-bit integers in `a` left by `IMM8` while
/// shifting in zeros, return the results;
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_slli_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsllq, IMM8 = 7))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_slli_epi64<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    transmute(pslliq(a.as_i64x4(), IMM8))
}

/// Shifts 128-bit lanes in `a` left by `imm8` bytes while shifting in zeros.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_slli_si256)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpslldq, IMM8 = 3))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_slli_si256<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    _mm256_bslli_epi128::<IMM8>(a)
}

/// Shifts 128-bit lanes in `a` left by `imm8` bytes while shifting in zeros.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_bslli_epi128)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpslldq, IMM8 = 3))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_bslli_epi128<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    let a = a.as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    let r: i8x32 = simd_shuffle32!(
        zero,
        a,
        <const IMM8: i32> [
            32 - (IMM8 as u32 & 0xff),
            33 - (IMM8 as u32 & 0xff),
            34 - (IMM8 as u32 & 0xff),
            35 - (IMM8 as u32 & 0xff),
            36 - (IMM8 as u32 & 0xff),
            37 - (IMM8 as u32 & 0xff),
            38 - (IMM8 as u32 & 0xff),
            39 - (IMM8 as u32 & 0xff),
            40 - (IMM8 as u32 & 0xff),
            41 - (IMM8 as u32 & 0xff),
            42 - (IMM8 as u32 & 0xff),
            43 - (IMM8 as u32 & 0xff),
            44 - (IMM8 as u32 & 0xff),
            45 - (IMM8 as u32 & 0xff),
            46 - (IMM8 as u32 & 0xff),
            47 - (IMM8 as u32 & 0xff),
            48 - (IMM8 as u32 & 0xff) - 16,
            49 - (IMM8 as u32 & 0xff) - 16,
            50 - (IMM8 as u32 & 0xff) - 16,
            51 - (IMM8 as u32 & 0xff) - 16,
            52 - (IMM8 as u32 & 0xff) - 16,
            53 - (IMM8 as u32 & 0xff) - 16,
            54 - (IMM8 as u32 & 0xff) - 16,
            55 - (IMM8 as u32 & 0xff) - 16,
            56 - (IMM8 as u32 & 0xff) - 16,
            57 - (IMM8 as u32 & 0xff) - 16,
            58 - (IMM8 as u32 & 0xff) - 16,
            59 - (IMM8 as u32 & 0xff) - 16,
            60 - (IMM8 as u32 & 0xff) - 16,
            61 - (IMM8 as u32 & 0xff) - 16,
            62 - (IMM8 as u32 & 0xff) - 16,
            63 - (IMM8 as u32 & 0xff) - 16,
        ],
    );
    transmute(r)
}

/// Shifts packed 32-bit integers in `a` left by the amount
/// specified by the corresponding element in `count` while
/// shifting in zeros, and returns the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sllv_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsllvd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_sllv_epi32(a: __m128i, count: __m128i) -> __m128i {
    transmute(psllvd(a.as_i32x4(), count.as_i32x4()))
}

/// Shifts packed 32-bit integers in `a` left by the amount
/// specified by the corresponding element in `count` while
/// shifting in zeros, and returns the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sllv_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsllvd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sllv_epi32(a: __m256i, count: __m256i) -> __m256i {
    transmute(psllvd256(a.as_i32x8(), count.as_i32x8()))
}

/// Shifts packed 64-bit integers in `a` left by the amount
/// specified by the corresponding element in `count` while
/// shifting in zeros, and returns the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sllv_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsllvq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_sllv_epi64(a: __m128i, count: __m128i) -> __m128i {
    transmute(psllvq(a.as_i64x2(), count.as_i64x2()))
}

/// Shifts packed 64-bit integers in `a` left by the amount
/// specified by the corresponding element in `count` while
/// shifting in zeros, and returns the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sllv_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsllvq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sllv_epi64(a: __m256i, count: __m256i) -> __m256i {
    transmute(psllvq256(a.as_i64x4(), count.as_i64x4()))
}

/// Shifts packed 16-bit integers in `a` right by `count` while
/// shifting in sign bits.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sra_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsraw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sra_epi16(a: __m256i, count: __m128i) -> __m256i {
    transmute(psraw(a.as_i16x16(), count.as_i16x8()))
}

/// Shifts packed 32-bit integers in `a` right by `count` while
/// shifting in sign bits.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sra_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsrad))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sra_epi32(a: __m256i, count: __m128i) -> __m256i {
    transmute(psrad(a.as_i32x8(), count.as_i32x4()))
}

/// Shifts packed 16-bit integers in `a` right by `IMM8` while
/// shifting in sign bits.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_srai_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsraw, IMM8 = 7))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_srai_epi16<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    transmute(psraiw(a.as_i16x16(), IMM8))
}

/// Shifts packed 32-bit integers in `a` right by `IMM8` while
/// shifting in sign bits.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_srai_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsrad, IMM8 = 7))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_srai_epi32<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    transmute(psraid(a.as_i32x8(), IMM8))
}

/// Shifts packed 32-bit integers in `a` right by the amount specified by the
/// corresponding element in `count` while shifting in sign bits.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_srav_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsravd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_srav_epi32(a: __m128i, count: __m128i) -> __m128i {
    transmute(psravd(a.as_i32x4(), count.as_i32x4()))
}

/// Shifts packed 32-bit integers in `a` right by the amount specified by the
/// corresponding element in `count` while shifting in sign bits.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_srav_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsravd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_srav_epi32(a: __m256i, count: __m256i) -> __m256i {
    transmute(psravd256(a.as_i32x8(), count.as_i32x8()))
}

/// Shifts 128-bit lanes in `a` right by `imm8` bytes while shifting in zeros.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_srli_si256)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsrldq, IMM8 = 1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_srli_si256<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    _mm256_bsrli_epi128::<IMM8>(a)
}

/// Shifts 128-bit lanes in `a` right by `imm8` bytes while shifting in zeros.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_bsrli_epi128)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsrldq, IMM8 = 1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_bsrli_epi128<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    let a = a.as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    let r: i8x32 = match IMM8 % 16 {
        0 => simd_shuffle32!(
            a,
            zero,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31,
            ],
        ),
        1 => simd_shuffle32!(
            a,
            zero,
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32,
            ],
        ),
        2 => simd_shuffle32!(
            a,
            zero,
            [
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 32, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 28, 29, 30, 31, 32, 32,
            ],
        ),
        3 => simd_shuffle32!(
            a,
            zero,
            [
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 32, 32, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 28, 29, 30, 31, 32, 32, 32,
            ],
        ),
        4 => simd_shuffle32!(
            a,
            zero,
            [
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 32, 32, 32, 20, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30, 31, 32, 32, 32, 32,
            ],
        ),
        5 => simd_shuffle32!(
            a,
            zero,
            [
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 32, 32, 32, 32, 21, 22, 23, 24, 25, 26,
                27, 28, 29, 30, 31, 32, 32, 32, 32, 32,
            ],
        ),
        6 => simd_shuffle32!(
            a,
            zero,
            [
                6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 32, 32, 32, 32, 32, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 32, 32, 32, 32, 32,
            ],
        ),
        7 => simd_shuffle32!(
            a,
            zero,
            [
                7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 32, 32, 32, 32, 32, 32, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 32, 32, 32, 32, 32, 32,
            ],
        ),
        8 => simd_shuffle32!(
            a,
            zero,
            [
                8, 9, 10, 11, 12, 13, 14, 15, 32, 32, 32, 32, 32, 32, 32, 32, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 32, 32, 32, 32, 32, 32, 32,
            ],
        ),
        9 => simd_shuffle32!(
            a,
            zero,
            [
                9, 10, 11, 12, 13, 14, 15, 32, 32, 32, 32, 32, 32, 32, 32, 32, 25, 26, 27, 28, 29,
                30, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            ],
        ),
        10 => simd_shuffle32!(
            a,
            zero,
            [
                10, 11, 12, 13, 14, 15, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 26, 27, 28, 29, 30,
                31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            ],
        ),
        11 => simd_shuffle32!(
            a,
            zero,
            [
                11, 12, 13, 14, 15, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 27, 28, 29, 30, 31,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            ],
        ),
        12 => simd_shuffle32!(
            a,
            zero,
            [
                12, 13, 14, 15, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 28, 29, 30, 31, 32,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            ],
        ),
        13 => simd_shuffle32!(
            a,
            zero,
            [
                13, 14, 15, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 29, 30, 31, 32, 32,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            ],
        ),
        14 => simd_shuffle32!(
            a,
            zero,
            [
                14, 15, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 30, 31, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            ],
        ),
        15 => simd_shuffle32!(
            a,
            zero,
            [
                14, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 31, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            ],
        ),
        _ => zero,
    };
    transmute(r)
}

/// Shifts packed 16-bit integers in `a` right by `count` while shifting in
/// zeros.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_srl_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsrlw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_srl_epi16(a: __m256i, count: __m128i) -> __m256i {
    transmute(psrlw(a.as_i16x16(), count.as_i16x8()))
}

/// Shifts packed 32-bit integers in `a` right by `count` while shifting in
/// zeros.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_srl_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsrld))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_srl_epi32(a: __m256i, count: __m128i) -> __m256i {
    transmute(psrld(a.as_i32x8(), count.as_i32x4()))
}

/// Shifts packed 64-bit integers in `a` right by `count` while shifting in
/// zeros.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_srl_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsrlq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_srl_epi64(a: __m256i, count: __m128i) -> __m256i {
    transmute(psrlq(a.as_i64x4(), count.as_i64x2()))
}

/// Shifts packed 16-bit integers in `a` right by `IMM8` while shifting in
/// zeros
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_srli_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsrlw, IMM8 = 7))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_srli_epi16<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    transmute(psrliw(a.as_i16x16(), IMM8))
}

/// Shifts packed 32-bit integers in `a` right by `IMM8` while shifting in
/// zeros
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_srli_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsrld, IMM8 = 7))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_srli_epi32<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    transmute(psrlid(a.as_i32x8(), IMM8))
}

/// Shifts packed 64-bit integers in `a` right by `IMM8` while shifting in
/// zeros
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_srli_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsrlq, IMM8 = 7))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_srli_epi64<const IMM8: i32>(a: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    transmute(psrliq(a.as_i64x4(), IMM8))
}

/// Shifts packed 32-bit integers in `a` right by the amount specified by
/// the corresponding element in `count` while shifting in zeros,
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_srlv_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsrlvd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_srlv_epi32(a: __m128i, count: __m128i) -> __m128i {
    transmute(psrlvd(a.as_i32x4(), count.as_i32x4()))
}

/// Shifts packed 32-bit integers in `a` right by the amount specified by
/// the corresponding element in `count` while shifting in zeros,
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_srlv_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsrlvd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_srlv_epi32(a: __m256i, count: __m256i) -> __m256i {
    transmute(psrlvd256(a.as_i32x8(), count.as_i32x8()))
}

/// Shifts packed 64-bit integers in `a` right by the amount specified by
/// the corresponding element in `count` while shifting in zeros,
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_srlv_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsrlvq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_srlv_epi64(a: __m128i, count: __m128i) -> __m128i {
    transmute(psrlvq(a.as_i64x2(), count.as_i64x2()))
}

/// Shifts packed 64-bit integers in `a` right by the amount specified by
/// the corresponding element in `count` while shifting in zeros,
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_srlv_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsrlvq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_srlv_epi64(a: __m256i, count: __m256i) -> __m256i {
    transmute(psrlvq256(a.as_i64x4(), count.as_i64x4()))
}

// TODO _mm256_stream_load_si256 (__m256i const* mem_addr)

/// Subtract packed 16-bit integers in `b` from packed 16-bit integers in `a`
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sub_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsubw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sub_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_sub(a.as_i16x16(), b.as_i16x16()))
}

/// Subtract packed 32-bit integers in `b` from packed 32-bit integers in `a`
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sub_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsubd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sub_epi32(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_sub(a.as_i32x8(), b.as_i32x8()))
}

/// Subtract packed 64-bit integers in `b` from packed 64-bit integers in `a`
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sub_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsubq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sub_epi64(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_sub(a.as_i64x4(), b.as_i64x4()))
}

/// Subtract packed 8-bit integers in `b` from packed 8-bit integers in `a`
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sub_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsubb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sub_epi8(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_sub(a.as_i8x32(), b.as_i8x32()))
}

/// Subtract packed 16-bit integers in `b` from packed 16-bit integers in
/// `a` using saturation.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_subs_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsubsw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_subs_epi16(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_saturating_sub(a.as_i16x16(), b.as_i16x16()))
}

/// Subtract packed 8-bit integers in `b` from packed 8-bit integers in
/// `a` using saturation.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_subs_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsubsb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_subs_epi8(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_saturating_sub(a.as_i8x32(), b.as_i8x32()))
}

/// Subtract packed unsigned 16-bit integers in `b` from packed 16-bit
/// integers in `a` using saturation.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_subs_epu16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsubusw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_subs_epu16(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_saturating_sub(a.as_u16x16(), b.as_u16x16()))
}

/// Subtract packed unsigned 8-bit integers in `b` from packed 8-bit
/// integers in `a` using saturation.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_subs_epu8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpsubusb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_subs_epu8(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_saturating_sub(a.as_u8x32(), b.as_u8x32()))
}

/// Unpacks and interleave 8-bit integers from the high half of each
/// 128-bit lane in `a` and `b`.
///
/// ```rust
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// #     if is_x86_feature_detected!("avx2") {
/// #         #[target_feature(enable = "avx2")]
/// #         unsafe fn worker() {
/// let a = _mm256_setr_epi8(
///     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
///     20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
/// );
/// let b = _mm256_setr_epi8(
///     0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15,
///     -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29,
///     -30, -31,
/// );
///
/// let c = _mm256_unpackhi_epi8(a, b);
///
/// let expected = _mm256_setr_epi8(
///     8, -8, 9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14, -14, 15, -15,
///     24, -24, 25, -25, 26, -26, 27, -27, 28, -28, 29, -29, 30, -30, 31,
///     -31,
/// );
/// assert_eq!(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c, expected)), !0);
///
/// #         }
/// #         unsafe { worker(); }
/// #     }
/// # }
/// ```
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_unpackhi_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_unpackhi_epi8(a: __m256i, b: __m256i) -> __m256i {
    #[rustfmt::skip]
    let r: i8x32 = simd_shuffle32!(a.as_i8x32(), b.as_i8x32(), [
            8, 40, 9, 41, 10, 42, 11, 43,
            12, 44, 13, 45, 14, 46, 15, 47,
            24, 56, 25, 57, 26, 58, 27, 59,
            28, 60, 29, 61, 30, 62, 31, 63,
    ]);
    transmute(r)
}

/// Unpacks and interleave 8-bit integers from the low half of each
/// 128-bit lane of `a` and `b`.
///
/// ```rust
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// #     if is_x86_feature_detected!("avx2") {
/// #         #[target_feature(enable = "avx2")]
/// #         unsafe fn worker() {
/// let a = _mm256_setr_epi8(
///     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
///     20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
/// );
/// let b = _mm256_setr_epi8(
///     0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15,
///     -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29,
///     -30, -31,
/// );
///
/// let c = _mm256_unpacklo_epi8(a, b);
///
/// let expected = _mm256_setr_epi8(
///     0, 0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 16, -16, 17,
///     -17, 18, -18, 19, -19, 20, -20, 21, -21, 22, -22, 23, -23,
/// );
/// assert_eq!(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c, expected)), !0);
///
/// #         }
/// #         unsafe { worker(); }
/// #     }
/// # }
/// ```
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_unpacklo_epi8)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_unpacklo_epi8(a: __m256i, b: __m256i) -> __m256i {
    #[rustfmt::skip]
    let r: i8x32 = simd_shuffle32!(a.as_i8x32(), b.as_i8x32(), [
        0, 32, 1, 33, 2, 34, 3, 35,
        4, 36, 5, 37, 6, 38, 7, 39,
        16, 48, 17, 49, 18, 50, 19, 51,
        20, 52, 21, 53, 22, 54, 23, 55,
    ]);
    transmute(r)
}

/// Unpacks and interleave 16-bit integers from the high half of each
/// 128-bit lane of `a` and `b`.
///
/// ```rust
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// #     if is_x86_feature_detected!("avx2") {
/// #         #[target_feature(enable = "avx2")]
/// #         unsafe fn worker() {
/// let a = _mm256_setr_epi16(
///     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
/// );
/// let b = _mm256_setr_epi16(
///     0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15,
/// );
///
/// let c = _mm256_unpackhi_epi16(a, b);
///
/// let expected = _mm256_setr_epi16(
///     4, -4, 5, -5, 6, -6, 7, -7, 12, -12, 13, -13, 14, -14, 15, -15,
/// );
/// assert_eq!(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c, expected)), !0);
///
/// #         }
/// #         unsafe { worker(); }
/// #     }
/// # }
/// ```
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_unpackhi_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_unpackhi_epi16(a: __m256i, b: __m256i) -> __m256i {
    let r: i16x16 = simd_shuffle16!(
        a.as_i16x16(),
        b.as_i16x16(),
        [4, 20, 5, 21, 6, 22, 7, 23, 12, 28, 13, 29, 14, 30, 15, 31],
    );
    transmute(r)
}

/// Unpacks and interleave 16-bit integers from the low half of each
/// 128-bit lane of `a` and `b`.
///
/// ```rust
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// #     if is_x86_feature_detected!("avx2") {
/// #         #[target_feature(enable = "avx2")]
/// #         unsafe fn worker() {
///
/// let a = _mm256_setr_epi16(
///     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
/// );
/// let b = _mm256_setr_epi16(
///     0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15,
/// );
///
/// let c = _mm256_unpacklo_epi16(a, b);
///
/// let expected = _mm256_setr_epi16(
///     0, 0, 1, -1, 2, -2, 3, -3, 8, -8, 9, -9, 10, -10, 11, -11,
/// );
/// assert_eq!(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c, expected)), !0);
///
/// #         }
/// #         unsafe { worker(); }
/// #     }
/// # }
/// ```
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_unpacklo_epi16)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_unpacklo_epi16(a: __m256i, b: __m256i) -> __m256i {
    let r: i16x16 = simd_shuffle16!(
        a.as_i16x16(),
        b.as_i16x16(),
        [0, 16, 1, 17, 2, 18, 3, 19, 8, 24, 9, 25, 10, 26, 11, 27],
    );
    transmute(r)
}

/// Unpacks and interleave 32-bit integers from the high half of each
/// 128-bit lane of `a` and `b`.
///
/// ```rust
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// #     if is_x86_feature_detected!("avx2") {
/// #         #[target_feature(enable = "avx2")]
/// #         unsafe fn worker() {
/// let a = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
/// let b = _mm256_setr_epi32(0, -1, -2, -3, -4, -5, -6, -7);
///
/// let c = _mm256_unpackhi_epi32(a, b);
///
/// let expected = _mm256_setr_epi32(2, -2, 3, -3, 6, -6, 7, -7);
/// assert_eq!(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c, expected)), !0);
///
/// #         }
/// #         unsafe { worker(); }
/// #     }
/// # }
/// ```
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_unpackhi_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vunpckhps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_unpackhi_epi32(a: __m256i, b: __m256i) -> __m256i {
    let r: i32x8 = simd_shuffle8!(a.as_i32x8(), b.as_i32x8(), [2, 10, 3, 11, 6, 14, 7, 15]);
    transmute(r)
}

/// Unpacks and interleave 32-bit integers from the low half of each
/// 128-bit lane of `a` and `b`.
///
/// ```rust
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// #     if is_x86_feature_detected!("avx2") {
/// #         #[target_feature(enable = "avx2")]
/// #         unsafe fn worker() {
/// let a = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
/// let b = _mm256_setr_epi32(0, -1, -2, -3, -4, -5, -6, -7);
///
/// let c = _mm256_unpacklo_epi32(a, b);
///
/// let expected = _mm256_setr_epi32(0, 0, 1, -1, 4, -4, 5, -5);
/// assert_eq!(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c, expected)), !0);
///
/// #         }
/// #         unsafe { worker(); }
/// #     }
/// # }
/// ```
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_unpacklo_epi32)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vunpcklps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_unpacklo_epi32(a: __m256i, b: __m256i) -> __m256i {
    let r: i32x8 = simd_shuffle8!(a.as_i32x8(), b.as_i32x8(), [0, 8, 1, 9, 4, 12, 5, 13]);
    transmute(r)
}

/// Unpacks and interleave 64-bit integers from the high half of each
/// 128-bit lane of `a` and `b`.
///
/// ```rust
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// #     if is_x86_feature_detected!("avx2") {
/// #         #[target_feature(enable = "avx2")]
/// #         unsafe fn worker() {
/// let a = _mm256_setr_epi64x(0, 1, 2, 3);
/// let b = _mm256_setr_epi64x(0, -1, -2, -3);
///
/// let c = _mm256_unpackhi_epi64(a, b);
///
/// let expected = _mm256_setr_epi64x(1, -1, 3, -3);
/// assert_eq!(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c, expected)), !0);
///
/// #         }
/// #         unsafe { worker(); }
/// #     }
/// # }
/// ```
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_unpackhi_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vunpckhpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_unpackhi_epi64(a: __m256i, b: __m256i) -> __m256i {
    let r: i64x4 = simd_shuffle4!(a.as_i64x4(), b.as_i64x4(), [1, 5, 3, 7]);
    transmute(r)
}

/// Unpacks and interleave 64-bit integers from the low half of each
/// 128-bit lane of `a` and `b`.
///
/// ```rust
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// #     if is_x86_feature_detected!("avx2") {
/// #         #[target_feature(enable = "avx2")]
/// #         unsafe fn worker() {
/// let a = _mm256_setr_epi64x(0, 1, 2, 3);
/// let b = _mm256_setr_epi64x(0, -1, -2, -3);
///
/// let c = _mm256_unpacklo_epi64(a, b);
///
/// let expected = _mm256_setr_epi64x(0, 0, 2, -2);
/// assert_eq!(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c, expected)), !0);
///
/// #         }
/// #         unsafe { worker(); }
/// #     }
/// # }
/// ```
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_unpacklo_epi64)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vunpcklpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_unpacklo_epi64(a: __m256i, b: __m256i) -> __m256i {
    let r: i64x4 = simd_shuffle4!(a.as_i64x4(), b.as_i64x4(), [0, 4, 2, 6]);
    transmute(r)
}

/// Computes the bitwise XOR of 256 bits (representing integer data)
/// in `a` and `b`
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_xor_si256)
#[inline]
#[target_feature(enable = "avx2")]
#[cfg_attr(test, assert_instr(vxorps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_xor_si256(a: __m256i, b: __m256i) -> __m256i {
    transmute(simd_xor(a.as_i64x4(), b.as_i64x4()))
}

/// Extracts an 8-bit integer from `a`, selected with `INDEX`. Returns a 32-bit
/// integer containing the zero-extended integer data.
///
/// See [LLVM commit D20468](https://reviews.llvm.org/D20468).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_extract_epi8)
#[inline]
#[target_feature(enable = "avx2")]
// This intrinsic has no corresponding instruction.
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_extract_epi8<const INDEX: i32>(a: __m256i) -> i32 {
    static_assert_imm5!(INDEX);
    simd_extract::<_, u8>(a.as_u8x32(), INDEX as u32) as i32
}

/// Extracts a 16-bit integer from `a`, selected with `INDEX`. Returns a 32-bit
/// integer containing the zero-extended integer data.
///
/// See [LLVM commit D20468](https://reviews.llvm.org/D20468).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_extract_epi16)
#[inline]
#[target_feature(enable = "avx2")]
// This intrinsic has no corresponding instruction.
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_extract_epi16<const INDEX: i32>(a: __m256i) -> i32 {
    static_assert_imm4!(INDEX);
    simd_extract::<_, u16>(a.as_u16x16(), INDEX as u32) as i32
}

/// Extracts a 32-bit integer from `a`, selected with `INDEX`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_extract_epi32)
#[inline]
#[target_feature(enable = "avx2")]
// This intrinsic has no corresponding instruction.
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_extract_epi32<const INDEX: i32>(a: __m256i) -> i32 {
    static_assert_imm3!(INDEX);
    simd_extract(a.as_i32x8(), INDEX as u32)
}

/// Returns the first element of the input vector of `[4 x double]`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtsd_f64)
#[inline]
#[target_feature(enable = "avx2")]
//#[cfg_attr(test, assert_instr(movsd))] FIXME
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtsd_f64(a: __m256d) -> f64 {
    simd_extract(a, 0)
}

/// Returns the first element of the input vector of `[8 x i32]`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtsi256_si32)
#[inline]
#[target_feature(enable = "avx2")]
//#[cfg_attr(test, assert_instr(movd))] FIXME
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtsi256_si32(a: __m256i) -> i32 {
    simd_extract(a.as_i32x8(), 0)
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx2.pabs.b"]
    fn pabsb(a: i8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.pabs.w"]
    fn pabsw(a: i16x16) -> u16x16;
    #[link_name = "llvm.x86.avx2.pabs.d"]
    fn pabsd(a: i32x8) -> u32x8;
    #[link_name = "llvm.x86.avx2.pavg.b"]
    fn pavgb(a: u8x32, b: u8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.pavg.w"]
    fn pavgw(a: u16x16, b: u16x16) -> u16x16;
    #[link_name = "llvm.x86.avx2.pblendvb"]
    fn pblendvb(a: i8x32, b: i8x32, mask: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx2.phadd.w"]
    fn phaddw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.phadd.d"]
    fn phaddd(a: i32x8, b: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.phadd.sw"]
    fn phaddsw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.phsub.w"]
    fn phsubw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.phsub.d"]
    fn phsubd(a: i32x8, b: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.phsub.sw"]
    fn phsubsw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.pmadd.wd"]
    fn pmaddwd(a: i16x16, b: i16x16) -> i32x8;
    #[link_name = "llvm.x86.avx2.pmadd.ub.sw"]
    fn pmaddubsw(a: u8x32, b: u8x32) -> i16x16;
    #[link_name = "llvm.x86.avx2.maskload.d"]
    fn maskloadd(mem_addr: *const i8, mask: i32x4) -> i32x4;
    #[link_name = "llvm.x86.avx2.maskload.d.256"]
    fn maskloadd256(mem_addr: *const i8, mask: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.maskload.q"]
    fn maskloadq(mem_addr: *const i8, mask: i64x2) -> i64x2;
    #[link_name = "llvm.x86.avx2.maskload.q.256"]
    fn maskloadq256(mem_addr: *const i8, mask: i64x4) -> i64x4;
    #[link_name = "llvm.x86.avx2.maskstore.d"]
    fn maskstored(mem_addr: *mut i8, mask: i32x4, a: i32x4);
    #[link_name = "llvm.x86.avx2.maskstore.d.256"]
    fn maskstored256(mem_addr: *mut i8, mask: i32x8, a: i32x8);
    #[link_name = "llvm.x86.avx2.maskstore.q"]
    fn maskstoreq(mem_addr: *mut i8, mask: i64x2, a: i64x2);
    #[link_name = "llvm.x86.avx2.maskstore.q.256"]
    fn maskstoreq256(mem_addr: *mut i8, mask: i64x4, a: i64x4);
    #[link_name = "llvm.x86.avx2.pmaxs.w"]
    fn pmaxsw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.pmaxs.d"]
    fn pmaxsd(a: i32x8, b: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.pmaxs.b"]
    fn pmaxsb(a: i8x32, b: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx2.pmaxu.w"]
    fn pmaxuw(a: u16x16, b: u16x16) -> u16x16;
    #[link_name = "llvm.x86.avx2.pmaxu.d"]
    fn pmaxud(a: u32x8, b: u32x8) -> u32x8;
    #[link_name = "llvm.x86.avx2.pmaxu.b"]
    fn pmaxub(a: u8x32, b: u8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.pmins.w"]
    fn pminsw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.pmins.d"]
    fn pminsd(a: i32x8, b: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.pmins.b"]
    fn pminsb(a: i8x32, b: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx2.pminu.w"]
    fn pminuw(a: u16x16, b: u16x16) -> u16x16;
    #[link_name = "llvm.x86.avx2.pminu.d"]
    fn pminud(a: u32x8, b: u32x8) -> u32x8;
    #[link_name = "llvm.x86.avx2.pminu.b"]
    fn pminub(a: u8x32, b: u8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.pmovmskb"]
    fn pmovmskb(a: i8x32) -> i32;
    #[link_name = "llvm.x86.avx2.mpsadbw"]
    fn mpsadbw(a: u8x32, b: u8x32, imm8: i32) -> u16x16;
    #[link_name = "llvm.x86.avx2.pmulhu.w"]
    fn pmulhuw(a: u16x16, b: u16x16) -> u16x16;
    #[link_name = "llvm.x86.avx2.pmulh.w"]
    fn pmulhw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.pmul.dq"]
    fn pmuldq(a: i32x8, b: i32x8) -> i64x4;
    #[link_name = "llvm.x86.avx2.pmulu.dq"]
    fn pmuludq(a: u32x8, b: u32x8) -> u64x4;
    #[link_name = "llvm.x86.avx2.pmul.hr.sw"]
    fn pmulhrsw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.packsswb"]
    fn packsswb(a: i16x16, b: i16x16) -> i8x32;
    #[link_name = "llvm.x86.avx2.packssdw"]
    fn packssdw(a: i32x8, b: i32x8) -> i16x16;
    #[link_name = "llvm.x86.avx2.packuswb"]
    fn packuswb(a: i16x16, b: i16x16) -> u8x32;
    #[link_name = "llvm.x86.avx2.packusdw"]
    fn packusdw(a: i32x8, b: i32x8) -> u16x16;
    #[link_name = "llvm.x86.avx2.psad.bw"]
    fn psadbw(a: u8x32, b: u8x32) -> u64x4;
    #[link_name = "llvm.x86.avx2.psign.b"]
    fn psignb(a: i8x32, b: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx2.psign.w"]
    fn psignw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.psign.d"]
    fn psignd(a: i32x8, b: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.psll.w"]
    fn psllw(a: i16x16, count: i16x8) -> i16x16;
    #[link_name = "llvm.x86.avx2.psll.d"]
    fn pslld(a: i32x8, count: i32x4) -> i32x8;
    #[link_name = "llvm.x86.avx2.psll.q"]
    fn psllq(a: i64x4, count: i64x2) -> i64x4;
    #[link_name = "llvm.x86.avx2.pslli.w"]
    fn pslliw(a: i16x16, imm8: i32) -> i16x16;
    #[link_name = "llvm.x86.avx2.pslli.d"]
    fn psllid(a: i32x8, imm8: i32) -> i32x8;
    #[link_name = "llvm.x86.avx2.pslli.q"]
    fn pslliq(a: i64x4, imm8: i32) -> i64x4;
    #[link_name = "llvm.x86.avx2.psllv.d"]
    fn psllvd(a: i32x4, count: i32x4) -> i32x4;
    #[link_name = "llvm.x86.avx2.psllv.d.256"]
    fn psllvd256(a: i32x8, count: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.psllv.q"]
    fn psllvq(a: i64x2, count: i64x2) -> i64x2;
    #[link_name = "llvm.x86.avx2.psllv.q.256"]
    fn psllvq256(a: i64x4, count: i64x4) -> i64x4;
    #[link_name = "llvm.x86.avx2.psra.w"]
    fn psraw(a: i16x16, count: i16x8) -> i16x16;
    #[link_name = "llvm.x86.avx2.psra.d"]
    fn psrad(a: i32x8, count: i32x4) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrai.w"]
    fn psraiw(a: i16x16, imm8: i32) -> i16x16;
    #[link_name = "llvm.x86.avx2.psrai.d"]
    fn psraid(a: i32x8, imm8: i32) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrav.d"]
    fn psravd(a: i32x4, count: i32x4) -> i32x4;
    #[link_name = "llvm.x86.avx2.psrav.d.256"]
    fn psravd256(a: i32x8, count: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrl.w"]
    fn psrlw(a: i16x16, count: i16x8) -> i16x16;
    #[link_name = "llvm.x86.avx2.psrl.d"]
    fn psrld(a: i32x8, count: i32x4) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrl.q"]
    fn psrlq(a: i64x4, count: i64x2) -> i64x4;
    #[link_name = "llvm.x86.avx2.psrli.w"]
    fn psrliw(a: i16x16, imm8: i32) -> i16x16;
    #[link_name = "llvm.x86.avx2.psrli.d"]
    fn psrlid(a: i32x8, imm8: i32) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrli.q"]
    fn psrliq(a: i64x4, imm8: i32) -> i64x4;
    #[link_name = "llvm.x86.avx2.psrlv.d"]
    fn psrlvd(a: i32x4, count: i32x4) -> i32x4;
    #[link_name = "llvm.x86.avx2.psrlv.d.256"]
    fn psrlvd256(a: i32x8, count: i32x8) -> i32x8;
    #[link_name = "llvm.x86.avx2.psrlv.q"]
    fn psrlvq(a: i64x2, count: i64x2) -> i64x2;
    #[link_name = "llvm.x86.avx2.psrlv.q.256"]
    fn psrlvq256(a: i64x4, count: i64x4) -> i64x4;
    #[link_name = "llvm.x86.avx2.pshuf.b"]
    fn pshufb(a: u8x32, b: u8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.permd"]
    fn permd(a: u32x8, b: u32x8) -> u32x8;
    #[link_name = "llvm.x86.avx2.permps"]
    fn permps(a: __m256, b: i32x8) -> __m256;
    #[link_name = "llvm.x86.avx2.vperm2i128"]
    fn vperm2i128(a: i64x4, b: i64x4, imm8: i8) -> i64x4;
    #[link_name = "llvm.x86.avx2.gather.d.d"]
    fn pgatherdd(src: i32x4, slice: *const i8, offsets: i32x4, mask: i32x4, scale: i8) -> i32x4;
    #[link_name = "llvm.x86.avx2.gather.d.d.256"]
    fn vpgatherdd(src: i32x8, slice: *const i8, offsets: i32x8, mask: i32x8, scale: i8) -> i32x8;
    #[link_name = "llvm.x86.avx2.gather.d.q"]
    fn pgatherdq(src: i64x2, slice: *const i8, offsets: i32x4, mask: i64x2, scale: i8) -> i64x2;
    #[link_name = "llvm.x86.avx2.gather.d.q.256"]
    fn vpgatherdq(src: i64x4, slice: *const i8, offsets: i32x4, mask: i64x4, scale: i8) -> i64x4;
    #[link_name = "llvm.x86.avx2.gather.q.d"]
    fn pgatherqd(src: i32x4, slice: *const i8, offsets: i64x2, mask: i32x4, scale: i8) -> i32x4;
    #[link_name = "llvm.x86.avx2.gather.q.d.256"]
    fn vpgatherqd(src: i32x4, slice: *const i8, offsets: i64x4, mask: i32x4, scale: i8) -> i32x4;
    #[link_name = "llvm.x86.avx2.gather.q.q"]
    fn pgatherqq(src: i64x2, slice: *const i8, offsets: i64x2, mask: i64x2, scale: i8) -> i64x2;
    #[link_name = "llvm.x86.avx2.gather.q.q.256"]
    fn vpgatherqq(src: i64x4, slice: *const i8, offsets: i64x4, mask: i64x4, scale: i8) -> i64x4;
    #[link_name = "llvm.x86.avx2.gather.d.pd"]
    fn pgatherdpd(
        src: __m128d,
        slice: *const i8,
        offsets: i32x4,
        mask: __m128d,
        scale: i8,
    ) -> __m128d;
    #[link_name = "llvm.x86.avx2.gather.d.pd.256"]
    fn vpgatherdpd(
        src: __m256d,
        slice: *const i8,
        offsets: i32x4,
        mask: __m256d,
        scale: i8,
    ) -> __m256d;
    #[link_name = "llvm.x86.avx2.gather.q.pd"]
    fn pgatherqpd(
        src: __m128d,
        slice: *const i8,
        offsets: i64x2,
        mask: __m128d,
        scale: i8,
    ) -> __m128d;
    #[link_name = "llvm.x86.avx2.gather.q.pd.256"]
    fn vpgatherqpd(
        src: __m256d,
        slice: *const i8,
        offsets: i64x4,
        mask: __m256d,
        scale: i8,
    ) -> __m256d;
    #[link_name = "llvm.x86.avx2.gather.d.ps"]
    fn pgatherdps(src: __m128, slice: *const i8, offsets: i32x4, mask: __m128, scale: i8)
        -> __m128;
    #[link_name = "llvm.x86.avx2.gather.d.ps.256"]
    fn vpgatherdps(
        src: __m256,
        slice: *const i8,
        offsets: i32x8,
        mask: __m256,
        scale: i8,
    ) -> __m256;
    #[link_name = "llvm.x86.avx2.gather.q.ps"]
    fn pgatherqps(src: __m128, slice: *const i8, offsets: i64x2, mask: __m128, scale: i8)
        -> __m128;
    #[link_name = "llvm.x86.avx2.gather.q.ps.256"]
    fn vpgatherqps(
        src: __m128,
        slice: *const i8,
        offsets: i64x4,
        mask: __m128,
        scale: i8,
    ) -> __m128;
    #[link_name = "llvm.x86.avx2.psll.dq"]
    fn vpslldq(a: i64x4, b: i32) -> i64x4;
    #[link_name = "llvm.x86.avx2.psrl.dq"]
    fn vpsrldq(a: i64x4, b: i32) -> i64x4;
}

#[cfg(test)]
mod tests {

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_abs_epi32() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi32(
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
        );
        let r = _mm256_abs_epi32(a);
        #[rustfmt::skip]
        let e = _mm256_setr_epi32(
            0, 1, 1, i32::MAX,
            i32::MAX.wrapping_add(1), 100, 100, 32,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_abs_epi16() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi16(
            0,  1, -1, 2, -2, 3, -3, 4,
            -4, 5, -5, i16::MAX, i16::MIN, 100, -100, -32,
        );
        let r = _mm256_abs_epi16(a);
        #[rustfmt::skip]
        let e = _mm256_setr_epi16(
            0, 1, 1, 2, 2, 3, 3, 4,
            4, 5, 5, i16::MAX, i16::MAX.wrapping_add(1), 100, 100, 32,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_abs_epi8() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            0, 1, -1, 2, -2, 3, -3, 4,
            -4, 5, -5, i8::MAX, i8::MIN, 100, -100, -32,
            0, 1, -1, 2, -2, 3, -3, 4,
            -4, 5, -5, i8::MAX, i8::MIN, 100, -100, -32,
        );
        let r = _mm256_abs_epi8(a);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            0, 1, 1, 2, 2, 3, 3, 4,
            4, 5, 5, i8::MAX, i8::MAX.wrapping_add(1), 100, 100, 32,
            0, 1, 1, 2, 2, 3, 3, 4,
            4, 5, 5, i8::MAX, i8::MAX.wrapping_add(1), 100, 100, 32,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_add_epi64() {
        let a = _mm256_setr_epi64x(-10, 0, 100, 1_000_000_000);
        let b = _mm256_setr_epi64x(-1, 0, 1, 2);
        let r = _mm256_add_epi64(a, b);
        let e = _mm256_setr_epi64x(-11, 0, 101, 1_000_000_002);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_add_epi32() {
        let a = _mm256_setr_epi32(-1, 0, 1, 2, 3, 4, 5, 6);
        let b = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm256_add_epi32(a, b);
        let e = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_add_epi16() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi16(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        #[rustfmt::skip]
        let b = _mm256_setr_epi16(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        let r = _mm256_add_epi16(a, b);
        #[rustfmt::skip]
        let e = _mm256_setr_epi16(
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 26, 28, 30,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_add_epi8() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        #[rustfmt::skip]
        let b = _mm256_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        let r = _mm256_add_epi8(a, b);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 26, 28, 30,
            32, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_adds_epi8() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        #[rustfmt::skip]
        let b = _mm256_setr_epi8(
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63,
        );
        let r = _mm256_adds_epi8(a, b);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            32, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62,
            64, 66, 68, 70, 72, 74, 76, 78,
            80, 82, 84, 86, 88, 90, 92, 94,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_adds_epi8_saturate_positive() {
        let a = _mm256_set1_epi8(0x7F);
        let b = _mm256_set1_epi8(1);
        let r = _mm256_adds_epi8(a, b);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_adds_epi8_saturate_negative() {
        let a = _mm256_set1_epi8(-0x80);
        let b = _mm256_set1_epi8(-1);
        let r = _mm256_adds_epi8(a, b);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_adds_epi16() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi16(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        #[rustfmt::skip]
        let b = _mm256_setr_epi16(
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
        );
        let r = _mm256_adds_epi16(a, b);
        #[rustfmt::skip]
        let e = _mm256_setr_epi16(
            32, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62,
        );

        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_adds_epi16_saturate_positive() {
        let a = _mm256_set1_epi16(0x7FFF);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_adds_epi16(a, b);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_adds_epi16_saturate_negative() {
        let a = _mm256_set1_epi16(-0x8000);
        let b = _mm256_set1_epi16(-1);
        let r = _mm256_adds_epi16(a, b);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_adds_epu8() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        #[rustfmt::skip]
        let b = _mm256_setr_epi8(
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63,
        );
        let r = _mm256_adds_epu8(a, b);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            32, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62,
            64, 66, 68, 70, 72, 74, 76, 78,
            80, 82, 84, 86, 88, 90, 92, 94,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_adds_epu8_saturate() {
        let a = _mm256_set1_epi8(!0);
        let b = _mm256_set1_epi8(1);
        let r = _mm256_adds_epu8(a, b);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_adds_epu16() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi16(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        #[rustfmt::skip]
        let b = _mm256_setr_epi16(
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
        );
        let r = _mm256_adds_epu16(a, b);
        #[rustfmt::skip]
        let e = _mm256_setr_epi16(
            32, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62,
        );

        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_adds_epu16_saturate() {
        let a = _mm256_set1_epi16(!0);
        let b = _mm256_set1_epi16(1);
        let r = _mm256_adds_epu16(a, b);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_and_si256() {
        let a = _mm256_set1_epi8(5);
        let b = _mm256_set1_epi8(3);
        let got = _mm256_and_si256(a, b);
        assert_eq_m256i(got, _mm256_set1_epi8(1));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_andnot_si256() {
        let a = _mm256_set1_epi8(5);
        let b = _mm256_set1_epi8(3);
        let got = _mm256_andnot_si256(a, b);
        assert_eq_m256i(got, _mm256_set1_epi8(2));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_avg_epu8() {
        let (a, b) = (_mm256_set1_epi8(3), _mm256_set1_epi8(9));
        let r = _mm256_avg_epu8(a, b);
        assert_eq_m256i(r, _mm256_set1_epi8(6));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_avg_epu16() {
        let (a, b) = (_mm256_set1_epi16(3), _mm256_set1_epi16(9));
        let r = _mm256_avg_epu16(a, b);
        assert_eq_m256i(r, _mm256_set1_epi16(6));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_blend_epi32() {
        let (a, b) = (_mm_set1_epi32(3), _mm_set1_epi32(9));
        let e = _mm_setr_epi32(9, 3, 3, 3);
        let r = _mm_blend_epi32::<0x01>(a, b);
        assert_eq_m128i(r, e);

        let r = _mm_blend_epi32::<0x0E>(b, a);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_blend_epi32() {
        let (a, b) = (_mm256_set1_epi32(3), _mm256_set1_epi32(9));
        let e = _mm256_setr_epi32(9, 3, 3, 3, 3, 3, 3, 3);
        let r = _mm256_blend_epi32::<0x01>(a, b);
        assert_eq_m256i(r, e);

        let e = _mm256_setr_epi32(3, 9, 3, 3, 3, 3, 3, 9);
        let r = _mm256_blend_epi32::<0x82>(a, b);
        assert_eq_m256i(r, e);

        let e = _mm256_setr_epi32(3, 3, 9, 9, 9, 9, 9, 3);
        let r = _mm256_blend_epi32::<0x7C>(a, b);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_blend_epi16() {
        let (a, b) = (_mm256_set1_epi16(3), _mm256_set1_epi16(9));
        let e = _mm256_setr_epi16(9, 3, 3, 3, 3, 3, 3, 3, 9, 3, 3, 3, 3, 3, 3, 3);
        let r = _mm256_blend_epi16::<0x01>(a, b);
        assert_eq_m256i(r, e);

        let r = _mm256_blend_epi16::<0xFE>(b, a);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_blendv_epi8() {
        let (a, b) = (_mm256_set1_epi8(4), _mm256_set1_epi8(2));
        let mask = _mm256_insert_epi8::<2>(_mm256_set1_epi8(0), -1);
        let e = _mm256_insert_epi8::<2>(_mm256_set1_epi8(4), 2);
        let r = _mm256_blendv_epi8(a, b, mask);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_broadcastb_epi8() {
        let a = _mm_insert_epi8::<0>(_mm_set1_epi8(0x00), 0x2a);
        let res = _mm_broadcastb_epi8(a);
        assert_eq_m128i(res, _mm_set1_epi8(0x2a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_broadcastb_epi8() {
        let a = _mm_insert_epi8::<0>(_mm_set1_epi8(0x00), 0x2a);
        let res = _mm256_broadcastb_epi8(a);
        assert_eq_m256i(res, _mm256_set1_epi8(0x2a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_broadcastd_epi32() {
        let a = _mm_setr_epi32(0x2a, 0x8000000, 0, 0);
        let res = _mm_broadcastd_epi32(a);
        assert_eq_m128i(res, _mm_set1_epi32(0x2a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_broadcastd_epi32() {
        let a = _mm_setr_epi32(0x2a, 0x8000000, 0, 0);
        let res = _mm256_broadcastd_epi32(a);
        assert_eq_m256i(res, _mm256_set1_epi32(0x2a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_broadcastq_epi64() {
        let a = _mm_setr_epi64x(0x1ffffffff, 0);
        let res = _mm_broadcastq_epi64(a);
        assert_eq_m128i(res, _mm_set1_epi64x(0x1ffffffff));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_broadcastq_epi64() {
        let a = _mm_setr_epi64x(0x1ffffffff, 0);
        let res = _mm256_broadcastq_epi64(a);
        assert_eq_m256i(res, _mm256_set1_epi64x(0x1ffffffff));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_broadcastsd_pd() {
        let a = _mm_setr_pd(6.28, 3.14);
        let res = _mm_broadcastsd_pd(a);
        assert_eq_m128d(res, _mm_set1_pd(6.28f64));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_broadcastsd_pd() {
        let a = _mm_setr_pd(6.28, 3.14);
        let res = _mm256_broadcastsd_pd(a);
        assert_eq_m256d(res, _mm256_set1_pd(6.28f64));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_broadcastsi128_si256() {
        let a = _mm_setr_epi64x(0x0987654321012334, 0x5678909876543210);
        let res = _mm256_broadcastsi128_si256(a);
        let retval = _mm256_setr_epi64x(
            0x0987654321012334,
            0x5678909876543210,
            0x0987654321012334,
            0x5678909876543210,
        );
        assert_eq_m256i(res, retval);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_broadcastss_ps() {
        let a = _mm_setr_ps(6.28, 3.14, 0.0, 0.0);
        let res = _mm_broadcastss_ps(a);
        assert_eq_m128(res, _mm_set1_ps(6.28f32));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_broadcastss_ps() {
        let a = _mm_setr_ps(6.28, 3.14, 0.0, 0.0);
        let res = _mm256_broadcastss_ps(a);
        assert_eq_m256(res, _mm256_set1_ps(6.28f32));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_broadcastw_epi16() {
        let a = _mm_insert_epi16::<0>(_mm_set1_epi16(0x2a), 0x22b);
        let res = _mm_broadcastw_epi16(a);
        assert_eq_m128i(res, _mm_set1_epi16(0x22b));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_broadcastw_epi16() {
        let a = _mm_insert_epi16::<0>(_mm_set1_epi16(0x2a), 0x22b);
        let res = _mm256_broadcastw_epi16(a);
        assert_eq_m256i(res, _mm256_set1_epi16(0x22b));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cmpeq_epi8() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        #[rustfmt::skip]
        let b = _mm256_setr_epi8(
            31, 30, 2, 28, 27, 26, 25, 24,
            23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0,
        );
        let r = _mm256_cmpeq_epi8(a, b);
        assert_eq_m256i(r, _mm256_insert_epi8::<2>(_mm256_set1_epi8(0), !0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cmpeq_epi16() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi16(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        #[rustfmt::skip]
        let b = _mm256_setr_epi16(
            15, 14, 2, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0,
        );
        let r = _mm256_cmpeq_epi16(a, b);
        assert_eq_m256i(r, _mm256_insert_epi16::<2>(_mm256_set1_epi16(0), !0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cmpeq_epi32() {
        let a = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm256_setr_epi32(7, 6, 2, 4, 3, 2, 1, 0);
        let r = _mm256_cmpeq_epi32(a, b);
        let e = _mm256_set1_epi32(0);
        let e = _mm256_insert_epi32::<2>(e, !0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cmpeq_epi64() {
        let a = _mm256_setr_epi64x(0, 1, 2, 3);
        let b = _mm256_setr_epi64x(3, 2, 2, 0);
        let r = _mm256_cmpeq_epi64(a, b);
        assert_eq_m256i(r, _mm256_insert_epi64::<2>(_mm256_set1_epi64x(0), !0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cmpgt_epi8() {
        let a = _mm256_insert_epi8::<0>(_mm256_set1_epi8(0), 5);
        let b = _mm256_set1_epi8(0);
        let r = _mm256_cmpgt_epi8(a, b);
        assert_eq_m256i(r, _mm256_insert_epi8::<0>(_mm256_set1_epi8(0), !0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cmpgt_epi16() {
        let a = _mm256_insert_epi16::<0>(_mm256_set1_epi16(0), 5);
        let b = _mm256_set1_epi16(0);
        let r = _mm256_cmpgt_epi16(a, b);
        assert_eq_m256i(r, _mm256_insert_epi16::<0>(_mm256_set1_epi16(0), !0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cmpgt_epi32() {
        let a = _mm256_insert_epi32::<0>(_mm256_set1_epi32(0), 5);
        let b = _mm256_set1_epi32(0);
        let r = _mm256_cmpgt_epi32(a, b);
        assert_eq_m256i(r, _mm256_insert_epi32::<0>(_mm256_set1_epi32(0), !0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cmpgt_epi64() {
        let a = _mm256_insert_epi64::<0>(_mm256_set1_epi64x(0), 5);
        let b = _mm256_set1_epi64x(0);
        let r = _mm256_cmpgt_epi64(a, b);
        assert_eq_m256i(r, _mm256_insert_epi64::<0>(_mm256_set1_epi64x(0), !0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cvtepi8_epi16() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            0, 0, -1, 1, -2, 2, -3, 3,
            -4, 4, -5, 5, -6, 6, -7, 7,
        );
        #[rustfmt::skip]
        let r = _mm256_setr_epi16(
            0, 0, -1, 1, -2, 2, -3, 3,
            -4, 4, -5, 5, -6, 6, -7, 7,
        );
        assert_eq_m256i(r, _mm256_cvtepi8_epi16(a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cvtepi8_epi32() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            0, 0, -1, 1, -2, 2, -3, 3,
            -4, 4, -5, 5, -6, 6, -7, 7,
        );
        let r = _mm256_setr_epi32(0, 0, -1, 1, -2, 2, -3, 3);
        assert_eq_m256i(r, _mm256_cvtepi8_epi32(a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cvtepi8_epi64() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            0, 0, -1, 1, -2, 2, -3, 3,
            -4, 4, -5, 5, -6, 6, -7, 7,
        );
        let r = _mm256_setr_epi64x(0, 0, -1, 1);
        assert_eq_m256i(r, _mm256_cvtepi8_epi64(a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cvtepi16_epi32() {
        let a = _mm_setr_epi16(0, 0, -1, 1, -2, 2, -3, 3);
        let r = _mm256_setr_epi32(0, 0, -1, 1, -2, 2, -3, 3);
        assert_eq_m256i(r, _mm256_cvtepi16_epi32(a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cvtepi16_epi64() {
        let a = _mm_setr_epi16(0, 0, -1, 1, -2, 2, -3, 3);
        let r = _mm256_setr_epi64x(0, 0, -1, 1);
        assert_eq_m256i(r, _mm256_cvtepi16_epi64(a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cvtepi32_epi64() {
        let a = _mm_setr_epi32(0, 0, -1, 1);
        let r = _mm256_setr_epi64x(0, 0, -1, 1);
        assert_eq_m256i(r, _mm256_cvtepi32_epi64(a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cvtepu16_epi32() {
        let a = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq_m256i(r, _mm256_cvtepu16_epi32(a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cvtepu16_epi64() {
        let a = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm256_setr_epi64x(0, 1, 2, 3);
        assert_eq_m256i(r, _mm256_cvtepu16_epi64(a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cvtepu32_epi64() {
        let a = _mm_setr_epi32(0, 1, 2, 3);
        let r = _mm256_setr_epi64x(0, 1, 2, 3);
        assert_eq_m256i(r, _mm256_cvtepu32_epi64(a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cvtepu8_epi16() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        #[rustfmt::skip]
        let r = _mm256_setr_epi16(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        assert_eq_m256i(r, _mm256_cvtepu8_epi16(a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cvtepu8_epi32() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        let r = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq_m256i(r, _mm256_cvtepu8_epi32(a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cvtepu8_epi64() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        let r = _mm256_setr_epi64x(0, 1, 2, 3);
        assert_eq_m256i(r, _mm256_cvtepu8_epi64(a));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_extracti128_si256() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let r = _mm256_extracti128_si256::<1>(a);
        let e = _mm_setr_epi64x(3, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_hadd_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_hadd_epi16(a, b);
        let e = _mm256_setr_epi16(4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_hadd_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(4);
        let r = _mm256_hadd_epi32(a, b);
        let e = _mm256_setr_epi32(4, 4, 8, 8, 4, 4, 8, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_hadds_epi16() {
        let a = _mm256_set1_epi16(2);
        let a = _mm256_insert_epi16::<0>(a, 0x7fff);
        let a = _mm256_insert_epi16::<1>(a, 1);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_hadds_epi16(a, b);
        #[rustfmt::skip]
        let e = _mm256_setr_epi16(
            0x7FFF, 4, 4, 4, 8, 8, 8, 8,
            4, 4, 4, 4, 8, 8, 8, 8,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_hsub_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_hsub_epi16(a, b);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_hsub_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(4);
        let r = _mm256_hsub_epi32(a, b);
        let e = _mm256_set1_epi32(0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_hsubs_epi16() {
        let a = _mm256_set1_epi16(2);
        let a = _mm256_insert_epi16::<0>(a, 0x7fff);
        let a = _mm256_insert_epi16::<1>(a, -1);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_hsubs_epi16(a, b);
        let e = _mm256_insert_epi16::<0>(_mm256_set1_epi16(0), 0x7FFF);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_madd_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_madd_epi16(a, b);
        let e = _mm256_set1_epi32(16);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_inserti128_si256() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let b = _mm_setr_epi64x(7, 8);
        let r = _mm256_inserti128_si256::<1>(a, b);
        let e = _mm256_setr_epi64x(1, 2, 7, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_maddubs_epi16() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_maddubs_epi16(a, b);
        let e = _mm256_set1_epi16(16);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_maskload_epi32() {
        let nums = [1, 2, 3, 4];
        let a = &nums as *const i32;
        let mask = _mm_setr_epi32(-1, 0, 0, -1);
        let r = _mm_maskload_epi32(a, mask);
        let e = _mm_setr_epi32(1, 0, 0, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_maskload_epi32() {
        let nums = [1, 2, 3, 4, 5, 6, 7, 8];
        let a = &nums as *const i32;
        let mask = _mm256_setr_epi32(-1, 0, 0, -1, 0, -1, -1, 0);
        let r = _mm256_maskload_epi32(a, mask);
        let e = _mm256_setr_epi32(1, 0, 0, 4, 0, 6, 7, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_maskload_epi64() {
        let nums = [1_i64, 2_i64];
        let a = &nums as *const i64;
        let mask = _mm_setr_epi64x(0, -1);
        let r = _mm_maskload_epi64(a, mask);
        let e = _mm_setr_epi64x(0, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_maskload_epi64() {
        let nums = [1_i64, 2_i64, 3_i64, 4_i64];
        let a = &nums as *const i64;
        let mask = _mm256_setr_epi64x(0, -1, -1, 0);
        let r = _mm256_maskload_epi64(a, mask);
        let e = _mm256_setr_epi64x(0, 2, 3, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_maskstore_epi32() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let mut arr = [-1, -1, -1, -1];
        let mask = _mm_setr_epi32(-1, 0, 0, -1);
        _mm_maskstore_epi32(arr.as_mut_ptr(), mask, a);
        let e = [1, -1, -1, 4];
        assert_eq!(arr, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_maskstore_epi32() {
        let a = _mm256_setr_epi32(1, 0x6d726f, 3, 42, 0x777161, 6, 7, 8);
        let mut arr = [-1, -1, -1, 0x776173, -1, 0x68657265, -1, -1];
        let mask = _mm256_setr_epi32(-1, 0, 0, -1, 0, -1, -1, 0);
        _mm256_maskstore_epi32(arr.as_mut_ptr(), mask, a);
        let e = [1, -1, -1, 42, -1, 6, 7, -1];
        assert_eq!(arr, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_maskstore_epi64() {
        let a = _mm_setr_epi64x(1_i64, 2_i64);
        let mut arr = [-1_i64, -1_i64];
        let mask = _mm_setr_epi64x(0, -1);
        _mm_maskstore_epi64(arr.as_mut_ptr(), mask, a);
        let e = [-1, 2];
        assert_eq!(arr, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_maskstore_epi64() {
        let a = _mm256_setr_epi64x(1_i64, 2_i64, 3_i64, 4_i64);
        let mut arr = [-1_i64, -1_i64, -1_i64, -1_i64];
        let mask = _mm256_setr_epi64x(0, -1, -1, 0);
        _mm256_maskstore_epi64(arr.as_mut_ptr(), mask, a);
        let e = [-1, 2, 3, -1];
        assert_eq!(arr, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_max_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_max_epi16(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_max_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(4);
        let r = _mm256_max_epi32(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_max_epi8() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_max_epi8(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_max_epu16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_max_epu16(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_max_epu32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(4);
        let r = _mm256_max_epu32(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_max_epu8() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_max_epu8(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_min_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_min_epi16(a, b);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_min_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(4);
        let r = _mm256_min_epi32(a, b);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_min_epi8() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_min_epi8(a, b);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_min_epu16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_min_epu16(a, b);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_min_epu32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(4);
        let r = _mm256_min_epu32(a, b);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_min_epu8() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_min_epu8(a, b);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_movemask_epi8() {
        let a = _mm256_set1_epi8(-1);
        let r = _mm256_movemask_epi8(a);
        let e = -1;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mpsadbw_epu8() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_mpsadbw_epu8::<0>(a, b);
        let e = _mm256_set1_epi16(8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mul_epi32() {
        let a = _mm256_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2);
        let b = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm256_mul_epi32(a, b);
        let e = _mm256_setr_epi64x(0, 0, 10, 14);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mul_epu32() {
        let a = _mm256_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2);
        let b = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm256_mul_epu32(a, b);
        let e = _mm256_setr_epi64x(0, 0, 10, 14);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mulhi_epi16() {
        let a = _mm256_set1_epi16(6535);
        let b = _mm256_set1_epi16(6535);
        let r = _mm256_mulhi_epi16(a, b);
        let e = _mm256_set1_epi16(651);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mulhi_epu16() {
        let a = _mm256_set1_epi16(6535);
        let b = _mm256_set1_epi16(6535);
        let r = _mm256_mulhi_epu16(a, b);
        let e = _mm256_set1_epi16(651);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mullo_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_mullo_epi16(a, b);
        let e = _mm256_set1_epi16(8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mullo_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(4);
        let r = _mm256_mullo_epi32(a, b);
        let e = _mm256_set1_epi32(8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mulhrs_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_mullo_epi16(a, b);
        let e = _mm256_set1_epi16(8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_or_si256() {
        let a = _mm256_set1_epi8(-1);
        let b = _mm256_set1_epi8(0);
        let r = _mm256_or_si256(a, b);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_packs_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_packs_epi16(a, b);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4,
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4,
        );

        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_packs_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(4);
        let r = _mm256_packs_epi32(a, b);
        let e = _mm256_setr_epi16(2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4);

        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_packus_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_packus_epi16(a, b);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4,
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4,
        );

        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_packus_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(4);
        let r = _mm256_packus_epi32(a, b);
        let e = _mm256_setr_epi16(2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4);

        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sad_epu8() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_sad_epu8(a, b);
        let e = _mm256_set1_epi64x(16);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_shufflehi_epi16() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi16(
            0, 1, 2, 3, 11, 22, 33, 44,
            4, 5, 6, 7, 55, 66, 77, 88,
        );
        #[rustfmt::skip]
        let e = _mm256_setr_epi16(
            0, 1, 2, 3, 44, 22, 22, 11,
            4, 5, 6, 7, 88, 66, 66, 55,
        );
        let r = _mm256_shufflehi_epi16::<0b00_01_01_11>(a);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_shufflelo_epi16() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi16(
            11, 22, 33, 44, 0, 1, 2, 3,
            55, 66, 77, 88, 4, 5, 6, 7,
        );
        #[rustfmt::skip]
        let e = _mm256_setr_epi16(
            44, 22, 22, 11, 0, 1, 2, 3,
            88, 66, 66, 55, 4, 5, 6, 7,
        );
        let r = _mm256_shufflelo_epi16::<0b00_01_01_11>(a);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sign_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(-1);
        let r = _mm256_sign_epi16(a, b);
        let e = _mm256_set1_epi16(-2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sign_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(-1);
        let r = _mm256_sign_epi32(a, b);
        let e = _mm256_set1_epi32(-2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sign_epi8() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(-1);
        let r = _mm256_sign_epi8(a, b);
        let e = _mm256_set1_epi8(-2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sll_epi16() {
        let a = _mm256_set1_epi16(0xFF);
        let b = _mm_insert_epi16::<0>(_mm_set1_epi16(0), 4);
        let r = _mm256_sll_epi16(a, b);
        assert_eq_m256i(r, _mm256_set1_epi16(0xFF0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sll_epi32() {
        let a = _mm256_set1_epi32(0xFFFF);
        let b = _mm_insert_epi32::<0>(_mm_set1_epi32(0), 4);
        let r = _mm256_sll_epi32(a, b);
        assert_eq_m256i(r, _mm256_set1_epi32(0xFFFF0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sll_epi64() {
        let a = _mm256_set1_epi64x(0xFFFFFFFF);
        let b = _mm_insert_epi64::<0>(_mm_set1_epi64x(0), 4);
        let r = _mm256_sll_epi64(a, b);
        assert_eq_m256i(r, _mm256_set1_epi64x(0xFFFFFFFF0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_slli_epi16() {
        assert_eq_m256i(
            _mm256_slli_epi16::<4>(_mm256_set1_epi16(0xFF)),
            _mm256_set1_epi16(0xFF0),
        );
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_slli_epi32() {
        assert_eq_m256i(
            _mm256_slli_epi32::<4>(_mm256_set1_epi32(0xFFFF)),
            _mm256_set1_epi32(0xFFFF0),
        );
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_slli_epi64() {
        assert_eq_m256i(
            _mm256_slli_epi64::<4>(_mm256_set1_epi64x(0xFFFFFFFF)),
            _mm256_set1_epi64x(0xFFFFFFFF0),
        );
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_slli_si256() {
        let a = _mm256_set1_epi64x(0xFFFFFFFF);
        let r = _mm256_slli_si256::<3>(a);
        assert_eq_m256i(r, _mm256_set1_epi64x(0xFFFFFFFF000000));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_sllv_epi32() {
        let a = _mm_set1_epi32(2);
        let b = _mm_set1_epi32(1);
        let r = _mm_sllv_epi32(a, b);
        let e = _mm_set1_epi32(4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sllv_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(1);
        let r = _mm256_sllv_epi32(a, b);
        let e = _mm256_set1_epi32(4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_sllv_epi64() {
        let a = _mm_set1_epi64x(2);
        let b = _mm_set1_epi64x(1);
        let r = _mm_sllv_epi64(a, b);
        let e = _mm_set1_epi64x(4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sllv_epi64() {
        let a = _mm256_set1_epi64x(2);
        let b = _mm256_set1_epi64x(1);
        let r = _mm256_sllv_epi64(a, b);
        let e = _mm256_set1_epi64x(4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sra_epi16() {
        let a = _mm256_set1_epi16(-1);
        let b = _mm_setr_epi16(1, 0, 0, 0, 0, 0, 0, 0);
        let r = _mm256_sra_epi16(a, b);
        assert_eq_m256i(r, _mm256_set1_epi16(-1));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sra_epi32() {
        let a = _mm256_set1_epi32(-1);
        let b = _mm_insert_epi32::<0>(_mm_set1_epi32(0), 1);
        let r = _mm256_sra_epi32(a, b);
        assert_eq_m256i(r, _mm256_set1_epi32(-1));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_srai_epi16() {
        assert_eq_m256i(
            _mm256_srai_epi16::<1>(_mm256_set1_epi16(-1)),
            _mm256_set1_epi16(-1),
        );
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_srai_epi32() {
        assert_eq_m256i(
            _mm256_srai_epi32::<1>(_mm256_set1_epi32(-1)),
            _mm256_set1_epi32(-1),
        );
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_srav_epi32() {
        let a = _mm_set1_epi32(4);
        let count = _mm_set1_epi32(1);
        let r = _mm_srav_epi32(a, count);
        let e = _mm_set1_epi32(2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_srav_epi32() {
        let a = _mm256_set1_epi32(4);
        let count = _mm256_set1_epi32(1);
        let r = _mm256_srav_epi32(a, count);
        let e = _mm256_set1_epi32(2);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_srli_si256() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm256_srli_si256::<3>(a);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            4, 5, 6, 7, 8, 9, 10, 11,
            12, 13, 14, 15, 16, 0, 0, 0,
            20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 0, 0, 0,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_srl_epi16() {
        let a = _mm256_set1_epi16(0xFF);
        let b = _mm_insert_epi16::<0>(_mm_set1_epi16(0), 4);
        let r = _mm256_srl_epi16(a, b);
        assert_eq_m256i(r, _mm256_set1_epi16(0xF));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_srl_epi32() {
        let a = _mm256_set1_epi32(0xFFFF);
        let b = _mm_insert_epi32::<0>(_mm_set1_epi32(0), 4);
        let r = _mm256_srl_epi32(a, b);
        assert_eq_m256i(r, _mm256_set1_epi32(0xFFF));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_srl_epi64() {
        let a = _mm256_set1_epi64x(0xFFFFFFFF);
        let b = _mm_setr_epi64x(4, 0);
        let r = _mm256_srl_epi64(a, b);
        assert_eq_m256i(r, _mm256_set1_epi64x(0xFFFFFFF));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_srli_epi16() {
        assert_eq_m256i(
            _mm256_srli_epi16::<4>(_mm256_set1_epi16(0xFF)),
            _mm256_set1_epi16(0xF),
        );
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_srli_epi32() {
        assert_eq_m256i(
            _mm256_srli_epi32::<4>(_mm256_set1_epi32(0xFFFF)),
            _mm256_set1_epi32(0xFFF),
        );
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_srli_epi64() {
        assert_eq_m256i(
            _mm256_srli_epi64::<4>(_mm256_set1_epi64x(0xFFFFFFFF)),
            _mm256_set1_epi64x(0xFFFFFFF),
        );
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_srlv_epi32() {
        let a = _mm_set1_epi32(2);
        let count = _mm_set1_epi32(1);
        let r = _mm_srlv_epi32(a, count);
        let e = _mm_set1_epi32(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_srlv_epi32() {
        let a = _mm256_set1_epi32(2);
        let count = _mm256_set1_epi32(1);
        let r = _mm256_srlv_epi32(a, count);
        let e = _mm256_set1_epi32(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_srlv_epi64() {
        let a = _mm_set1_epi64x(2);
        let count = _mm_set1_epi64x(1);
        let r = _mm_srlv_epi64(a, count);
        let e = _mm_set1_epi64x(1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_srlv_epi64() {
        let a = _mm256_set1_epi64x(2);
        let count = _mm256_set1_epi64x(1);
        let r = _mm256_srlv_epi64(a, count);
        let e = _mm256_set1_epi64x(1);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sub_epi16() {
        let a = _mm256_set1_epi16(4);
        let b = _mm256_set1_epi16(2);
        let r = _mm256_sub_epi16(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sub_epi32() {
        let a = _mm256_set1_epi32(4);
        let b = _mm256_set1_epi32(2);
        let r = _mm256_sub_epi32(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sub_epi64() {
        let a = _mm256_set1_epi64x(4);
        let b = _mm256_set1_epi64x(2);
        let r = _mm256_sub_epi64(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_sub_epi8() {
        let a = _mm256_set1_epi8(4);
        let b = _mm256_set1_epi8(2);
        let r = _mm256_sub_epi8(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_subs_epi16() {
        let a = _mm256_set1_epi16(4);
        let b = _mm256_set1_epi16(2);
        let r = _mm256_subs_epi16(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_subs_epi8() {
        let a = _mm256_set1_epi8(4);
        let b = _mm256_set1_epi8(2);
        let r = _mm256_subs_epi8(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_subs_epu16() {
        let a = _mm256_set1_epi16(4);
        let b = _mm256_set1_epi16(2);
        let r = _mm256_subs_epu16(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_subs_epu8() {
        let a = _mm256_set1_epi8(4);
        let b = _mm256_set1_epi8(2);
        let r = _mm256_subs_epu8(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_xor_si256() {
        let a = _mm256_set1_epi8(5);
        let b = _mm256_set1_epi8(3);
        let r = _mm256_xor_si256(a, b);
        assert_eq_m256i(r, _mm256_set1_epi8(6));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_alignr_epi8() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        #[rustfmt::skip]
        let b = _mm256_setr_epi8(
            -1, -2, -3, -4, -5, -6, -7, -8,
            -9, -10, -11, -12, -13, -14, -15, -16,
            -17, -18, -19, -20, -21, -22, -23, -24,
            -25, -26, -27, -28, -29, -30, -31, -32,
        );
        let r = _mm256_alignr_epi8::<33>(a, b);
        assert_eq_m256i(r, _mm256_set1_epi8(0));

        let r = _mm256_alignr_epi8::<17>(a, b);
        #[rustfmt::skip]
        let expected = _mm256_setr_epi8(
            2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 0,
            18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 0,
        );
        assert_eq_m256i(r, expected);

        let r = _mm256_alignr_epi8::<4>(a, b);
        #[rustfmt::skip]
        let expected = _mm256_setr_epi8(
            -5, -6, -7, -8, -9, -10, -11, -12,
            -13, -14, -15, -16, 1, 2, 3, 4,
            -21, -22, -23, -24, -25, -26, -27, -28,
            -29, -30, -31, -32, 17, 18, 19, 20,
        );
        assert_eq_m256i(r, expected);

        #[rustfmt::skip]
        let expected = _mm256_setr_epi8(
            -1, -2, -3, -4, -5, -6, -7, -8,
            -9, -10, -11, -12, -13, -14, -15, -16, -17,
            -18, -19, -20, -21, -22, -23, -24, -25,
            -26, -27, -28, -29, -30, -31, -32,
        );
        let r = _mm256_alignr_epi8::<16>(a, b);
        assert_eq_m256i(r, expected);

        let r = _mm256_alignr_epi8::<15>(a, b);
        #[rustfmt::skip]
        let expected = _mm256_setr_epi8(
            -16, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            -32, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        assert_eq_m256i(r, expected);

        let r = _mm256_alignr_epi8::<0>(a, b);
        assert_eq_m256i(r, b);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_shuffle_epi8() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        #[rustfmt::skip]
        let b = _mm256_setr_epi8(
            4, 128u8 as i8, 4, 3, 24, 12, 6, 19,
            12, 5, 5, 10, 4, 1, 8, 0,
            4, 128u8 as i8, 4, 3, 24, 12, 6, 19,
            12, 5, 5, 10, 4, 1, 8, 0,
        );
        #[rustfmt::skip]
        let expected = _mm256_setr_epi8(
            5, 0, 5, 4, 9, 13, 7, 4,
            13, 6, 6, 11, 5, 2, 9, 1,
            21, 0, 21, 20, 25, 29, 23, 20,
            29, 22, 22, 27, 21, 18, 25, 17,
        );
        let r = _mm256_shuffle_epi8(a, b);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_permutevar8x32_epi32() {
        let a = _mm256_setr_epi32(100, 200, 300, 400, 500, 600, 700, 800);
        let b = _mm256_setr_epi32(5, 0, 5, 1, 7, 6, 3, 4);
        let expected = _mm256_setr_epi32(600, 100, 600, 200, 800, 700, 400, 500);
        let r = _mm256_permutevar8x32_epi32(a, b);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_permute4x64_epi64() {
        let a = _mm256_setr_epi64x(100, 200, 300, 400);
        let expected = _mm256_setr_epi64x(400, 100, 200, 100);
        let r = _mm256_permute4x64_epi64::<0b00010011>(a);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_permute2x128_si256() {
        let a = _mm256_setr_epi64x(100, 200, 500, 600);
        let b = _mm256_setr_epi64x(300, 400, 700, 800);
        let r = _mm256_permute2x128_si256::<0b00_01_00_11>(a, b);
        let e = _mm256_setr_epi64x(700, 800, 500, 600);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_permute4x64_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let r = _mm256_permute4x64_pd::<0b00_01_00_11>(a);
        let e = _mm256_setr_pd(4., 1., 2., 1.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_permutevar8x32_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm256_setr_epi32(5, 0, 5, 1, 7, 6, 3, 4);
        let r = _mm256_permutevar8x32_ps(a, b);
        let e = _mm256_setr_ps(6., 1., 6., 2., 8., 7., 4., 5.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_i32gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = _mm_i32gather_epi32::<4>(arr.as_ptr(), _mm_setr_epi32(0, 16, 32, 48));
        assert_eq_m128i(r, _mm_setr_epi32(0, 16, 32, 48));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_mask_i32gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = _mm_mask_i32gather_epi32::<4>(
            _mm_set1_epi32(256),
            arr.as_ptr(),
            _mm_setr_epi32(0, 16, 64, 96),
            _mm_setr_epi32(-1, -1, -1, 0),
        );
        assert_eq_m128i(r, _mm_setr_epi32(0, 16, 64, 256));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_i32gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r =
            _mm256_i32gather_epi32::<4>(arr.as_ptr(), _mm256_setr_epi32(0, 16, 32, 48, 1, 2, 3, 4));
        assert_eq_m256i(r, _mm256_setr_epi32(0, 16, 32, 48, 1, 2, 3, 4));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mask_i32gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = _mm256_mask_i32gather_epi32::<4>(
            _mm256_set1_epi32(256),
            arr.as_ptr(),
            _mm256_setr_epi32(0, 16, 64, 96, 0, 0, 0, 0),
            _mm256_setr_epi32(-1, -1, -1, 0, 0, 0, 0, 0),
        );
        assert_eq_m256i(r, _mm256_setr_epi32(0, 16, 64, 256, 256, 256, 256, 256));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_i32gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r = _mm_i32gather_ps::<4>(arr.as_ptr(), _mm_setr_epi32(0, 16, 32, 48));
        assert_eq_m128(r, _mm_setr_ps(0.0, 16.0, 32.0, 48.0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_mask_i32gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r = _mm_mask_i32gather_ps::<4>(
            _mm_set1_ps(256.0),
            arr.as_ptr(),
            _mm_setr_epi32(0, 16, 64, 96),
            _mm_setr_ps(-1.0, -1.0, -1.0, 0.0),
        );
        assert_eq_m128(r, _mm_setr_ps(0.0, 16.0, 64.0, 256.0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_i32gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r =
            _mm256_i32gather_ps::<4>(arr.as_ptr(), _mm256_setr_epi32(0, 16, 32, 48, 1, 2, 3, 4));
        assert_eq_m256(r, _mm256_setr_ps(0.0, 16.0, 32.0, 48.0, 1.0, 2.0, 3.0, 4.0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mask_i32gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r = _mm256_mask_i32gather_ps::<4>(
            _mm256_set1_ps(256.0),
            arr.as_ptr(),
            _mm256_setr_epi32(0, 16, 64, 96, 0, 0, 0, 0),
            _mm256_setr_ps(-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        );
        assert_eq_m256(
            r,
            _mm256_setr_ps(0.0, 16.0, 64.0, 256.0, 256.0, 256.0, 256.0, 256.0),
        );
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_i32gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm_i32gather_epi64::<8>(arr.as_ptr(), _mm_setr_epi32(0, 16, 0, 0));
        assert_eq_m128i(r, _mm_setr_epi64x(0, 16));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_mask_i32gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm_mask_i32gather_epi64::<8>(
            _mm_set1_epi64x(256),
            arr.as_ptr(),
            _mm_setr_epi32(16, 16, 16, 16),
            _mm_setr_epi64x(-1, 0),
        );
        assert_eq_m128i(r, _mm_setr_epi64x(16, 256));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_i32gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm256_i32gather_epi64::<8>(arr.as_ptr(), _mm_setr_epi32(0, 16, 32, 48));
        assert_eq_m256i(r, _mm256_setr_epi64x(0, 16, 32, 48));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mask_i32gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm256_mask_i32gather_epi64::<8>(
            _mm256_set1_epi64x(256),
            arr.as_ptr(),
            _mm_setr_epi32(0, 16, 64, 96),
            _mm256_setr_epi64x(-1, -1, -1, 0),
        );
        assert_eq_m256i(r, _mm256_setr_epi64x(0, 16, 64, 256));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_i32gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm_i32gather_pd::<8>(arr.as_ptr(), _mm_setr_epi32(0, 16, 0, 0));
        assert_eq_m128d(r, _mm_setr_pd(0.0, 16.0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_mask_i32gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm_mask_i32gather_pd::<8>(
            _mm_set1_pd(256.0),
            arr.as_ptr(),
            _mm_setr_epi32(16, 16, 16, 16),
            _mm_setr_pd(-1.0, 0.0),
        );
        assert_eq_m128d(r, _mm_setr_pd(16.0, 256.0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_i32gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm256_i32gather_pd::<8>(arr.as_ptr(), _mm_setr_epi32(0, 16, 32, 48));
        assert_eq_m256d(r, _mm256_setr_pd(0.0, 16.0, 32.0, 48.0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mask_i32gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm256_mask_i32gather_pd::<8>(
            _mm256_set1_pd(256.0),
            arr.as_ptr(),
            _mm_setr_epi32(0, 16, 64, 96),
            _mm256_setr_pd(-1.0, -1.0, -1.0, 0.0),
        );
        assert_eq_m256d(r, _mm256_setr_pd(0.0, 16.0, 64.0, 256.0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_i64gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = _mm_i64gather_epi32::<4>(arr.as_ptr(), _mm_setr_epi64x(0, 16));
        assert_eq_m128i(r, _mm_setr_epi32(0, 16, 0, 0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_mask_i64gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = _mm_mask_i64gather_epi32::<4>(
            _mm_set1_epi32(256),
            arr.as_ptr(),
            _mm_setr_epi64x(0, 16),
            _mm_setr_epi32(-1, 0, -1, 0),
        );
        assert_eq_m128i(r, _mm_setr_epi32(0, 256, 0, 0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_i64gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = _mm256_i64gather_epi32::<4>(arr.as_ptr(), _mm256_setr_epi64x(0, 16, 32, 48));
        assert_eq_m128i(r, _mm_setr_epi32(0, 16, 32, 48));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mask_i64gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = _mm256_mask_i64gather_epi32::<4>(
            _mm_set1_epi32(256),
            arr.as_ptr(),
            _mm256_setr_epi64x(0, 16, 64, 96),
            _mm_setr_epi32(-1, -1, -1, 0),
        );
        assert_eq_m128i(r, _mm_setr_epi32(0, 16, 64, 256));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_i64gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r = _mm_i64gather_ps::<4>(arr.as_ptr(), _mm_setr_epi64x(0, 16));
        assert_eq_m128(r, _mm_setr_ps(0.0, 16.0, 0.0, 0.0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_mask_i64gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r = _mm_mask_i64gather_ps::<4>(
            _mm_set1_ps(256.0),
            arr.as_ptr(),
            _mm_setr_epi64x(0, 16),
            _mm_setr_ps(-1.0, 0.0, -1.0, 0.0),
        );
        assert_eq_m128(r, _mm_setr_ps(0.0, 256.0, 0.0, 0.0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_i64gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r = _mm256_i64gather_ps::<4>(arr.as_ptr(), _mm256_setr_epi64x(0, 16, 32, 48));
        assert_eq_m128(r, _mm_setr_ps(0.0, 16.0, 32.0, 48.0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mask_i64gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r = _mm256_mask_i64gather_ps::<4>(
            _mm_set1_ps(256.0),
            arr.as_ptr(),
            _mm256_setr_epi64x(0, 16, 64, 96),
            _mm_setr_ps(-1.0, -1.0, -1.0, 0.0),
        );
        assert_eq_m128(r, _mm_setr_ps(0.0, 16.0, 64.0, 256.0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_i64gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm_i64gather_epi64::<8>(arr.as_ptr(), _mm_setr_epi64x(0, 16));
        assert_eq_m128i(r, _mm_setr_epi64x(0, 16));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_mask_i64gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm_mask_i64gather_epi64::<8>(
            _mm_set1_epi64x(256),
            arr.as_ptr(),
            _mm_setr_epi64x(16, 16),
            _mm_setr_epi64x(-1, 0),
        );
        assert_eq_m128i(r, _mm_setr_epi64x(16, 256));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_i64gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm256_i64gather_epi64::<8>(arr.as_ptr(), _mm256_setr_epi64x(0, 16, 32, 48));
        assert_eq_m256i(r, _mm256_setr_epi64x(0, 16, 32, 48));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mask_i64gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm256_mask_i64gather_epi64::<8>(
            _mm256_set1_epi64x(256),
            arr.as_ptr(),
            _mm256_setr_epi64x(0, 16, 64, 96),
            _mm256_setr_epi64x(-1, -1, -1, 0),
        );
        assert_eq_m256i(r, _mm256_setr_epi64x(0, 16, 64, 256));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_i64gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm_i64gather_pd::<8>(arr.as_ptr(), _mm_setr_epi64x(0, 16));
        assert_eq_m128d(r, _mm_setr_pd(0.0, 16.0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm_mask_i64gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm_mask_i64gather_pd::<8>(
            _mm_set1_pd(256.0),
            arr.as_ptr(),
            _mm_setr_epi64x(16, 16),
            _mm_setr_pd(-1.0, 0.0),
        );
        assert_eq_m128d(r, _mm_setr_pd(16.0, 256.0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_i64gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm256_i64gather_pd::<8>(arr.as_ptr(), _mm256_setr_epi64x(0, 16, 32, 48));
        assert_eq_m256d(r, _mm256_setr_pd(0.0, 16.0, 32.0, 48.0));
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_mask_i64gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm256_mask_i64gather_pd::<8>(
            _mm256_set1_pd(256.0),
            arr.as_ptr(),
            _mm256_setr_epi64x(0, 16, 64, 96),
            _mm256_setr_pd(-1.0, -1.0, -1.0, 0.0),
        );
        assert_eq_m256d(r, _mm256_setr_pd(0.0, 16.0, 64.0, 256.0));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_extract_epi8() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            -1, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31
        );
        let r1 = _mm256_extract_epi8::<0>(a);
        let r2 = _mm256_extract_epi8::<3>(a);
        assert_eq!(r1, 0xFF);
        assert_eq!(r2, 3);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_extract_epi16() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi16(
            -1, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        let r1 = _mm256_extract_epi16::<0>(a);
        let r2 = _mm256_extract_epi16::<3>(a);
        assert_eq!(r1, 0xFFFF);
        assert_eq!(r2, 3);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_extract_epi32() {
        let a = _mm256_setr_epi32(-1, 1, 2, 3, 4, 5, 6, 7);
        let r1 = _mm256_extract_epi32::<0>(a);
        let r2 = _mm256_extract_epi32::<3>(a);
        assert_eq!(r1, -1);
        assert_eq!(r2, 3);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cvtsd_f64() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let r = _mm256_cvtsd_f64(a);
        assert_eq!(r, 1.);
    }

    #[simd_test(enable = "avx2")]
    unsafe fn test_mm256_cvtsi256_si32() {
        let a = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm256_cvtsi256_si32(a);
        assert_eq!(r, 1);
    }
}
