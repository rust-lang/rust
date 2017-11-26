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

use simd_llvm::simd_cast;
use simd_llvm::{simd_shuffle2, simd_shuffle4, simd_shuffle8};
use simd_llvm::{simd_shuffle16, simd_shuffle32};

use v256::*;
use v128::*;
use x86::{__m128i, __m256i};

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Computes the absolute values of packed 32-bit integers in `a`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpabsd))]
pub unsafe fn _mm256_abs_epi32(a: i32x8) -> u32x8 {
    pabsd(a)
}

/// Computes the absolute values of packed 16-bit integers in `a`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpabsw))]
pub unsafe fn _mm256_abs_epi16(a: i16x16) -> u16x16 {
    pabsw(a)
}

/// Computes the absolute values of packed 8-bit integers in `a`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpabsb))]
pub unsafe fn _mm256_abs_epi8(a: i8x32) -> u8x32 {
    pabsb(a)
}

/// Add packed 64-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpaddq))]
pub unsafe fn _mm256_add_epi64(a: i64x4, b: i64x4) -> i64x4 {
    a + b
}

/// Add packed 32-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm256_add_epi32(a: i32x8, b: i32x8) -> i32x8 {
    a + b
}

/// Add packed 16-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpaddw))]
pub unsafe fn _mm256_add_epi16(a: i16x16, b: i16x16) -> i16x16 {
    a + b
}

/// Add packed 8-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpaddb))]
pub unsafe fn _mm256_add_epi8(a: i8x32, b: i8x32) -> i8x32 {
    a + b
}

/// Add packed 8-bit integers in `a` and `b` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub unsafe fn _mm256_adds_epi8(a: i8x32, b: i8x32) -> i8x32 {
    paddsb(a, b)
}

/// Add packed 16-bit integers in `a` and `b` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub unsafe fn _mm256_adds_epi16(a: i16x16, b: i16x16) -> i16x16 {
    paddsw(a, b)
}

/// Add packed unsigned 8-bit integers in `a` and `b` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub unsafe fn _mm256_adds_epu8(a: u8x32, b: u8x32) -> u8x32 {
    paddusb(a, b)
}

/// Add packed unsigned 16-bit integers in `a` and `b` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub unsafe fn _mm256_adds_epu16(a: u16x16, b: u16x16) -> u16x16 {
    paddusw(a, b)
}

/// Concatenate pairs of 16-byte blocks in `a` and `b` into a 32-byte temporary
/// result, shift the result right by `n` bytes, and return the low 16 bytes.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpalignr, n = 15))]
pub unsafe fn _mm256_alignr_epi8(a: i8x32, b: i8x32, n: i32) -> i8x32 {
    let n = n as u32;
    // If palignr is shifting the pair of vectors more than the size of two
    // lanes, emit zero.
    if n > 32 {
        return i8x32::splat(0);
    }
    // If palignr is shifting the pair of input vectors more than one lane,
    // but less than two lanes, convert to shifting in zeroes.
    let (a, b, n) = if n > 16 {
        (i8x32::splat(0), a, n - 16)
    } else {
        (a, b, n)
    };

    macro_rules! shuffle {
        ($shift:expr) => {
            simd_shuffle32(b, a, [
                0 + $shift, 1 + $shift,
                2 + $shift, 3 + $shift,
                4 + $shift, 5 + $shift,
                6 + $shift, 7 + $shift,
                8 + $shift, 9 + $shift,
                10 + $shift, 11 + $shift,
                12 + $shift, 13 + $shift,
                14 + $shift, 15 + $shift,
                16 + $shift, 17 + $shift,
                18 + $shift, 19 + $shift,
                20 + $shift, 21 + $shift,
                22 + $shift, 23 + $shift,
                24 + $shift, 25 + $shift,
                26 + $shift, 27 + $shift,
                28 + $shift, 29 + $shift,
                30 + $shift, 31 + $shift,
            ])
        }
    }
    match n {
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
    }
}

/// Compute the bitwise AND of 256 bits (representing integer data)
/// in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vandps))]
pub unsafe fn _mm256_and_si256(a: __m256i, b: __m256i) -> __m256i {
    a & b
}

/// Compute the bitwise NOT of 256 bits (representing integer data)
/// in `a` and then AND with `b`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vandnps))]
pub unsafe fn _mm256_andnot_si256(a: __m256i, b: __m256i) -> __m256i {
    (!a) & b
}

/// Average packed unsigned 16-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpavgw))]
pub unsafe fn _mm256_avg_epu16(a: u16x16, b: u16x16) -> u16x16 {
    pavgw(a, b)
}

/// Average packed unsigned 8-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpavgb))]
pub unsafe fn _mm256_avg_epu8(a: u8x32, b: u8x32) -> u8x32 {
    pavgb(a, b)
}

/// Blend packed 32-bit integers from `a` and `b` using control mask `imm8`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpblendd, imm8 = 9))]
pub unsafe fn _mm_blend_epi32(a: i32x4, b: i32x4, imm8: i32) -> i32x4 {
    let imm8 = (imm8 & 0xFF) as u8;
    macro_rules! blend2 {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {
            simd_shuffle4(a, b, [$a, $b, $c, $d]);
        }
    }
    macro_rules! blend1 {
        ($a:expr, $b:expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => blend2!($a, $b, 2, 3),
                0b01 => blend2!($a, $b, 6, 3),
                0b10 => blend2!($a, $b, 2, 7),
                _ => blend2!($a, $b, 6, 7),
            }
        }
    }
    match imm8 & 0b11 {
        0b00 => blend1!(0, 1),
        0b01 => blend1!(4, 1),
        0b10 => blend1!(0, 5),
        _ => blend1!(4, 5),
    }
}


/// Blend packed 32-bit integers from `a` and `b` using control mask `imm8`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpblendd, imm8 = 9))]
pub unsafe fn _mm256_blend_epi32(a: i32x8, b: i32x8, imm8: i32) -> i32x8 {
    let imm8 = (imm8 & 0xFF) as u8;
    macro_rules! blend4 {
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr) => {
            simd_shuffle8(a, b, [$a, $b, $c, $d, $e, $f, $g, $h]);
        }
    }
    macro_rules! blend3 {
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr) => {
            match (imm8 >> 6) & 0b11 {
                0b00 => blend4!($a, $b, $c, $d, $e, $f, 6, 7),
                0b01 => blend4!($a, $b, $c, $d, $e, $f, 14, 7),
                0b10 => blend4!($a, $b, $c, $d, $e, $f, 6, 15),
                _ => blend4!($a, $b, $c, $d, $e, $f, 14, 15),
            }
        }
    }
    macro_rules! blend2 {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {
            match (imm8 >> 4) & 0b11 {
                0b00 => blend3!($a, $b, $c, $d, 4, 5),
                0b01 => blend3!($a, $b, $c, $d, 12, 5),
                0b10 => blend3!($a, $b, $c, $d, 4, 13),
                _ => blend3!($a, $b, $c, $d, 12, 13),
            }
        }
    }
    macro_rules! blend1 {
        ($a:expr, $b:expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => blend2!($a, $b, 2, 3),
                0b01 => blend2!($a, $b, 10, 3),
                0b10 => blend2!($a, $b, 2, 11),
                _ => blend2!($a, $b, 10, 11),
            }
        }
    }
    match imm8 & 0b11 {
        0b00 => blend1!(0, 1),
        0b01 => blend1!(8, 1),
        0b10 => blend1!(0, 9),
        _ => blend1!(8, 9),
    }
}

/// Blend packed 16-bit integers from `a` and `b` using control mask `imm8`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpblendw, imm8 = 9))]
pub unsafe fn _mm256_blend_epi16(a: i16x16, b: i16x16, imm8: i32) -> i16x16 {
    let imm8 = (imm8 & 0xFF) as u8;
    macro_rules! blend4 {
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr,
            $i:expr, $j:expr, $k:expr, $l:expr, $m:expr, $n:expr, $o:expr, $p:expr) => {
            simd_shuffle16(a, b, [$a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m, $n, $o, $p])
        }
    }
    macro_rules! blend3 {
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr,
            $a2:expr, $b2:expr, $c2:expr, $d2:expr, $e2:expr, $f2:expr) => {
            match (imm8 >> 6) & 0b11 {
                0b00 => blend4!($a, $b, $c, $d, $e, $f, 6, 7, $a2, $b2, $c2, $d2, $e2, $f2, 14, 15),
                0b01 => blend4!($a, $b, $c, $d, $e, $f, 22, 7, $a2, $b2, $c2, $d2, $e2, $f2, 30, 15),
                0b10 => blend4!($a, $b, $c, $d, $e, $f, 6, 23, $a2, $b2, $c2, $d2, $e2, $f2, 14, 31),
                _ => blend4!($a, $b, $c, $d, $e, $f, 22, 23, $a2, $b2, $c2, $d2, $e2, $f2, 30, 31),
            }
        }
    }
    macro_rules! blend2 {
        ($a:expr, $b:expr, $c:expr, $d:expr, $a2:expr, $b2:expr, $c2:expr, $d2:expr) => {
            match (imm8 >> 4) & 0b11 {
                0b00 => blend3!($a, $b, $c, $d, 4, 5, $a2, $b2, $c2, $d2, 12, 13),
                0b01 => blend3!($a, $b, $c, $d, 20, 5, $a2, $b2, $c2, $d2, 28, 13),
                0b10 => blend3!($a, $b, $c, $d, 4, 21, $a2, $b2, $c2, $d2, 12, 29),
                _ => blend3!($a, $b, $c, $d, 20, 21, $a2, $b2, $c2, $d2, 28, 29),
            }
        }
    }
    macro_rules! blend1 {
        ($a1:expr, $b1:expr, $a2:expr, $b2:expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => blend2!($a1, $b1, 2, 3, $a2, $b2, 10, 11),
                0b01 => blend2!($a1, $b1, 18, 3, $a2, $b2, 26, 11),
                0b10 => blend2!($a1, $b1, 2, 19, $a2, $b2, 10, 27),
                _ => blend2!($a1, $b1, 18, 19, $a2, $b2, 26, 27),
            }
        }
    }
    match imm8 & 0b11 {
        0b00 => blend1!(0, 1, 8, 9),
        0b01 => blend1!(16, 1, 24, 9),
        0b10 => blend1!(0, 17, 8, 25),
        _ => blend1!(16, 17, 24, 25),
    }
}

/// Blend packed 8-bit integers from `a` and `b` using `mask`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpblendvb))]
pub unsafe fn _mm256_blendv_epi8(a: i8x32, b: i8x32, mask: __m256i) -> i8x32 {
    pblendvb(a, b, mask)
}

/// Broadcast the low packed 8-bit integer from `a` to all elements of
/// the 128-bit returned value.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm_broadcastb_epi8(a: i8x16) -> i8x16 {
    simd_shuffle16(a, i8x16::splat(0_i8), [0_u32; 16])
}

/// Broadcast the low packed 8-bit integer from `a` to all elements of
/// the 256-bit returned value.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpbroadcastb))]
pub unsafe fn _mm256_broadcastb_epi8(a: i8x16) -> i8x32 {
    simd_shuffle32(a, i8x16::splat(0_i8), [0_u32; 32])
}

// NB: simd_shuffle4 with integer data types for `a` and `b` is
// often compiled to vbroadcastss.
/// Broadcast the low packed 32-bit integer from `a` to all elements of
/// the 128-bit returned value.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vbroadcastss))]
pub unsafe fn _mm_broadcastd_epi32(a: i32x4) -> i32x4 {
    simd_shuffle4(a, i32x4::splat(0_i32), [0_u32; 4])
}

// NB: simd_shuffle4 with integer data types for `a` and `b` is
// often compiled to vbroadcastss.
/// Broadcast the low packed 32-bit integer from `a` to all elements of
/// the 256-bit returned value.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vbroadcastss))]
pub unsafe fn _mm256_broadcastd_epi32(a: i32x4) -> i32x8 {
    simd_shuffle8(a, i32x4::splat(0_i32), [0_u32; 8])
}

/// Broadcast the low packed 64-bit integer from `a` to all elements of
/// the 128-bit returned value.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpbroadcastq))]
pub unsafe fn _mm_broadcastq_epi64(a: i64x2) -> i64x2 {
    simd_shuffle2(a, i64x2::splat(0_i64), [0_u32; 2])
}

// NB: simd_shuffle4 with integer data types for `a` and `b` is
// often compiled to vbroadcastsd.
/// Broadcast the low packed 64-bit integer from `a` to all elements of
/// the 256-bit returned value.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vbroadcastsd))]
pub unsafe fn _mm256_broadcastq_epi64(a: i64x2) -> i64x4 {
    simd_shuffle4(a, i64x2::splat(0_i64), [0_u32; 4])
}

/// Broadcast the low double-precision (64-bit) floating-point element
/// from `a` to all elements of the 128-bit returned value.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vmovddup))]
pub unsafe fn _mm_broadcastsd_pd(a: f64x2) -> f64x2 {
    simd_shuffle2(a, f64x2::splat(0_f64), [0_u32; 2])
}

/// Broadcast the low double-precision (64-bit) floating-point element
/// from `a` to all elements of the 256-bit returned value.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vbroadcastsd))]
pub unsafe fn _mm256_broadcastsd_pd(a: f64x2) -> f64x4 {
    simd_shuffle4(a, f64x2::splat(0_f64), [0_u32; 4])
}

// NB: broadcastsi128_si256 is often compiled to vinsertf128 or
// vbroadcastf128.
/// Broadcast 128 bits of integer data from a to all 128-bit lanes in
/// the 256-bit returned value.
#[inline(always)]
#[target_feature = "+avx2"]
pub unsafe fn _mm256_broadcastsi128_si256(a: i64x2) -> i64x4 {
    simd_shuffle4(a, i64x2::splat(0_i64), [0, 1, 0, 1])
}

/// Broadcast the low single-precision (32-bit) floating-point element
/// from `a` to all elements of the 128-bit returned value.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vbroadcastss))]
pub unsafe fn _mm_broadcastss_ps(a: f32x4) -> f32x4 {
    simd_shuffle4(a, f32x4::splat(0_f32), [0_u32; 4])
}

/// Broadcast the low single-precision (32-bit) floating-point element
/// from `a` to all elements of the 256-bit returned value.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vbroadcastss))]
pub unsafe fn _mm256_broadcastss_ps(a: f32x4) -> f32x8 {
    simd_shuffle8(a, f32x4::splat(0_f32), [0_u32; 8])
}

/// Broadcast the low packed 16-bit integer from a to all elements of
/// the 128-bit returned value
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm_broadcastw_epi16(a: i16x8) -> i16x8 {
    simd_shuffle8(a, i16x8::splat(0_i16), [0_u32; 8])
}

/// Broadcast the low packed 16-bit integer from a to all elements of
/// the 256-bit returned value
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpbroadcastw))]
pub unsafe fn _mm256_broadcastw_epi16(a: i16x8) -> i16x16 {
    simd_shuffle16(a, i16x8::splat(0_i16), [0_u32; 16])
}

// TODO _mm256_bslli_epi128
// TODO _mm256_bsrli_epi128

/// Compare packed 64-bit integers in `a` and `b` for equality.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpcmpeqq))]
pub unsafe fn _mm256_cmpeq_epi64(a: i64x4, b: i64x4) -> i64x4 {
    a.eq(b)
}

/// Compare packed 32-bit integers in `a` and `b` for equality.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpcmpeqd))]
pub unsafe fn _mm256_cmpeq_epi32(a: i32x8, b: i32x8) -> i32x8 {
    a.eq(b)
}

/// Compare packed 16-bit integers in `a` and `b` for equality.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpcmpeqw))]
pub unsafe fn _mm256_cmpeq_epi16(a: i16x16, b: i16x16) -> i16x16 {
    a.eq(b)
}

/// Compare packed 8-bit integers in `a` and `b` for equality.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpcmpeqb))]
pub unsafe fn _mm256_cmpeq_epi8(a: i8x32, b: i8x32) -> i8x32 {
    a.eq(b)
}

/// Compare packed 64-bit integers in `a` and `b` for greater-than.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpcmpgtq))]
pub unsafe fn _mm256_cmpgt_epi64(a: i64x4, b: i64x4) -> i64x4 {
    a.gt(b)
}

/// Compare packed 32-bit integers in `a` and `b` for greater-than.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpcmpgtd))]
pub unsafe fn _mm256_cmpgt_epi32(a: i32x8, b: i32x8) -> i32x8 {
    a.gt(b)
}

/// Compare packed 16-bit integers in `a` and `b` for greater-than.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpcmpgtw))]
pub unsafe fn _mm256_cmpgt_epi16(a: i16x16, b: i16x16) -> i16x16 {
    a.gt(b)
}

/// Compare packed 8-bit integers in `a` and `b` for greater-than.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpcmpgtb))]
pub unsafe fn _mm256_cmpgt_epi8(a: i8x32, b: i8x32) -> i8x32 {
    a.gt(b)
}

/// Sign-extend 16-bit integers to 32-bit integers.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmovsxwd))]
pub unsafe fn _mm256_cvtepi16_epi32(a: i16x8) -> i32x8 {
    simd_cast(a)
}

/// Sign-extend 16-bit integers to 64-bit integers.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmovsxwq))]
pub unsafe fn _mm256_cvtepi16_epi64(a: i16x8) -> i64x4 {
    simd_cast::<::v64::i16x4, _>(simd_shuffle4(a, a, [0, 1, 2, 3]))
}

/// Sign-extend 32-bit integers to 64-bit integers.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmovsxdq))]
pub unsafe fn _mm256_cvtepi32_epi64(a: i32x4) -> i64x4 {
    simd_cast(a)
}

/// Sign-extend 8-bit integers to 16-bit integers.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmovsxbw))]
pub unsafe fn _mm256_cvtepi8_epi16(a: i8x16) -> i16x16 {
    simd_cast(a)
}

/// Sign-extend 8-bit integers to 32-bit integers.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmovsxbd))]
pub unsafe fn _mm256_cvtepi8_epi32(a: i8x16) -> i32x8 {
    simd_cast::<::v64::i8x8, _>(simd_shuffle8(a, a, [0, 1, 2, 3, 4, 5, 6, 7]))
}

/// Sign-extend 8-bit integers to 64-bit integers.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmovsxbq))]
pub unsafe fn _mm256_cvtepi8_epi64(a: i8x16) -> i64x4 {
    simd_cast::<::v32::i8x4, _>(simd_shuffle4(a, a, [0, 1, 2, 3]))
}

/// Zero-extend the lower four unsigned 16-bit integers in `a` to 32-bit
/// integers. The upper four elements of `a` are unused.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmovzxwd))]
pub unsafe fn _mm256_cvtepu16_epi32(a: u16x8) -> i32x8 {
    simd_cast(a)
}

/// Zero-extend the lower four unsigned 16-bit integers in `a` to 64-bit
/// integers. The upper four elements of `a` are unused.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmovzxwq))]
pub unsafe fn _mm256_cvtepu16_epi64(a: u16x8) -> i64x4 {
    simd_cast::<::v64::u16x4, _>(simd_shuffle4(a, a, [0, 1, 2, 3]))
}

/// Zero-extend unsigned 32-bit integers in `a` to 64-bit integers.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmovzxdq))]
pub unsafe fn _mm256_cvtepu32_epi64(a: u32x4) -> i64x4 {
    simd_cast(a)
}

/// Zero-extend unsigned 8-bit integers in `a` to 16-bit integers.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmovzxbw))]
pub unsafe fn _mm256_cvtepu8_epi16(a: u8x16) -> i16x16 {
    simd_cast(a)
}

/// Zero-extend the lower eight unsigned 8-bit integers in `a` to 32-bit
/// integers. The upper eight elements of `a` are unused.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmovzxbd))]
pub unsafe fn _mm256_cvtepu8_epi32(a: u8x16) -> i32x8 {
    simd_cast::<::v64::u8x8, _>(simd_shuffle8(a, a, [0, 1, 2, 3, 4, 5, 6, 7]))
}

/// Zero-extend the lower four unsigned 8-bit integers in `a` to 64-bit
/// integers. The upper twelve elements of `a` are unused.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmovzxbq))]
pub unsafe fn _mm256_cvtepu8_epi64(a: u8x16) -> i64x4 {
    simd_cast::<::v32::u8x4, _>(simd_shuffle4(a, a, [0, 1, 2, 3]))
}

/// Extract 128 bits (of integer data) from `a` selected with `imm8`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vextractf128, imm8 = 1))]
pub unsafe fn _mm256_extracti128_si256(a: __m256i, imm8: i32) -> __m128i {
    use x86::i586::avx::_mm256_undefined_si256;
    let imm8 = (imm8 & 0xFF) as u8;
    let b = i64x4::from(_mm256_undefined_si256());
    let dst: i64x2 = match imm8 & 0b01 {
        0 => simd_shuffle2(i64x4::from(a), b, [0, 1]),
        _ => simd_shuffle2(i64x4::from(a), b, [2, 3]),
    };
    __m128i::from(dst)
}

/// Horizontally add adjacent pairs of 16-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vphaddw))]
pub unsafe fn _mm256_hadd_epi16(a: i16x16, b: i16x16) -> i16x16 {
    phaddw(a, b)
}

/// Horizontally add adjacent pairs of 32-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vphaddd))]
pub unsafe fn _mm256_hadd_epi32(a: i32x8, b: i32x8) -> i32x8 {
    phaddd(a, b)
}

/// Horizontally add adjacent pairs of 16-bit integers in `a` and `b`
/// using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vphaddsw))]
pub unsafe fn _mm256_hadds_epi16(a: i16x16, b: i16x16) -> i16x16 {
    phaddsw(a, b)
}

/// Horizontally substract adjacent pairs of 16-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vphsubw))]
pub unsafe fn _mm256_hsub_epi16(a: i16x16, b: i16x16) -> i16x16 {
    phsubw(a, b)
}

/// Horizontally substract adjacent pairs of 32-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vphsubd))]
pub unsafe fn _mm256_hsub_epi32(a: i32x8, b: i32x8) -> i32x8 {
    phsubd(a, b)
}

/// Horizontally subtract adjacent pairs of 16-bit integers in `a` and `b`
/// using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vphsubsw))]
pub unsafe fn _mm256_hsubs_epi16(a: i16x16, b: i16x16) -> i16x16 {
    phsubsw(a, b)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherdd, scale = 1))]
pub unsafe fn _mm_i32gather_epi32(
    slice: *const i32, offsets: i32x4, scale: i8
) -> i32x4 {
    macro_rules! call {
        ($imm8:expr) => (pgatherdd(i32x4::splat(0), slice as *const i8, offsets, i32x4::splat(-1), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherdd, scale = 1))]
pub unsafe fn _mm_mask_i32gather_epi32(
    src: i32x4, slice: *const i32, offsets: i32x4, mask: i32x4, scale: i8
) -> i32x4 {
    macro_rules! call {
        ($imm8:expr) => (pgatherdd(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherdd, scale = 1))]
pub unsafe fn _mm256_i32gather_epi32(
    slice: *const i32, offsets: i32x8, scale: i8
) -> i32x8 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherdd(i32x8::splat(0), slice as *const i8, offsets, i32x8::splat(-1), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherdd, scale = 1))]
pub unsafe fn _mm256_mask_i32gather_epi32(
    src: i32x8, slice: *const i32, offsets: i32x8, mask: i32x8, scale: i8
) -> i32x8 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherdd(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherdps, scale = 1))]
pub unsafe fn _mm_i32gather_ps(
    slice: *const f32, offsets: i32x4, scale: i8
) -> f32x4 {
    macro_rules! call {
        ($imm8:expr) => (pgatherdps(f32x4::splat(0.0), slice as *const i8, offsets, f32x4::splat(-1.0), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherdps, scale = 1))]
pub unsafe fn _mm_mask_i32gather_ps(
    src: f32x4, slice: *const f32, offsets: i32x4, mask: f32x4, scale: i8
) -> f32x4 {
    macro_rules! call {
        ($imm8:expr) => (pgatherdps(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherdps, scale = 1))]
pub unsafe fn _mm256_i32gather_ps(
    slice: *const f32, offsets: i32x8, scale: i8
) -> f32x8 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherdps(f32x8::splat(0.0), slice as *const i8, offsets, f32x8::splat(-1.0), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherdps, scale = 1))]
pub unsafe fn _mm256_mask_i32gather_ps(
    src: f32x8, slice: *const f32, offsets: i32x8, mask: f32x8, scale: i8
) -> f32x8 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherdps(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherdq, scale = 1))]
pub unsafe fn _mm_i32gather_epi64(
    slice: *const i64, offsets: i32x4, scale: i8
) -> i64x2 {
    macro_rules! call {
        ($imm8:expr) => (pgatherdq(i64x2::splat(0), slice as *const i8, offsets, i64x2::splat(-1), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherdq, scale = 1))]
pub unsafe fn _mm_mask_i32gather_epi64(
    src: i64x2, slice: *const i64, offsets: i32x4, mask: i64x2, scale: i8
) -> i64x2 {
    macro_rules! call {
        ($imm8:expr) => (pgatherdq(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherdq, scale = 1))]
pub unsafe fn _mm256_i32gather_epi64(
    slice: *const i64, offsets: i32x4, scale: i8
) -> i64x4 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherdq(i64x4::splat(0), slice as *const i8, offsets, i64x4::splat(-1), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherdq, scale = 1))]
pub unsafe fn _mm256_mask_i32gather_epi64(
    src: i64x4, slice: *const i64, offsets: i32x4, mask: i64x4, scale: i8
) -> i64x4 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherdq(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherdpd, scale = 1))]
pub unsafe fn _mm_i32gather_pd(
    slice: *const f64, offsets: i32x4, scale: i8
) -> f64x2 {
    macro_rules! call {
        ($imm8:expr) => (pgatherdpd(f64x2::splat(0.0), slice as *const i8, offsets, f64x2::splat(-1.0), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherdpd, scale = 1))]
pub unsafe fn _mm_mask_i32gather_pd(
    src: f64x2, slice: *const f64, offsets: i32x4, mask: f64x2, scale: i8
) -> f64x2 {
    macro_rules! call {
        ($imm8:expr) => (pgatherdpd(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherdpd, scale = 1))]
pub unsafe fn _mm256_i32gather_pd(
    slice: *const f64, offsets: i32x4, scale: i8
) -> f64x4 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherdpd(f64x4::splat(0.0), slice as *const i8, offsets, f64x4::splat(-1.0), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherdpd, scale = 1))]
pub unsafe fn _mm256_mask_i32gather_pd(
    src: f64x4, slice: *const f64, offsets: i32x4, mask: f64x4, scale: i8
) -> f64x4 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherdpd(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherqd, scale = 1))]
pub unsafe fn _mm_i64gather_epi32(
    slice: *const i32, offsets: i64x2, scale: i8
) -> i32x4 {
    macro_rules! call {
        ($imm8:expr) => (pgatherqd(i32x4::splat(0), slice as *const i8, offsets, i32x4::splat(-1), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherqd, scale = 1))]
pub unsafe fn _mm_mask_i64gather_epi32(
    src: i32x4, slice: *const i32, offsets: i64x2, mask: i32x4, scale: i8
) -> i32x4 {
    macro_rules! call {
        ($imm8:expr) => (pgatherqd(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherqd, scale = 1))]
pub unsafe fn _mm256_i64gather_epi32(
    slice: *const i32, offsets: i64x4, scale: i8
) -> i32x4 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherqd(i32x4::splat(0), slice as *const i8, offsets, i32x4::splat(-1), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherqd, scale = 1))]
pub unsafe fn _mm256_mask_i64gather_epi32(
    src: i32x4, slice: *const i32, offsets: i64x4, mask: i32x4, scale: i8
) -> i32x4 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherqd(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherqps, scale = 1))]
pub unsafe fn _mm_i64gather_ps(
    slice: *const f32, offsets: i64x2, scale: i8
) -> f32x4 {
    macro_rules! call {
        ($imm8:expr) => (pgatherqps(f32x4::splat(0.0), slice as *const i8, offsets, f32x4::splat(-1.0), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherqps, scale = 1))]
pub unsafe fn _mm_mask_i64gather_ps(
    src: f32x4, slice: *const f32, offsets: i64x2, mask: f32x4, scale: i8
) -> f32x4 {
    macro_rules! call {
        ($imm8:expr) => (pgatherqps(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherqps, scale = 1))]
pub unsafe fn _mm256_i64gather_ps(
    slice: *const f32, offsets: i64x4, scale: i8
) -> f32x4 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherqps(f32x4::splat(0.0), slice as *const i8, offsets, f32x4::splat(-1.0), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherqps, scale = 1))]
pub unsafe fn _mm256_mask_i64gather_ps(
    src: f32x4, slice: *const f32, offsets: i64x4, mask: f32x4, scale: i8
) -> f32x4 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherqps(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherqq, scale = 1))]
pub unsafe fn _mm_i64gather_epi64(
    slice: *const i64, offsets: i64x2, scale: i8
) -> i64x2 {
    macro_rules! call {
        ($imm8:expr) => (pgatherqq(i64x2::splat(0), slice as *const i8, offsets, i64x2::splat(-1), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherqq, scale = 1))]
pub unsafe fn _mm_mask_i64gather_epi64(
    src: i64x2, slice: *const i64, offsets: i64x2, mask: i64x2, scale: i8
) -> i64x2 {
    macro_rules! call {
        ($imm8:expr) => (pgatherqq(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherqq, scale = 1))]
pub unsafe fn _mm256_i64gather_epi64(
    slice: *const i64, offsets: i64x4, scale: i8
) -> i64x4 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherqq(i64x4::splat(0), slice as *const i8, offsets, i64x4::splat(-1), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpgatherqq, scale = 1))]
pub unsafe fn _mm256_mask_i64gather_epi64(
    src: i64x4, slice: *const i64, offsets: i64x4, mask: i64x4, scale: i8
) -> i64x4 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherqq(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherqpd, scale = 1))]
pub unsafe fn _mm_i64gather_pd(
    slice: *const f64, offsets: i64x2, scale: i8
) -> f64x2 {
    macro_rules! call {
        ($imm8:expr) => (pgatherqpd(f64x2::splat(0.0), slice as *const i8, offsets, f64x2::splat(-1.0), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherqpd, scale = 1))]
pub unsafe fn _mm_mask_i64gather_pd(
    src: f64x2, slice: *const f64, offsets: i64x2, mask: f64x2, scale: i8
) -> f64x2 {
    macro_rules! call {
        ($imm8:expr) => (pgatherqpd(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherqpd, scale = 1))]
pub unsafe fn _mm256_i64gather_pd(
    slice: *const f64, offsets: i64x4, scale: i8
) -> f64x4 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherqpd(f64x4::splat(0.0), slice as *const i8, offsets, f64x4::splat(-1.0), $imm8))
    }
    constify_imm8!(scale, call)
}

/// Return values from `slice` at offsets determined by `offsets * scale`,
/// where
/// `scale` is between 1 and 8. If mask is set, load the value from `src` in
/// that position instead.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vgatherqpd, scale = 1))]
pub unsafe fn _mm256_mask_i64gather_pd(
    src: f64x4, slice: *const f64, offsets: i64x4, mask: f64x4, scale: i8
) -> f64x4 {
    macro_rules! call {
        ($imm8:expr) => (vpgatherqpd(src, slice as *const i8, offsets, mask, $imm8))
    }
    constify_imm8!(scale, call)
}

/// Copy `a` to `dst`, then insert 128 bits (of integer data) from `b` at the
/// location specified by `imm8`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vinsertf128, imm8 = 1))]
pub unsafe fn _mm256_inserti128_si256(
    a: __m256i, b: __m128i, imm8: i32
) -> __m256i {
    use x86::i586::avx::_mm256_castsi128_si256;
    let imm8 = (imm8 & 0b01) as u8;
    let b = i64x4::from(_mm256_castsi128_si256(b));
    let dst: i64x4 = match imm8 & 0b01 {
        0 => simd_shuffle4(i64x4::from(a), b, [4, 5, 2, 3]),
        _ => simd_shuffle4(i64x4::from(a), b, [0, 1, 4, 5]),
    };
    __m256i::from(dst)
}

/// Multiply packed signed 16-bit integers in `a` and `b`, producing
/// intermediate signed 32-bit integers. Horizontally add adjacent pairs
/// of intermediate 32-bit integers.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaddwd))]
pub unsafe fn _mm256_madd_epi16(a: i16x16, b: i16x16) -> i32x8 {
    pmaddwd(a, b)
}

/// Vertically multiply each unsigned 8-bit integer from `a` with the
/// corresponding signed 8-bit integer from `b`, producing intermediate
/// signed 16-bit integers. Horizontally add adjacent pairs of intermediate
/// signed 16-bit integers
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaddubsw))]
pub unsafe fn _mm256_maddubs_epi16(a: u8x32, b: u8x32) -> i16x16 {
    pmaddubsw(a, b)
}

/// Load packed 32-bit integers from memory pointed by `mem_addr` using `mask`
/// (elements are zeroed out when the highest bit is not set in the
/// corresponding element).
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaskmovd))]
pub unsafe fn _mm_maskload_epi32(mem_addr: *const i32, mask: i32x4) -> i32x4 {
    maskloadd(mem_addr as *const i8, mask)
}

/// Load packed 32-bit integers from memory pointed by `mem_addr` using `mask`
/// (elements are zeroed out when the highest bit is not set in the
/// corresponding element).
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaskmovd))]
pub unsafe fn _mm256_maskload_epi32(
    mem_addr: *const i32, mask: i32x8
) -> i32x8 {
    maskloadd256(mem_addr as *const i8, mask)
}

/// Load packed 64-bit integers from memory pointed by `mem_addr` using `mask`
/// (elements are zeroed out when the highest bit is not set in the
/// corresponding element).
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaskmovq))]
pub unsafe fn _mm_maskload_epi64(mem_addr: *const i64, mask: i64x2) -> i64x2 {
    maskloadq(mem_addr as *const i8, mask)
}

/// Load packed 64-bit integers from memory pointed by `mem_addr` using `mask`
/// (elements are zeroed out when the highest bit is not set in the
/// corresponding element).
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaskmovq))]
pub unsafe fn _mm256_maskload_epi64(
    mem_addr: *const i64, mask: i64x4
) -> i64x4 {
    maskloadq256(mem_addr as *const i8, mask)
}

/// Store packed 32-bit integers from `a` into memory pointed by `mem_addr`
/// using `mask` (elements are not stored when the highest bit is not set
/// in the corresponding element).
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaskmovd))]
pub unsafe fn _mm_maskstore_epi32(mem_addr: *mut i32, mask: i32x4, a: i32x4) {
    maskstored(mem_addr as *mut i8, mask, a)
}

/// Store packed 32-bit integers from `a` into memory pointed by `mem_addr`
/// using `mask` (elements are not stored when the highest bit is not set
/// in the corresponding element).
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaskmovd))]
pub unsafe fn _mm256_maskstore_epi32(
    mem_addr: *mut i32, mask: i32x8, a: i32x8
) {
    maskstored256(mem_addr as *mut i8, mask, a)
}

/// Store packed 64-bit integers from `a` into memory pointed by `mem_addr`
/// using `mask` (elements are not stored when the highest bit is not set
/// in the corresponding element).
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaskmovq))]
pub unsafe fn _mm_maskstore_epi64(mem_addr: *mut i64, mask: i64x2, a: i64x2) {
    maskstoreq(mem_addr as *mut i8, mask, a)
}

/// Store packed 64-bit integers from `a` into memory pointed by `mem_addr`
/// using `mask` (elements are not stored when the highest bit is not set
/// in the corresponding element).
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaskmovq))]
pub unsafe fn _mm256_maskstore_epi64(
    mem_addr: *mut i64, mask: i64x4, a: i64x4
) {
    maskstoreq256(mem_addr as *mut i8, mask, a)
}

/// Compare packed 16-bit integers in `a` and `b`, and return the packed
/// maximum values.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub unsafe fn _mm256_max_epi16(a: i16x16, b: i16x16) -> i16x16 {
    pmaxsw(a, b)
}

/// Compare packed 32-bit integers in `a` and `b`, and return the packed
/// maximum values.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaxsd))]
pub unsafe fn _mm256_max_epi32(a: i32x8, b: i32x8) -> i32x8 {
    pmaxsd(a, b)
}

/// Compare packed 8-bit integers in `a` and `b`, and return the packed
/// maximum values.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub unsafe fn _mm256_max_epi8(a: i8x32, b: i8x32) -> i8x32 {
    pmaxsb(a, b)
}

/// Compare packed unsigned 16-bit integers in `a` and `b`, and return
/// the packed maximum values.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub unsafe fn _mm256_max_epu16(a: u16x16, b: u16x16) -> u16x16 {
    pmaxuw(a, b)
}

/// Compare packed unsigned 32-bit integers in `a` and `b`, and return
/// the packed maximum values.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaxud))]
pub unsafe fn _mm256_max_epu32(a: u32x8, b: u32x8) -> u32x8 {
    pmaxud(a, b)
}

/// Compare packed unsigned 8-bit integers in `a` and `b`, and return
/// the packed maximum values.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub unsafe fn _mm256_max_epu8(a: u8x32, b: u8x32) -> u8x32 {
    pmaxub(a, b)
}

/// Compare packed 16-bit integers in `a` and `b`, and return the packed
/// minimum values.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpminsw))]
pub unsafe fn _mm256_min_epi16(a: i16x16, b: i16x16) -> i16x16 {
    pminsw(a, b)
}

/// Compare packed 32-bit integers in `a` and `b`, and return the packed
/// minimum values.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpminsd))]
pub unsafe fn _mm256_min_epi32(a: i32x8, b: i32x8) -> i32x8 {
    pminsd(a, b)
}

/// Compare packed 8-bit integers in `a` and `b`, and return the packed
/// minimum values.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpminsb))]
pub unsafe fn _mm256_min_epi8(a: i8x32, b: i8x32) -> i8x32 {
    pminsb(a, b)
}

/// Compare packed unsigned 16-bit integers in `a` and `b`, and return
/// the packed minimum values.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpminuw))]
pub unsafe fn _mm256_min_epu16(a: u16x16, b: u16x16) -> u16x16 {
    pminuw(a, b)
}

/// Compare packed unsigned 32-bit integers in `a` and `b`, and return
/// the packed minimum values.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpminud))]
pub unsafe fn _mm256_min_epu32(a: u32x8, b: u32x8) -> u32x8 {
    pminud(a, b)
}

/// Compare packed unsigned 8-bit integers in `a` and `b`, and return
/// the packed minimum values.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpminub))]
pub unsafe fn _mm256_min_epu8(a: u8x32, b: u8x32) -> u8x32 {
    pminub(a, b)
}


/// Create mask from the most significant bit of each 8-bit element in `a`,
/// return the result.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmovmskb))]
pub unsafe fn _mm256_movemask_epi8(a: i8x32) -> i32 {
    pmovmskb(a)
}

/// Compute the sum of absolute differences (SADs) of quadruplets of unsigned
/// 8-bit integers in `a` compared to those in `b`, and store the 16-bit
/// results in dst. Eight SADs are performed for each 128-bit lane using one
/// quadruplet from `b` and eight quadruplets from `a`. One quadruplet is
/// selected from `b` starting at on the offset specified in `imm8`. Eight
/// quadruplets are formed from sequential 8-bit integers selected from `a`
/// starting at the offset specified in `imm8`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vmpsadbw, imm8 = 0))]
pub unsafe fn _mm256_mpsadbw_epu8(a: u8x32, b: u8x32, imm8: i32) -> u16x16 {
    macro_rules! call {
        ($imm8:expr) => (mpsadbw(a, b, $imm8))
    }
    constify_imm8!(imm8, call)
}

/// Multiply the low 32-bit integers from each packed 64-bit element in
/// `a` and `b`
///
/// Return the 64-bit results.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmuldq))]
pub unsafe fn _mm256_mul_epi32(a: i32x8, b: i32x8) -> i64x4 {
    pmuldq(a, b)
}

/// Multiply the low unsigned 32-bit integers from each packed 64-bit
/// element in `a` and `b`
///
/// Return the unsigned 64-bit results.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmuludq))]
pub unsafe fn _mm256_mul_epu32(a: u32x8, b: u32x8) -> u64x4 {
    pmuludq(a, b)
}

/// Multiply the packed 16-bit integers in `a` and `b`, producing
/// intermediate 32-bit integers and returning the high 16 bits of the
/// intermediate integers.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub unsafe fn _mm256_mulhi_epi16(a: i16x16, b: i16x16) -> i16x16 {
    pmulhw(a, b)
}

/// Multiply the packed unsigned 16-bit integers in `a` and `b`, producing
/// intermediate 32-bit integers and returning the high 16 bits of the
/// intermediate integers.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub unsafe fn _mm256_mulhi_epu16(a: u16x16, b: u16x16) -> u16x16 {
    pmulhuw(a, b)
}

/// Multiply the packed 16-bit integers in `a` and `b`, producing
/// intermediate 32-bit integers, and return the low 16 bits of the
/// intermediate integers
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmullw))]
pub unsafe fn _mm256_mullo_epi16(a: i16x16, b: i16x16) -> i16x16 {
    a * b
}


/// Multiply the packed 32-bit integers in `a` and `b`, producing
/// intermediate 64-bit integers, and return the low 16 bits of the
/// intermediate integers
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmulld))]
pub unsafe fn _mm256_mullo_epi32(a: i32x8, b: i32x8) -> i32x8 {
    a * b
}

/// Multiply packed 16-bit integers in `a` and `b`, producing
/// intermediate signed 32-bit integers. Truncate each intermediate
/// integer to the 18 most significant bits, round by adding 1, and
/// return bits [16:1]
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub unsafe fn _mm256_mulhrs_epi16(a: i16x16, b: i16x16) -> i16x16 {
    pmulhrsw(a, b)
}

/// Compute the bitwise OR of 256 bits (representing integer data) in `a`
/// and `b`
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vorps))]
pub unsafe fn _mm256_or_si256(a: __m256i, b: __m256i) -> __m256i {
    a | b
}

/// Convert packed 16-bit integers from `a` and `b` to packed 8-bit integers
/// using signed saturation
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpacksswb))]
pub unsafe fn _mm256_packs_epi16(a: i16x16, b: i16x16) -> i8x32 {
    packsswb(a, b)
}

/// Convert packed 32-bit integers from `a` and `b` to packed 16-bit integers
/// using signed saturation
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpackssdw))]
pub unsafe fn _mm256_packs_epi32(a: i32x8, b: i32x8) -> i16x16 {
    packssdw(a, b)
}

/// Convert packed 16-bit integers from `a` and `b` to packed 8-bit integers
/// using unsigned saturation
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpackuswb))]
pub unsafe fn _mm256_packus_epi16(a: i16x16, b: i16x16) -> u8x32 {
    packuswb(a, b)
}

/// Convert packed 32-bit integers from `a` and `b` to packed 16-bit integers
/// using unsigned saturation
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpackusdw))]
pub unsafe fn _mm256_packus_epi32(a: i32x8, b: i32x8) -> u16x16 {
    packusdw(a, b)
}

/// Permutes packed 32-bit integers from `a` according to the content of `b`.
///
/// The last 3 bits of each integer of `b` are used as addresses into the 8
/// integers of `a`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpermd))]
pub unsafe fn _mm256_permutevar8x32_epi32(a: u32x8, b: u32x8) -> u32x8 {
    permd(a, b)
}

/// Permutes 64-bit integers from `a` using control mask `imm8`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpermq, imm8 = 9))]
pub unsafe fn _mm256_permute4x64_epi64(a: i64x4, imm8: i32) -> i64x4 {
    let imm8 = (imm8 & 0xFF) as u8;
    macro_rules! permute4 {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {
            simd_shuffle4(a, i64x4::splat(0), [$a, $b, $c, $d]);
        }
    }
    macro_rules! permute3 {
        ($a:expr, $b:expr, $c:expr) => {
            match (imm8 >> 6) & 0b11 {
                0b00 => permute4!($a, $b, $c, 0),
                0b01 => permute4!($a, $b, $c, 1),
                0b10 => permute4!($a, $b, $c, 2),
                _ => permute4!($a, $b, $c, 3),
            }
        }
    }
    macro_rules! permute2 {
        ($a:expr, $b:expr) => {
            match (imm8 >> 4) & 0b11 {
                0b00 => permute3!($a, $b, 0),
                0b01 => permute3!($a, $b, 1),
                0b10 => permute3!($a, $b, 2),
                _ => permute3!($a, $b, 3),
            }
        }
    }
    macro_rules! permute1 {
        ($a:expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => permute2!($a, 0),
                0b01 => permute2!($a, 1),
                0b10 => permute2!($a, 2),
                _ => permute2!($a, 3),
            }
        }
    }
    match imm8 & 0b11 {
        0b00 => permute1!(0),
        0b01 => permute1!(1),
        0b10 => permute1!(2),
        _ => permute1!(3),
    }
}

/// Shuffle 128-bits of integer data selected by `imm8` from `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vperm2f128, imm8 = 9))]
pub unsafe fn _mm256_permute2x128_si256(
    a: __m256i, b: __m256i, imm8: i32
) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            __m256i::from(vperm2i128(i64x4::from(a), i64x4::from(b), $imm8))
        }
    }
    constify_imm8!(imm8, call)
}

/// Shuffle 64-bit floating-point elements in `a` across lanes using the
/// control in `imm8`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpermpd, imm8 = 1))]
pub unsafe fn _mm256_permute4x64_pd(a: f64x4, imm8: i32) -> f64x4 {
    use x86::i586::avx::_mm256_undefined_pd;
    let imm8 = (imm8 & 0xFF) as u8;
    macro_rules! shuffle_done {
        ($x01:expr, $x23:expr, $x45:expr, $x67:expr) => {
            simd_shuffle4(a, _mm256_undefined_pd(), [$x01, $x23, $x45, $x67])
        }
    }
    macro_rules! shuffle_x67 {
        ($x01:expr, $x23:expr, $x45:expr) => {
            match (imm8 >> 6) & 0b11 {
                0b00 => shuffle_done!($x01, $x23, $x45, 0),
                0b01 => shuffle_done!($x01, $x23, $x45, 1),
                0b10 => shuffle_done!($x01, $x23, $x45, 2),
                _ => shuffle_done!($x01, $x23, $x45, 3),
            }
        }
    }
    macro_rules! shuffle_x45 {
        ($x01:expr, $x23:expr) => {
            match (imm8 >> 4) & 0b11 {
                0b00 => shuffle_x67!($x01, $x23, 0),
                0b01 => shuffle_x67!($x01, $x23, 1),
                0b10 => shuffle_x67!($x01, $x23, 2),
                _ => shuffle_x67!($x01, $x23, 3),
            }
        }
    }
    macro_rules! shuffle_x23 {
        ($x01:expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => shuffle_x45!($x01, 0),
                0b01 => shuffle_x45!($x01, 1),
                0b10 => shuffle_x45!($x01, 2),
                _ => shuffle_x45!($x01, 3),
            }
        }
    }
    match imm8 & 0b11 {
        0b00 => shuffle_x23!(0),
        0b01 => shuffle_x23!(1),
        0b10 => shuffle_x23!(2),
        _ => shuffle_x23!(3),
    }
}

/// Shuffle eight 32-bit foating-point elements in `a` across lanes using
/// the corresponding 32-bit integer index in `idx`.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpermps))]
pub unsafe fn _mm256_permutevar8x32_ps(a: f32x8, idx: i32x8) -> f32x8 {
    permps(a, idx)
}

/// Compute the absolute differences of packed unsigned 8-bit integers in `a`
/// and `b`, then horizontally sum each consecutive 8 differences to
/// produce four unsigned 16-bit integers, and pack these unsigned 16-bit
/// integers in the low 16 bits of the 64-bit return value
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsadbw))]
pub unsafe fn _mm256_sad_epu8(a: u8x32, b: u8x32) -> u64x4 {
    psadbw(a, b)
}

/// Shuffle bytes from `a` according to the content of `b`.
///
/// The last 4 bits of each byte of `b` are used as addresses into the 32 bytes
/// of `a`.
///
/// In addition, if the highest significant bit of a byte of `b` is set, the
/// respective destination byte is set to 0.
///
/// The low and high halves of the vectors are shuffled separately.
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
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpshufb))]
pub unsafe fn _mm256_shuffle_epi8(a: u8x32, b: u8x32) -> u8x32 {
    pshufb(a, b)
}

/// Shuffle 32-bit integers in 128-bit lanes of `a` using the control in
/// `imm8`.
///
/// ```rust
/// # #![feature(cfg_target_feature)]
/// # #![feature(target_feature)]
/// #
/// # #[macro_use] extern crate coresimd as stdsimd;
/// #
/// # fn main() {
/// #     if cfg_feature_enabled!("avx2") {
/// #         #[target_feature = "+avx2"]
/// #         fn worker() {
/// use stdsimd::simd::i32x8;
/// use stdsimd::vendor::_mm256_shuffle_epi32;
///
/// let a = i32x8::new(0, 1, 2, 3, 4, 5, 6, 7);
///
/// let shuffle1 = 0b00_11_10_01;
/// let shuffle2 = 0b01_00_10_11;
///
/// let c1: i32x8; let c2: i32x8;
/// unsafe {
///     c1 = _mm256_shuffle_epi32(a, shuffle1);
///     c2 = _mm256_shuffle_epi32(a, shuffle2);
/// }
///
/// let expected1 = i32x8::new(1, 2, 3, 0, 5, 6, 7, 4);
/// let expected2 = i32x8::new(3, 2, 0, 1, 7, 6, 4, 5);
///
/// assert_eq!(c1, expected1);
/// assert_eq!(c2, expected2);
/// #         }
/// #         worker();
/// #     }
/// # }
/// ```
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpshufd, imm8 = 9))]
pub unsafe fn _mm256_shuffle_epi32(a: i32x8, imm8: i32) -> i32x8 {
    // simd_shuffleX requires that its selector parameter be made up of
    // constant values, but we can't enforce that here. In spirit, we need
    // to write a `match` on all possible values of a byte, and for each value,
    // hard-code the correct `simd_shuffleX` call using only constants. We
    // then hope for LLVM to do the rest.
    //
    // Of course, that's... awful. So we try to use macros to do it for us.
    let imm8 = (imm8 & 0xFF) as u8;

    macro_rules! shuffle_done {
        ($x01:expr, $x23:expr, $x45:expr, $x67:expr) => {
            simd_shuffle8(a, a, [$x01, $x23, $x45, $x67, 4+$x01, 4+$x23, 4+$x45, 4+$x67])
        }
    }
    macro_rules! shuffle_x67 {
        ($x01:expr, $x23:expr, $x45:expr) => {
            match (imm8 >> 6) & 0b11 {
                0b00 => shuffle_done!($x01, $x23, $x45, 0),
                0b01 => shuffle_done!($x01, $x23, $x45, 1),
                0b10 => shuffle_done!($x01, $x23, $x45, 2),
                _ => shuffle_done!($x01, $x23, $x45, 3),
            }
        }
    }
    macro_rules! shuffle_x45 {
        ($x01:expr, $x23:expr) => {
            match (imm8 >> 4) & 0b11 {
                0b00 => shuffle_x67!($x01, $x23, 0),
                0b01 => shuffle_x67!($x01, $x23, 1),
                0b10 => shuffle_x67!($x01, $x23, 2),
                _ => shuffle_x67!($x01, $x23, 3),
            }
        }
    }
    macro_rules! shuffle_x23 {
        ($x01:expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => shuffle_x45!($x01, 0),
                0b01 => shuffle_x45!($x01, 1),
                0b10 => shuffle_x45!($x01, 2),
                _ => shuffle_x45!($x01, 3),
            }
        }
    }
    match imm8 & 0b11 {
        0b00 => shuffle_x23!(0),
        0b01 => shuffle_x23!(1),
        0b10 => shuffle_x23!(2),
        _ => shuffle_x23!(3),
    }
}

/// Shuffle 16-bit integers in the high 64 bits of 128-bit lanes of `a` using
/// the control in `imm8`. The low 64 bits of 128-bit lanes of `a` are copied
/// to the output.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpshufhw, imm8 = 9))]
pub unsafe fn _mm256_shufflehi_epi16(a: i16x16, imm8: i32) -> i16x16 {
    let imm8 = (imm8 & 0xFF) as u8;
    macro_rules! shuffle_done {
        ($x01:expr, $x23:expr, $x45:expr, $x67:expr) => {
            #[cfg_attr(rustfmt, rustfmt_skip)]
            simd_shuffle16(a, a, [
                0, 1, 2, 3, 4+$x01, 4+$x23, 4+$x45, 4+$x67,
                8, 9, 10, 11, 12+$x01, 12+$x23, 12+$x45, 12+$x67
            ]);
        }
    }
    macro_rules! shuffle_x67 {
        ($x01:expr, $x23:expr, $x45:expr) => {
            match (imm8 >> 6) & 0b11 {
                0b00 => shuffle_done!($x01, $x23, $x45, 0),
                0b01 => shuffle_done!($x01, $x23, $x45, 1),
                0b10 => shuffle_done!($x01, $x23, $x45, 2),
                _ => shuffle_done!($x01, $x23, $x45, 3),
            }
        }
    }
    macro_rules! shuffle_x45 {
        ($x01:expr, $x23:expr) => {
            match (imm8 >> 4) & 0b11 {
                0b00 => shuffle_x67!($x01, $x23, 0),
                0b01 => shuffle_x67!($x01, $x23, 1),
                0b10 => shuffle_x67!($x01, $x23, 2),
                _ => shuffle_x67!($x01, $x23, 3),
            }
        }
    }
    macro_rules! shuffle_x23 {
        ($x01:expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => shuffle_x45!($x01, 0),
                0b01 => shuffle_x45!($x01, 1),
                0b10 => shuffle_x45!($x01, 2),
                _ => shuffle_x45!($x01, 3),
            }
        }
    }
    match imm8 & 0b11 {
        0b00 => shuffle_x23!(0),
        0b01 => shuffle_x23!(1),
        0b10 => shuffle_x23!(2),
        _ => shuffle_x23!(3),
    }
}

/// Shuffle 16-bit integers in the low 64 bits of 128-bit lanes of `a` using
/// the control in `imm8`. The high 64 bits of 128-bit lanes of `a` are copied
/// to the output.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpshuflw, imm8 = 9))]
pub unsafe fn _mm256_shufflelo_epi16(a: i16x16, imm8: i32) -> i16x16 {
    let imm8 = (imm8 & 0xFF) as u8;
    macro_rules! shuffle_done {
        ($x01:expr, $x23:expr, $x45:expr, $x67:expr) => {
            #[cfg_attr(rustfmt, rustfmt_skip)]
            simd_shuffle16(a, a, [
                0+$x01, 0+$x23, 0+$x45, 0+$x67, 4, 5, 6, 7,
                8+$x01, 8+$x23, 8+$x45, 8+$x67, 12, 13, 14, 15,
            ]);
        }
    }
    macro_rules! shuffle_x67 {
        ($x01:expr, $x23:expr, $x45:expr) => {
            match (imm8 >> 6) & 0b11 {
                0b00 => shuffle_done!($x01, $x23, $x45, 0),
                0b01 => shuffle_done!($x01, $x23, $x45, 1),
                0b10 => shuffle_done!($x01, $x23, $x45, 2),
                _ => shuffle_done!($x01, $x23, $x45, 3),
            }
        }
    }
    macro_rules! shuffle_x45 {
        ($x01:expr, $x23:expr) => {
            match (imm8 >> 4) & 0b11 {
                0b00 => shuffle_x67!($x01, $x23, 0),
                0b01 => shuffle_x67!($x01, $x23, 1),
                0b10 => shuffle_x67!($x01, $x23, 2),
                _ => shuffle_x67!($x01, $x23, 3),
            }
        }
    }
    macro_rules! shuffle_x23 {
        ($x01:expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => shuffle_x45!($x01, 0),
                0b01 => shuffle_x45!($x01, 1),
                0b10 => shuffle_x45!($x01, 2),
                _ => shuffle_x45!($x01, 3),
            }
        }
    }
    match imm8 & 0b11 {
        0b00 => shuffle_x23!(0),
        0b01 => shuffle_x23!(1),
        0b10 => shuffle_x23!(2),
        _ => shuffle_x23!(3),
    }
}

/// Negate packed 16-bit integers in `a` when the corresponding signed
/// 16-bit integer in `b` is negative, and return the results.
/// Results are zeroed out when the corresponding element in `b` is zero.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsignw))]
pub unsafe fn _mm256_sign_epi16(a: i16x16, b: i16x16) -> i16x16 {
    psignw(a, b)
}

/// Negate packed 32-bit integers in `a` when the corresponding signed
/// 32-bit integer in `b` is negative, and return the results.
/// Results are zeroed out when the corresponding element in `b` is zero.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsignd))]
pub unsafe fn _mm256_sign_epi32(a: i32x8, b: i32x8) -> i32x8 {
    psignd(a, b)
}

/// Negate packed 8-bit integers in `a` when the corresponding signed
/// 8-bit integer in `b` is negative, and return the results.
/// Results are zeroed out when the corresponding element in `b` is zero.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsignb))]
pub unsafe fn _mm256_sign_epi8(a: i8x32, b: i8x32) -> i8x32 {
    psignb(a, b)
}

/// Shift packed 16-bit integers in `a` left by `count` while
/// shifting in zeros, and return the result
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsllw))]
pub unsafe fn _mm256_sll_epi16(a: i16x16, count: i16x8) -> i16x16 {
    psllw(a, count)
}

/// Shift packed 32-bit integers in `a` left by `count` while
/// shifting in zeros, and return the result
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpslld))]
pub unsafe fn _mm256_sll_epi32(a: i32x8, count: i32x4) -> i32x8 {
    pslld(a, count)
}

/// Shift packed 64-bit integers in `a` left by `count` while
/// shifting in zeros, and return the result
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsllq))]
pub unsafe fn _mm256_sll_epi64(a: i64x4, count: i64x2) -> i64x4 {
    psllq(a, count)
}

/// Shift packed 16-bit integers in `a` left by `imm8` while
/// shifting in zeros, return the results;
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsllw))]
pub unsafe fn _mm256_slli_epi16(a: i16x16, imm8: i32) -> i16x16 {
    pslliw(a, imm8)
}

/// Shift packed 32-bit integers in `a` left by `imm8` while
/// shifting in zeros, return the results;
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpslld))]
pub unsafe fn _mm256_slli_epi32(a: i32x8, imm8: i32) -> i32x8 {
    psllid(a, imm8)
}

/// Shift packed 64-bit integers in `a` left by `imm8` while
/// shifting in zeros, return the results;
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsllq))]
pub unsafe fn _mm256_slli_epi64(a: i64x4, imm8: i32) -> i64x4 {
    pslliq(a, imm8)
}

// TODO _mm256_slli_si256 (__m256i a, const int imm8)

/// Shift packed 32-bit integers in `a` left by the amount
/// specified by the corresponding element in `count` while
/// shifting in zeros, and return the result.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsllvd))]
pub unsafe fn _mm_sllv_epi32(a: i32x4, count: i32x4) -> i32x4 {
    psllvd(a, count)
}

/// Shift packed 32-bit integers in `a` left by the amount
/// specified by the corresponding element in `count` while
/// shifting in zeros, and return the result.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsllvd))]
pub unsafe fn _mm256_sllv_epi32(a: i32x8, count: i32x8) -> i32x8 {
    psllvd256(a, count)
}

/// Shift packed 64-bit integers in `a` left by the amount
/// specified by the corresponding element in `count` while
/// shifting in zeros, and return the result.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsllvq))]
pub unsafe fn _mm_sllv_epi64(a: i64x2, count: i64x2) -> i64x2 {
    psllvq(a, count)
}

/// Shift packed 64-bit integers in `a` left by the amount
/// specified by the corresponding element in `count` while
/// shifting in zeros, and return the result.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsllvq))]
pub unsafe fn _mm256_sllv_epi64(a: i64x4, count: i64x4) -> i64x4 {
    psllvq256(a, count)
}

/// Shift packed 16-bit integers in `a` right by `count` while
/// shifting in sign bits.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsraw))]
pub unsafe fn _mm256_sra_epi16(a: i16x16, count: i16x8) -> i16x16 {
    psraw(a, count)
}

/// Shift packed 32-bit integers in `a` right by `count` while
/// shifting in sign bits.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsrad))]
pub unsafe fn _mm256_sra_epi32(a: i32x8, count: i32x4) -> i32x8 {
    psrad(a, count)
}

/// Shift packed 16-bit integers in `a` right by `imm8` while
/// shifting in sign bits.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsraw))]
pub unsafe fn _mm256_srai_epi16(a: i16x16, imm8: i32) -> i16x16 {
    psraiw(a, imm8)
}

/// Shift packed 32-bit integers in `a` right by `imm8` while
/// shifting in sign bits.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsrad))]
pub unsafe fn _mm256_srai_epi32(a: i32x8, imm8: i32) -> i32x8 {
    psraid(a, imm8)
}

/// Shift packed 32-bit integers in `a` right by the amount specified by the
/// corresponding element in `count` while shifting in sign bits.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsravd))]
pub unsafe fn _mm_srav_epi32(a: i32x4, count: i32x4) -> i32x4 {
    psravd(a, count)
}

/// Shift packed 32-bit integers in `a` right by the amount specified by the
/// corresponding element in `count` while shifting in sign bits.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsravd))]
pub unsafe fn _mm256_srav_epi32(a: i32x8, count: i32x8) -> i32x8 {
    psravd256(a, count)
}


/// Shift packed 16-bit integers in `a` right by `count` while shifting in
/// zeros.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub unsafe fn _mm256_srl_epi16(a: i16x16, count: i16x8) -> i16x16 {
    psrlw(a, count)
}

/// Shift packed 32-bit integers in `a` right by `count` while shifting in
/// zeros.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsrld))]
pub unsafe fn _mm256_srl_epi32(a: i32x8, count: i32x4) -> i32x8 {
    psrld(a, count)
}

/// Shift packed 64-bit integers in `a` right by `count` while shifting in
/// zeros.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsrlq))]
pub unsafe fn _mm256_srl_epi64(a: i64x4, count: i64x2) -> i64x4 {
    psrlq(a, count)
}

/// Shift packed 16-bit integers in `a` right by `imm8` while shifting in
/// zeros
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsrlw))]
pub unsafe fn _mm256_srli_epi16(a: i16x16, imm8: i32) -> i16x16 {
    psrliw(a, imm8)
}

/// Shift packed 32-bit integers in `a` right by `imm8` while shifting in
/// zeros
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsrld))]
pub unsafe fn _mm256_srli_epi32(a: i32x8, imm8: i32) -> i32x8 {
    psrlid(a, imm8)
}

/// Shift packed 64-bit integers in `a` right by `imm8` while shifting in
/// zeros
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsrlq))]
pub unsafe fn _mm256_srli_epi64(a: i64x4, imm8: i32) -> i64x4 {
    psrliq(a, imm8)
}

/// Shift packed 32-bit integers in `a` right by the amount specified by
/// the corresponding element in `count` while shifting in zeros,
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsrlvd))]
pub unsafe fn _mm_srlv_epi32(a: i32x4, count: i32x4) -> i32x4 {
    psrlvd(a, count)
}

/// Shift packed 32-bit integers in `a` right by the amount specified by
/// the corresponding element in `count` while shifting in zeros,
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsrlvd))]
pub unsafe fn _mm256_srlv_epi32(a: i32x8, count: i32x8) -> i32x8 {
    psrlvd256(a, count)
}

/// Shift packed 64-bit integers in `a` right by the amount specified by
/// the corresponding element in `count` while shifting in zeros,
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsrlvq))]
pub unsafe fn _mm_srlv_epi64(a: i64x2, count: i64x2) -> i64x2 {
    psrlvq(a, count)
}

/// Shift packed 64-bit integers in `a` right by the amount specified by
/// the corresponding element in `count` while shifting in zeros,
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsrlvq))]
pub unsafe fn _mm256_srlv_epi64(a: i64x4, count: i64x4) -> i64x4 {
    psrlvq256(a, count)
}

// TODO _mm256_stream_load_si256 (__m256i const* mem_addr)

/// Subtract packed 16-bit integers in `b` from packed 16-bit integers in `a`
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsubw))]
pub unsafe fn _mm256_sub_epi16(a: i16x16, b: i16x16) -> i16x16 {
    a - b
}

/// Subtract packed 32-bit integers in `b` from packed 16-bit integers in `a`
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsubd))]
pub unsafe fn _mm256_sub_epi32(a: i32x8, b: i32x8) -> i32x8 {
    a - b
}

/// Subtract packed 64-bit integers in `b` from packed 16-bit integers in `a`
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsubq))]
pub unsafe fn _mm256_sub_epi64(a: i64x4, b: i64x4) -> i64x4 {
    a - b
}

/// Subtract packed 8-bit integers in `b` from packed 16-bit integers in `a`
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsubb))]
pub unsafe fn _mm256_sub_epi8(a: i8x32, b: i8x32) -> i8x32 {
    a - b
}

/// Subtract packed 16-bit integers in `b` from packed 16-bit integers in
/// `a` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub unsafe fn _mm256_subs_epi16(a: i16x16, b: i16x16) -> i16x16 {
    psubsw(a, b)
}

/// Subtract packed 8-bit integers in `b` from packed 8-bit integers in
/// `a` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub unsafe fn _mm256_subs_epi8(a: i8x32, b: i8x32) -> i8x32 {
    psubsb(a, b)
}

/// Subtract packed unsigned 16-bit integers in `b` from packed 16-bit
/// integers in `a` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub unsafe fn _mm256_subs_epu16(a: u16x16, b: u16x16) -> u16x16 {
    psubusw(a, b)
}

/// Subtract packed unsigned 8-bit integers in `b` from packed 8-bit
/// integers in `a` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub unsafe fn _mm256_subs_epu8(a: u8x32, b: u8x32) -> u8x32 {
    psubusb(a, b)
}

/// Unpack and interleave 8-bit integers from the high half of each
/// 128-bit lane in `a` and `b`.
///
/// ```rust
/// # #![feature(cfg_target_feature)]
/// # #![feature(target_feature)]
/// #
/// # #[macro_use] extern crate coresimd as stdsimd;
/// #
/// # fn main() {
/// #     if cfg_feature_enabled!("avx2") {
/// #         #[target_feature = "+avx2"]
/// #         fn worker() {
/// use stdsimd::simd::i8x32;
/// use stdsimd::vendor::_mm256_unpackhi_epi8;
///
/// let a = i8x32::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
/// 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
/// let b = i8x32::new(0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,
/// -16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-31);
///
/// let c: i8x32;
/// unsafe {
///     c = _mm256_unpackhi_epi8(a, b);
/// }
///
/// let expected = i8x32::new(8,-8, 9,-9, 10,-10, 11,-11, 12,-12, 13,-13,
/// 14,-14, 15,-15, 24,-24, 25,-25, 26,-26, 27,-27, 28,-28, 29,-29, 30,-30,
/// 31,-31);
/// assert_eq!(c, expected);
///
/// #         }
/// #         worker();
/// #     }
/// # }
/// ```
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpunpckhbw))]
pub unsafe fn _mm256_unpackhi_epi8(a: i8x32, b: i8x32) -> i8x32 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    simd_shuffle32(a, b, [
            8, 40, 9, 41, 10, 42, 11, 43,
            12, 44, 13, 45, 14, 46, 15, 47,
            24, 56, 25, 57, 26, 58, 27, 59,
            28, 60, 29, 61, 30, 62, 31, 63,
    ])
}

/// Unpack and interleave 8-bit integers from the low half of each
/// 128-bit lane of `a` and `b`.
///
/// ```rust
/// # #![feature(cfg_target_feature)]
/// # #![feature(target_feature)]
/// #
/// # #[macro_use] extern crate coresimd as stdsimd;
/// #
/// # fn main() {
/// #     if cfg_feature_enabled!("avx2") {
/// #         #[target_feature = "+avx2"]
/// #         fn worker() {
/// use stdsimd::simd::i8x32;
/// use stdsimd::vendor::_mm256_unpacklo_epi8;
///
/// let a = i8x32::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
/// 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
/// let b = i8x32::new(0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,
/// -16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-31);
///
/// let c: i8x32;
/// unsafe {
///     c = _mm256_unpacklo_epi8(a, b);
/// }
///
/// let expected = i8x32::new(0, 0, 1,-1, 2,-2, 3,-3, 4,-4, 5,-5, 6,-6, 7,-7,
/// 16,-16, 17,-17, 18,-18, 19,-19, 20,-20, 21,-21, 22,-22, 23,-23);
/// assert_eq!(c, expected);
///
/// #         }
/// #         worker();
/// #     }
/// # }
/// ```
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpunpcklbw))]
pub unsafe fn _mm256_unpacklo_epi8(a: i8x32, b: i8x32) -> i8x32 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    simd_shuffle32(a, b, [
        0, 32, 1, 33, 2, 34, 3, 35,
        4, 36, 5, 37, 6, 38, 7, 39,
        16, 48, 17, 49, 18, 50, 19, 51,
        20, 52, 21, 53, 22, 54, 23, 55,
    ])
}

/// Unpack and interleave 16-bit integers from the high half of each
/// 128-bit lane of `a` and `b`.
///
/// ```rust
/// # #![feature(cfg_target_feature)]
/// # #![feature(target_feature)]
/// #
/// # #[macro_use] extern crate coresimd as stdsimd;
/// #
/// # fn main() {
/// #     if cfg_feature_enabled!("avx2") {
/// #         #[target_feature = "+avx2"]
/// #         fn worker() {
/// use stdsimd::simd::i16x16;
/// use stdsimd::vendor::_mm256_unpackhi_epi16;
///
/// let a = i16x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
/// let b = i16x16::new(0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15);
///
/// let c: i16x16;
/// unsafe {
///     c = _mm256_unpackhi_epi16(a, b);
/// }
///
/// let expected = i16x16::new(4,-4, 5,-5, 6,-6, 7,-7, 12,-12, 13,-13, 14,-14,
/// 15,-15);
/// assert_eq!(c, expected);
///
/// #         }
/// #         worker();
/// #     }
/// # }
/// ```
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpunpckhwd))]
pub unsafe fn _mm256_unpackhi_epi16(a: i16x16, b: i16x16) -> i16x16 {
    simd_shuffle16(
        a,
        b,
        [4, 20, 5, 21, 6, 22, 7, 23, 12, 28, 13, 29, 14, 30, 15, 31],
    )
}

/// Unpack and interleave 16-bit integers from the low half of each
/// 128-bit lane of `a` and `b`.
///
/// ```rust
/// # #![feature(cfg_target_feature)]
/// # #![feature(target_feature)]
/// #
/// # #[macro_use] extern crate coresimd as stdsimd;
/// #
/// # fn main() {
/// #     if cfg_feature_enabled!("avx2") {
/// #         #[target_feature = "+avx2"]
/// #         fn worker() {
/// use stdsimd::simd::i16x16;
/// use stdsimd::vendor::_mm256_unpacklo_epi16;
///
/// let a = i16x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
/// let b = i16x16::new(0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15);
///
/// let c: i16x16;
/// unsafe {
///     c = _mm256_unpacklo_epi16(a, b);
/// }
///
/// let expected = i16x16::new(0, 0, 1,-1, 2,-2, 3,-3, 8,-8, 9,-9, 10,-10,
/// 11,-11);
/// assert_eq!(c, expected);
///
/// #         }
/// #         worker();
/// #     }
/// # }
/// ```
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpunpcklwd))]
pub unsafe fn _mm256_unpacklo_epi16(a: i16x16, b: i16x16) -> i16x16 {
    simd_shuffle16(
        a,
        b,
        [0, 16, 1, 17, 2, 18, 3, 19, 8, 24, 9, 25, 10, 26, 11, 27],
    )
}

/// Unpack and interleave 32-bit integers from the high half of each
/// 128-bit lane of `a` and `b`.
///
/// ```rust
/// # #![feature(cfg_target_feature)]
/// # #![feature(target_feature)]
/// #
/// # #[macro_use] extern crate coresimd as stdsimd;
/// #
/// # fn main() {
/// #     if cfg_feature_enabled!("avx2") {
/// #         #[target_feature = "+avx2"]
/// #         fn worker() {
/// use stdsimd::simd::i32x8;
/// use stdsimd::vendor::_mm256_unpackhi_epi32;
///
/// let a = i32x8::new(0, 1, 2, 3, 4, 5, 6, 7);
/// let b = i32x8::new(0,-1,-2,-3,-4,-5,-6,-7);
///
/// let c: i32x8;
/// unsafe {
///     c = _mm256_unpackhi_epi32(a, b);
/// }
///
/// let expected = i32x8::new(2,-2, 3,-3, 6,-6, 7,-7);
/// assert_eq!(c, expected);
///
/// #         }
/// #         worker();
/// #     }
/// # }
/// ```
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpunpckhdq))]
pub unsafe fn _mm256_unpackhi_epi32(a: i32x8, b: i32x8) -> i32x8 {
    simd_shuffle8(a, b, [2, 10, 3, 11, 6, 14, 7, 15])
}

/// Unpack and interleave 32-bit integers from the low half of each
/// 128-bit lane of `a` and `b`.
///
/// ```rust
/// # #![feature(cfg_target_feature)]
/// # #![feature(target_feature)]
/// #
/// # #[macro_use] extern crate coresimd as stdsimd;
/// #
/// # fn main() {
/// #     if cfg_feature_enabled!("avx2") {
/// #         #[target_feature = "+avx2"]
/// #         fn worker() {
/// use stdsimd::simd::i32x8;
/// use stdsimd::vendor::_mm256_unpacklo_epi32;
///
/// let a = i32x8::new(0, 1, 2, 3, 4, 5, 6, 7);
/// let b = i32x8::new(0,-1,-2,-3,-4,-5,-6,-7);
///
/// let c: i32x8;
/// unsafe {
///     c = _mm256_unpacklo_epi32(a, b);
/// }
///
/// let expected = i32x8::new(0, 0, 1,-1, 4,-4, 5,-5);
/// assert_eq!(c, expected);
///
/// #         }
/// #         worker();
/// #     }
/// # }
/// ```
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpunpckldq))]
pub unsafe fn _mm256_unpacklo_epi32(a: i32x8, b: i32x8) -> i32x8 {
    simd_shuffle8(a, b, [0, 8, 1, 9, 4, 12, 5, 13])
}

/// Unpack and interleave 64-bit integers from the high half of each
/// 128-bit lane of `a` and `b`.
///
/// ```rust
/// # #![feature(cfg_target_feature)]
/// # #![feature(target_feature)]
/// #
/// # #[macro_use] extern crate coresimd as stdsimd;
/// #
/// # fn main() {
/// #     if cfg_feature_enabled!("avx2") {
/// #         #[target_feature = "+avx2"]
/// #         fn worker() {
/// use stdsimd::simd::i64x4;
/// use stdsimd::vendor::_mm256_unpackhi_epi64;
///
/// let a = i64x4::new(0, 1, 2, 3);
/// let b = i64x4::new(0,-1,-2,-3);
///
/// let c: i64x4;
/// unsafe {
///     c = _mm256_unpackhi_epi64(a, b);
/// }
///
/// let expected = i64x4::new(1,-1, 3,-3);
/// assert_eq!(c, expected);
///
/// #         }
/// #         worker();
/// #     }
/// # }
/// ```
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpunpckhqdq))]
pub unsafe fn _mm256_unpackhi_epi64(a: i64x4, b: i64x4) -> i64x4 {
    simd_shuffle4(a, b, [1, 5, 3, 7])
}

/// Unpack and interleave 64-bit integers from the low half of each
/// 128-bit lane of `a` and `b`.
///
/// ```rust
/// # #![feature(cfg_target_feature)]
/// # #![feature(target_feature)]
/// #
/// # #[macro_use] extern crate coresimd as stdsimd;
/// #
/// # fn main() {
/// #     if cfg_feature_enabled!("avx2") {
/// #         #[target_feature = "+avx2"]
/// #         fn worker() {
/// use stdsimd::simd::i64x4;
/// use stdsimd::vendor::_mm256_unpacklo_epi64;
///
/// let a = i64x4::new(0, 1, 2, 3);
/// let b = i64x4::new(0,-1,-2,-3);
///
/// let c: i64x4;
/// unsafe {
///     c = _mm256_unpacklo_epi64(a, b);
/// }
///
/// let expected = i64x4::new(0, 0, 2,-2);
/// assert_eq!(c, expected);
///
/// #         }
/// #         worker();
/// #     }
/// # }
/// ```
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vpunpcklqdq))]
pub unsafe fn _mm256_unpacklo_epi64(a: i64x4, b: i64x4) -> i64x4 {
    simd_shuffle4(a, b, [0, 4, 2, 6])
}

/// Compute the bitwise XOR of 256 bits (representing integer data)
/// in `a` and `b`
#[inline(always)]
#[target_feature = "+avx2"]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm256_xor_si256(a: __m256i, b: __m256i) -> __m256i {
    a ^ b
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx2.pabs.b"]
    fn pabsb(a: i8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.pabs.w"]
    fn pabsw(a: i16x16) -> u16x16;
    #[link_name = "llvm.x86.avx2.pabs.d"]
    fn pabsd(a: i32x8) -> u32x8;
    #[link_name = "llvm.x86.avx2.padds.b"]
    fn paddsb(a: i8x32, b: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx2.padds.w"]
    fn paddsw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.paddus.b"]
    fn paddusb(a: u8x32, b: u8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.paddus.w"]
    fn paddusw(a: u16x16, b: u16x16) -> u16x16;
    #[link_name = "llvm.x86.avx2.pavg.b"]
    fn pavgb(a: u8x32, b: u8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.pavg.w"]
    fn pavgw(a: u16x16, b: u16x16) -> u16x16;
    #[link_name = "llvm.x86.avx2.pblendvb"]
    fn pblendvb(a: i8x32, b: i8x32, mask: __m256i) -> i8x32;
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
    #[link_name = "llvm.x86.avx2.psubs.b"]
    fn psubsb(a: i8x32, b: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx2.psubs.w"]
    fn psubsw(a: i16x16, b: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.psubus.b"]
    fn psubusb(a: u8x32, b: u8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.psubus.w"]
    fn psubusw(a: u16x16, b: u16x16) -> u16x16;
    #[link_name = "llvm.x86.avx2.pshuf.b"]
    fn pshufb(a: u8x32, b: u8x32) -> u8x32;
    #[link_name = "llvm.x86.avx2.permd"]
    fn permd(a: u32x8, b: u32x8) -> u32x8;
    #[link_name = "llvm.x86.avx2.permps"]
    fn permps(a: f32x8, b: i32x8) -> f32x8;
    #[link_name = "llvm.x86.avx2.vperm2i128"]
    fn vperm2i128(a: i64x4, b: i64x4, imm8: i8) -> i64x4;
    #[link_name = "llvm.x86.avx2.gather.d.d"]
    fn pgatherdd(
        src: i32x4, slice: *const i8, offsets: i32x4, mask: i32x4, scale: i8
    ) -> i32x4;
    #[link_name = "llvm.x86.avx2.gather.d.d.256"]
    fn vpgatherdd(
        src: i32x8, slice: *const i8, offsets: i32x8, mask: i32x8, scale: i8
    ) -> i32x8;
    #[link_name = "llvm.x86.avx2.gather.d.q"]
    fn pgatherdq(
        src: i64x2, slice: *const i8, offsets: i32x4, mask: i64x2, scale: i8
    ) -> i64x2;
    #[link_name = "llvm.x86.avx2.gather.d.q.256"]
    fn vpgatherdq(
        src: i64x4, slice: *const i8, offsets: i32x4, mask: i64x4, scale: i8
    ) -> i64x4;
    #[link_name = "llvm.x86.avx2.gather.q.d"]
    fn pgatherqd(
        src: i32x4, slice: *const i8, offsets: i64x2, mask: i32x4, scale: i8
    ) -> i32x4;
    #[link_name = "llvm.x86.avx2.gather.q.d.256"]
    fn vpgatherqd(
        src: i32x4, slice: *const i8, offsets: i64x4, mask: i32x4, scale: i8
    ) -> i32x4;
    #[link_name = "llvm.x86.avx2.gather.q.q"]
    fn pgatherqq(
        src: i64x2, slice: *const i8, offsets: i64x2, mask: i64x2, scale: i8
    ) -> i64x2;
    #[link_name = "llvm.x86.avx2.gather.q.q.256"]
    fn vpgatherqq(
        src: i64x4, slice: *const i8, offsets: i64x4, mask: i64x4, scale: i8
    ) -> i64x4;
    #[link_name = "llvm.x86.avx2.gather.d.pd"]
    fn pgatherdpd(
        src: f64x2, slice: *const i8, offsets: i32x4, mask: f64x2, scale: i8
    ) -> f64x2;
    #[link_name = "llvm.x86.avx2.gather.d.pd.256"]
    fn vpgatherdpd(
        src: f64x4, slice: *const i8, offsets: i32x4, mask: f64x4, scale: i8
    ) -> f64x4;
    #[link_name = "llvm.x86.avx2.gather.q.pd"]
    fn pgatherqpd(
        src: f64x2, slice: *const i8, offsets: i64x2, mask: f64x2, scale: i8
    ) -> f64x2;
    #[link_name = "llvm.x86.avx2.gather.q.pd.256"]
    fn vpgatherqpd(
        src: f64x4, slice: *const i8, offsets: i64x4, mask: f64x4, scale: i8
    ) -> f64x4;
    #[link_name = "llvm.x86.avx2.gather.d.ps"]
    fn pgatherdps(
        src: f32x4, slice: *const i8, offsets: i32x4, mask: f32x4, scale: i8
    ) -> f32x4;
    #[link_name = "llvm.x86.avx2.gather.d.ps.256"]
    fn vpgatherdps(
        src: f32x8, slice: *const i8, offsets: i32x8, mask: f32x8, scale: i8
    ) -> f32x8;
    #[link_name = "llvm.x86.avx2.gather.q.ps"]
    fn pgatherqps(
        src: f32x4, slice: *const i8, offsets: i64x2, mask: f32x4, scale: i8
    ) -> f32x4;
    #[link_name = "llvm.x86.avx2.gather.q.ps.256"]
    fn vpgatherqps(
        src: f32x4, slice: *const i8, offsets: i64x4, mask: f32x4, scale: i8
    ) -> f32x4;

}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use v256::*;
    use v128::*;
    use x86::i586::avx2;
    use x86::{__m128i, __m256i};
    use std;

    #[simd_test = "avx2"]
    unsafe fn _mm256_abs_epi32() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = i32x8::new(
            0, 1, -1, std::i32::MAX,
            std::i32::MIN, 100, -100, -32,
        );
        let r = avx2::_mm256_abs_epi32(a);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = u32x8::new(
            0, 1, 1, std::i32::MAX as u32,
            std::i32::MAX as u32 + 1, 100, 100, 32,
        );
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_abs_epi16() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = i16x16::new(
            0,  1, -1, 2, -2, 3, -3, 4,
            -4, 5, -5, std::i16::MAX, std::i16::MIN, 100, -100, -32,
        );
        let r = avx2::_mm256_abs_epi16(a);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = u16x16::new(
            0, 1, 1, 2, 2, 3, 3, 4,
            4, 5, 5, std::i16::MAX as u16, std::i16::MAX as u16 + 1, 100, 100, 32,
        );
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_abs_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = i8x32::new(
            0, 1, -1, 2, -2, 3, -3, 4,
            -4, 5, -5, std::i8::MAX, std::i8::MIN, 100, -100, -32,
            0, 1, -1, 2, -2, 3, -3, 4,
            -4, 5, -5, std::i8::MAX, std::i8::MIN, 100, -100, -32,
        );
        let r = avx2::_mm256_abs_epi8(a);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = u8x32::new(
            0, 1, 1, 2, 2, 3, 3, 4,
            4, 5, 5, std::i8::MAX as u8, std::i8::MAX as u8 + 1, 100, 100, 32,
            0, 1, 1, 2, 2, 3, 3, 4,
            4, 5, 5, std::i8::MAX as u8, std::i8::MAX as u8 + 1, 100, 100, 32,
        );
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_add_epi64() {
        let a = i64x4::new(-10, 0, 100, 1_000_000_000);
        let b = i64x4::new(-1, 0, 1, 2);
        let r = avx2::_mm256_add_epi64(a, b);
        let e = i64x4::new(-11, 0, 101, 1_000_000_002);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_add_epi32() {
        let a = i32x8::new(-1, 0, 1, 2, 3, 4, 5, 6);
        let b = i32x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = avx2::_mm256_add_epi32(a, b);
        let e = i32x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_add_epi16() {
        let a =
            i16x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b =
            i16x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = avx2::_mm256_add_epi16(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i16x16::new(
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 26, 28, 30,
        );
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_add_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = i8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = i8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        let r = avx2::_mm256_add_epi8(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i8x32::new(
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 26, 28, 30,
            32, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62,
        );
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_adds_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = i8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = i8x32::new(
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63,
        );
        let r = avx2::_mm256_adds_epi8(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i8x32::new(
            32, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62,
            64, 66, 68, 70, 72, 74, 76, 78,
            80, 82, 84, 86, 88, 90, 92, 94,
        );
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_adds_epi8_saturate_positive() {
        let a = i8x32::splat(0x7F);
        let b = i8x32::splat(1);
        let r = avx2::_mm256_adds_epi8(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_adds_epi8_saturate_negative() {
        let a = i8x32::splat(-0x80);
        let b = i8x32::splat(-1);
        let r = avx2::_mm256_adds_epi8(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_adds_epi16() {
        let a =
            i16x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = i16x16::new(
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
        );
        let r = avx2::_mm256_adds_epi16(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i16x16::new(
            32, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62,
        );

        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_adds_epi16_saturate_positive() {
        let a = i16x16::splat(0x7FFF);
        let b = i16x16::splat(1);
        let r = avx2::_mm256_adds_epi16(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_adds_epi16_saturate_negative() {
        let a = i16x16::splat(-0x8000);
        let b = i16x16::splat(-1);
        let r = avx2::_mm256_adds_epi16(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_adds_epu8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = u8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = u8x32::new(
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63,
        );
        let r = avx2::_mm256_adds_epu8(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = u8x32::new(
            32, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62,
            64, 66, 68, 70, 72, 74, 76, 78,
            80, 82, 84, 86, 88, 90, 92, 94,
        );
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_adds_epu8_saturate() {
        let a = u8x32::splat(0xFF);
        let b = u8x32::splat(1);
        let r = avx2::_mm256_adds_epu8(a, b);
        assert_eq!(r, a);
    }


    #[simd_test = "avx2"]
    unsafe fn _mm256_adds_epu16() {
        let a =
            u16x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = u16x16::new(
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
        );
        let r = avx2::_mm256_adds_epu16(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = u16x16::new(
            32, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62,
        );

        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_adds_epu16_saturate() {
        let a = u16x16::splat(0xFFFF);
        let b = u16x16::splat(1);
        let r = avx2::_mm256_adds_epu16(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_and_si256() {
        let a = __m256i::splat(5);
        let b = __m256i::splat(3);
        let got = avx2::_mm256_and_si256(a, b);
        assert_eq!(got, __m256i::splat(1));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_andnot_si256() {
        let a = __m256i::splat(5);
        let b = __m256i::splat(3);
        let got = avx2::_mm256_andnot_si256(a, b);
        assert_eq!(got, __m256i::splat(2));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_avg_epu8() {
        let (a, b) = (u8x32::splat(3), u8x32::splat(9));
        let r = avx2::_mm256_avg_epu8(a, b);
        assert_eq!(r, u8x32::splat(6));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_avg_epu16() {
        let (a, b) = (u16x16::splat(3), u16x16::splat(9));
        let r = avx2::_mm256_avg_epu16(a, b);
        assert_eq!(r, u16x16::splat(6));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_blend_epi32() {
        let (a, b) = (i32x4::splat(3), i32x4::splat(9));
        let e = i32x4::splat(3).replace(0, 9);
        let r = avx2::_mm_blend_epi32(a, b, 0x01 as i32);
        assert_eq!(r, e);

        let r = avx2::_mm_blend_epi32(b, a, 0x0E as i32);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_blend_epi32() {
        let (a, b) = (i32x8::splat(3), i32x8::splat(9));
        let e = i32x8::splat(3).replace(0, 9);
        let r = avx2::_mm256_blend_epi32(a, b, 0x01 as i32);
        assert_eq!(r, e);

        let e = i32x8::splat(3).replace(1, 9).replace(7, 9);
        let r = avx2::_mm256_blend_epi32(a, b, 0x82 as i32);
        assert_eq!(r, e);

        let e = i32x8::splat(9).replace(0, 3).replace(1, 3).replace(7, 3);
        let r = avx2::_mm256_blend_epi32(a, b, 0x7C as i32);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_blend_epi16() {
        let (a, b) = (i16x16::splat(3), i16x16::splat(9));
        let e = i16x16::splat(3).replace(0, 9).replace(8, 9);
        let r = avx2::_mm256_blend_epi16(a, b, 0x01 as i32);
        assert_eq!(r, e);

        let r = avx2::_mm256_blend_epi16(b, a, 0xFE as i32);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_blendv_epi8() {
        let (a, b) = (i8x32::splat(4), i8x32::splat(2));
        let mask = i8x32::splat(0).replace(2, -1);
        let e = i8x32::splat(4).replace(2, 2);
        let r = avx2::_mm256_blendv_epi8(a, b, mask);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_broadcastb_epi8() {
        let a = i8x16::splat(0x00).replace(0, 0x2a);
        let res = avx2::_mm_broadcastb_epi8(a);
        assert_eq!(res, i8x16::splat(0x2a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_broadcastb_epi8() {
        let a = i8x16::splat(0x00).replace(0, 0x2a);
        let res = avx2::_mm256_broadcastb_epi8(a);
        assert_eq!(res, i8x32::splat(0x2a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_broadcastd_epi32() {
        let a = i32x4::splat(0x00).replace(0, 0x2a).replace(1, 0x8000000);
        let res = avx2::_mm_broadcastd_epi32(a);
        assert_eq!(res, i32x4::splat(0x2a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_broadcastd_epi32() {
        let a = i32x4::splat(0x00).replace(0, 0x2a).replace(1, 0x8000000);
        let res = avx2::_mm256_broadcastd_epi32(a);
        assert_eq!(res, i32x8::splat(0x2a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_broadcastq_epi64() {
        let a = i64x2::splat(0x00).replace(0, 0x1ffffffff);
        let res = avx2::_mm_broadcastq_epi64(a);
        assert_eq!(res, i64x2::splat(0x1ffffffff));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_broadcastq_epi64() {
        let a = i64x2::splat(0x00).replace(0, 0x1ffffffff);
        let res = avx2::_mm256_broadcastq_epi64(a);
        assert_eq!(res, i64x4::splat(0x1ffffffff));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_broadcastsd_pd() {
        let a = f64x2::splat(3.14f64).replace(0, 6.28f64);
        let res = avx2::_mm_broadcastsd_pd(a);
        assert_eq!(res, f64x2::splat(6.28f64));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_broadcastsd_pd() {
        let a = f64x2::splat(3.14f64).replace(0, 6.28f64);
        let res = avx2::_mm256_broadcastsd_pd(a);
        assert_eq!(res, f64x4::splat(6.28f64));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_broadcastsi128_si256() {
        let a = i64x2::new(0x0987654321012334, 0x5678909876543210);
        let res = avx2::_mm256_broadcastsi128_si256(a);
        let retval = i64x4::new(
            0x0987654321012334,
            0x5678909876543210,
            0x0987654321012334,
            0x5678909876543210,
        );
        assert_eq!(res, retval);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_broadcastss_ps() {
        let a = f32x4::splat(3.14f32).replace(0, 6.28f32);
        let res = avx2::_mm_broadcastss_ps(a);
        assert_eq!(res, f32x4::splat(6.28f32));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_broadcastss_ps() {
        let a = f32x4::splat(3.14f32).replace(0, 6.28f32);
        let res = avx2::_mm256_broadcastss_ps(a);
        assert_eq!(res, f32x8::splat(6.28f32));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_broadcastw_epi16() {
        let a = i16x8::splat(0x2a).replace(0, 0x22b);
        let res = avx2::_mm_broadcastw_epi16(a);
        assert_eq!(res, i16x8::splat(0x22b));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_broadcastw_epi16() {
        let a = i16x8::splat(0x2a).replace(0, 0x22b);
        let res = avx2::_mm256_broadcastw_epi16(a);
        assert_eq!(res, i16x16::splat(0x22b));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cmpeq_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = i8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = i8x32::new(
            31, 30, 2, 28, 27, 26, 25, 24,
            23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0,
        );
        let r = avx2::_mm256_cmpeq_epi8(a, b);
        assert_eq!(r, i8x32::splat(0).replace(2, 0xFFu8 as i8));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cmpeq_epi16() {
        let a =
            i16x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b =
            i16x16::new(15, 14, 2, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = avx2::_mm256_cmpeq_epi16(a, b);
        assert_eq!(r, i16x16::splat(0).replace(2, 0xFFFFu16 as i16));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cmpeq_epi32() {
        let a = i32x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = i32x8::new(7, 6, 2, 4, 3, 2, 1, 0);
        let r = avx2::_mm256_cmpeq_epi32(a, b);
        assert_eq!(r, i32x8::splat(0).replace(2, 0xFFFFFFFFu32 as i32));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cmpeq_epi64() {
        let a = i64x4::new(0, 1, 2, 3);
        let b = i64x4::new(3, 2, 2, 0);
        let r = avx2::_mm256_cmpeq_epi64(a, b);
        assert_eq!(
            r,
            i64x4::splat(0).replace(2, 0xFFFFFFFFFFFFFFFFu64 as i64)
        );
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cmpgt_epi8() {
        let a = i8x32::splat(0).replace(0, 5);
        let b = i8x32::splat(0);
        let r = avx2::_mm256_cmpgt_epi8(a, b);
        assert_eq!(r, i8x32::splat(0).replace(0, 0xFFu8 as i8));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cmpgt_epi16() {
        let a = i16x16::splat(0).replace(0, 5);
        let b = i16x16::splat(0);
        let r = avx2::_mm256_cmpgt_epi16(a, b);
        assert_eq!(r, i16x16::splat(0).replace(0, 0xFFFFu16 as i16));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cmpgt_epi32() {
        let a = i32x8::splat(0).replace(0, 5);
        let b = i32x8::splat(0);
        let r = avx2::_mm256_cmpgt_epi32(a, b);
        assert_eq!(r, i32x8::splat(0).replace(0, 0xFFFFFFFFu32 as i32));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cmpgt_epi64() {
        let a = i64x4::splat(0).replace(0, 5);
        let b = i64x4::splat(0);
        let r = avx2::_mm256_cmpgt_epi64(a, b);
        assert_eq!(
            r,
            i64x4::splat(0).replace(0, 0xFFFFFFFFFFFFFFFFu64 as i64)
        );
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cvtepi8_epi16() {
        let a =
            i8x16::new(0, 0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7);
        let r =
            i16x16::new(0, 0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7);
        assert_eq!(r, avx2::_mm256_cvtepi8_epi16(a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cvtepi8_epi32() {
        let a =
            i8x16::new(0, 0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7);
        let r = i32x8::new(0, 0, -1, 1, -2, 2, -3, 3);
        assert_eq!(r, avx2::_mm256_cvtepi8_epi32(a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cvtepi8_epi64() {
        let a =
            i8x16::new(0, 0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7);
        let r = i64x4::new(0, 0, -1, 1);
        assert_eq!(r, avx2::_mm256_cvtepi8_epi64(a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cvtepi16_epi32() {
        let a = i16x8::new(0, 0, -1, 1, -2, 2, -3, 3);
        let r = i32x8::new(0, 0, -1, 1, -2, 2, -3, 3);
        assert_eq!(r, avx2::_mm256_cvtepi16_epi32(a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cvtepi16_epi64() {
        let a = i16x8::new(0, 0, -1, 1, -2, 2, -3, 3);
        let r = i64x4::new(0, 0, -1, 1);
        assert_eq!(r, avx2::_mm256_cvtepi16_epi64(a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cvtepi32_epi64() {
        let a = i32x4::new(0, 0, -1, 1);
        let r = i64x4::new(0, 0, -1, 1);
        assert_eq!(r, avx2::_mm256_cvtepi32_epi64(a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cvtepu16_epi32() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = i32x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq!(r, avx2::_mm256_cvtepu16_epi32(a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cvtepu16_epi64() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = i64x4::new(0, 1, 2, 3);
        assert_eq!(r, avx2::_mm256_cvtepu16_epi64(a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cvtepu32_epi64() {
        let a = u32x4::new(0, 1, 2, 3);
        let r = i64x4::new(0, 1, 2, 3);
        assert_eq!(r, avx2::_mm256_cvtepu32_epi64(a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cvtepu8_epi16() {
        let a =
            u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r =
            i16x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq!(r, avx2::_mm256_cvtepu8_epi16(a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cvtepu8_epi32() {
        let a =
            u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = i32x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq!(r, avx2::_mm256_cvtepu8_epi32(a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_cvtepu8_epi64() {
        let a =
            u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = i64x4::new(0, 1, 2, 3);
        assert_eq!(r, avx2::_mm256_cvtepu8_epi64(a));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_extracti128_si256() {
        let a = __m256i::from(i64x4::new(1, 2, 3, 4));
        let r = avx2::_mm256_extracti128_si256(a, 0b01);
        let e = __m128i::from(i64x2::new(3, 4));
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_hadd_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_hadd_epi16(a, b);
        let e = i16x16::new(4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_hadd_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_hadd_epi32(a, b);
        let e = i32x8::new(4, 4, 8, 8, 4, 4, 8, 8);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_hadds_epi16() {
        let a = i16x16::splat(2).replace(0, 0x7FFF).replace(1, 1);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_hadds_epi16(a, b);
        let e =
            i16x16::new(0x7FFF, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_hsub_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_hsub_epi16(a, b);
        let e = i16x16::splat(0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_hsub_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_hsub_epi32(a, b);
        let e = i32x8::splat(0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_hsubs_epi16() {
        let a = i16x16::splat(2).replace(0, 0x7FFF).replace(1, -1);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_hsubs_epi16(a, b);
        let e = i16x16::splat(0).replace(0, 0x7FFF);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_madd_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_madd_epi16(a, b);
        let e = i32x8::splat(16);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_inserti128_si256() {
        let a = __m256i::from(i64x4::new(1, 2, 3, 4));
        let b = __m128i::from(i64x2::new(7, 8));
        let r = avx2::_mm256_inserti128_si256(a, b, 0b01);
        let e = i64x4::new(1, 2, 7, 8);
        assert_eq!(r, __m256i::from(e));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_maddubs_epi16() {
        let a = u8x32::splat(2);
        let b = u8x32::splat(4);
        let r = avx2::_mm256_maddubs_epi16(a, b);
        let e = i16x16::splat(16);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_maskload_epi32() {
        let nums = [1, 2, 3, 4];
        let a = &nums as *const i32;
        let mask = i32x4::new(-1, 0, 0, -1);
        let r = avx2::_mm_maskload_epi32(a, mask);
        let e = i32x4::new(1, 0, 0, 4);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_maskload_epi32() {
        let nums = [1, 2, 3, 4, 5, 6, 7, 8];
        let a = &nums as *const i32;
        let mask = i32x8::new(-1, 0, 0, -1, 0, -1, -1, 0);
        let r = avx2::_mm256_maskload_epi32(a, mask);
        let e = i32x8::new(1, 0, 0, 4, 0, 6, 7, 0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_maskload_epi64() {
        let nums = [1_i64, 2_i64];
        let a = &nums as *const i64;
        let mask = i64x2::new(0, -1);
        let r = avx2::_mm_maskload_epi64(a, mask);
        let e = i64x2::new(0, 2);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_maskload_epi64() {
        let nums = [1_i64, 2_i64, 3_i64, 4_i64];
        let a = &nums as *const i64;
        let mask = i64x4::new(0, -1, -1, 0);
        let r = avx2::_mm256_maskload_epi64(a, mask);
        let e = i64x4::new(0, 2, 3, 0);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_maskstore_epi32() {
        let a = i32x4::new(1, 2, 3, 4);
        let mut arr = [-1, -1, -1, -1];
        let mask = i32x4::new(-1, 0, 0, -1);
        avx2::_mm_maskstore_epi32(arr.as_mut_ptr(), mask, a);
        let e = [1, -1, -1, 4];
        assert_eq!(arr, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_maskstore_epi32() {
        let a = i32x8::new(1, 0x6d726f, 3, 42, 0x777161, 6, 7, 8);
        let mut arr = [-1, -1, -1, 0x776173, -1, 0x68657265, -1, -1];
        let mask = i32x8::new(-1, 0, 0, -1, 0, -1, -1, 0);
        avx2::_mm256_maskstore_epi32(arr.as_mut_ptr(), mask, a);
        let e = [1, -1, -1, 42, -1, 6, 7, -1];
        assert_eq!(arr, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_maskstore_epi64() {
        let a = i64x2::new(1_i64, 2_i64);
        let mut arr = [-1_i64, -1_i64];
        let mask = i64x2::new(0, -1);
        avx2::_mm_maskstore_epi64(arr.as_mut_ptr(), mask, a);
        let e = [-1, 2];
        assert_eq!(arr, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_maskstore_epi64() {
        let a = i64x4::new(1_i64, 2_i64, 3_i64, 4_i64);
        let mut arr = [-1_i64, -1_i64, -1_i64, -1_i64];
        let mask = i64x4::new(0, -1, -1, 0);
        avx2::_mm256_maskstore_epi64(arr.as_mut_ptr(), mask, a);
        let e = [-1, 2, 3, -1];
        assert_eq!(arr, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_max_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_max_epi16(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_max_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_max_epi32(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_max_epi8() {
        let a = i8x32::splat(2);
        let b = i8x32::splat(4);
        let r = avx2::_mm256_max_epi8(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_max_epu16() {
        let a = u16x16::splat(2);
        let b = u16x16::splat(4);
        let r = avx2::_mm256_max_epu16(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_max_epu32() {
        let a = u32x8::splat(2);
        let b = u32x8::splat(4);
        let r = avx2::_mm256_max_epu32(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_max_epu8() {
        let a = u8x32::splat(2);
        let b = u8x32::splat(4);
        let r = avx2::_mm256_max_epu8(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_min_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_min_epi16(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_min_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_min_epi32(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_min_epi8() {
        let a = i8x32::splat(2);
        let b = i8x32::splat(4);
        let r = avx2::_mm256_min_epi8(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_min_epu16() {
        let a = u16x16::splat(2);
        let b = u16x16::splat(4);
        let r = avx2::_mm256_min_epu16(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_min_epu32() {
        let a = u32x8::splat(2);
        let b = u32x8::splat(4);
        let r = avx2::_mm256_min_epu32(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_min_epu8() {
        let a = u8x32::splat(2);
        let b = u8x32::splat(4);
        let r = avx2::_mm256_min_epu8(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_movemask_epi8() {
        let a = i8x32::splat(-1);
        let r = avx2::_mm256_movemask_epi8(a);
        let e = -1;
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mpsadbw_epu8() {
        let a = u8x32::splat(2);
        let b = u8x32::splat(4);
        let r = avx2::_mm256_mpsadbw_epu8(a, b, 0);
        let e = u16x16::splat(8);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mul_epi32() {
        let a = i32x8::new(0, 0, 0, 0, 2, 2, 2, 2);
        let b = i32x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = avx2::_mm256_mul_epi32(a, b);
        let e = i64x4::new(0, 0, 10, 14);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mul_epu32() {
        let a = u32x8::new(0, 0, 0, 0, 2, 2, 2, 2);
        let b = u32x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = avx2::_mm256_mul_epu32(a, b);
        let e = u64x4::new(0, 0, 10, 14);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mulhi_epi16() {
        let a = i16x16::splat(6535);
        let b = i16x16::splat(6535);
        let r = avx2::_mm256_mulhi_epi16(a, b);
        let e = i16x16::splat(651);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mulhi_epu16() {
        let a = u16x16::splat(6535);
        let b = u16x16::splat(6535);
        let r = avx2::_mm256_mulhi_epu16(a, b);
        let e = u16x16::splat(651);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mullo_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_mullo_epi16(a, b);
        let e = i16x16::splat(8);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mullo_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_mullo_epi32(a, b);
        let e = i32x8::splat(8);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mulhrs_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_mullo_epi16(a, b);
        let e = i16x16::splat(8);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_or_si256() {
        let a = __m256i::splat(-1);
        let b = __m256i::splat(0);
        let r = avx2::_mm256_or_si256(a, b);
        assert_eq!(r, a);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_packs_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_packs_epi16(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i8x32::new(
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4,
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4,
        );

        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_packs_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_packs_epi32(a, b);
        let e = i16x16::new(2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4);

        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_packus_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_packus_epi16(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = u8x32::new(
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4,
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4,
        );

        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_packus_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_packus_epi32(a, b);
        let e = u16x16::new(2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4);

        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sad_epu8() {
        let a = u8x32::splat(2);
        let b = u8x32::splat(4);
        let r = avx2::_mm256_sad_epu8(a, b);
        let e = u64x4::splat(16);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_shufflehi_epi16() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = i16x16::new(
            0, 1, 2, 3, 11, 22, 33, 44,
            4, 5, 6, 7, 55, 66, 77, 88,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i16x16::new(
            0, 1, 2, 3, 44, 22, 22, 11,
            4, 5, 6, 7, 88, 66, 66, 55,
        );
        let r = avx2::_mm256_shufflehi_epi16(a, 0b00_01_01_11);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_shufflelo_epi16() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = i16x16::new(
            11, 22, 33, 44, 0, 1, 2, 3,
            55, 66, 77, 88, 4, 5, 6, 7,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = i16x16::new(
            44, 22, 22, 11, 0, 1, 2, 3,
            88, 66, 66, 55, 4, 5, 6, 7,
        );
        let r = avx2::_mm256_shufflelo_epi16(a, 0b00_01_01_11);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sign_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(-1);
        let r = avx2::_mm256_sign_epi16(a, b);
        let e = i16x16::splat(-2);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sign_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(-1);
        let r = avx2::_mm256_sign_epi32(a, b);
        let e = i32x8::splat(-2);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sign_epi8() {
        let a = i8x32::splat(2);
        let b = i8x32::splat(-1);
        let r = avx2::_mm256_sign_epi8(a, b);
        let e = i8x32::splat(-2);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sll_epi16() {
        let a = i16x16::splat(0xFF);
        let b = i16x8::splat(0).replace(0, 4);
        let r = avx2::_mm256_sll_epi16(a, b);
        assert_eq!(r, i16x16::splat(0xFF0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sll_epi32() {
        let a = i32x8::splat(0xFFFF);
        let b = i32x4::splat(0).replace(0, 4);
        let r = avx2::_mm256_sll_epi32(a, b);
        assert_eq!(r, i32x8::splat(0xFFFF0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sll_epi64() {
        let a = i64x4::splat(0xFFFFFFFF);
        let b = i64x2::splat(0).replace(0, 4);
        let r = avx2::_mm256_sll_epi64(a, b);
        assert_eq!(r, i64x4::splat(0xFFFFFFFF0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_slli_epi16() {
        assert_eq!(
            avx2::_mm256_slli_epi16(i16x16::splat(0xFF), 4),
            i16x16::splat(0xFF0)
        );
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_slli_epi32() {
        assert_eq!(
            avx2::_mm256_slli_epi32(i32x8::splat(0xFFFF), 4),
            i32x8::splat(0xFFFF0)
        );
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_slli_epi64() {
        assert_eq!(
            avx2::_mm256_slli_epi64(i64x4::splat(0xFFFFFFFF), 4),
            i64x4::splat(0xFFFFFFFF0)
        );
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_sllv_epi32() {
        let a = i32x4::splat(2);
        let b = i32x4::splat(1);
        let r = avx2::_mm_sllv_epi32(a, b);
        let e = i32x4::splat(4);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sllv_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(1);
        let r = avx2::_mm256_sllv_epi32(a, b);
        let e = i32x8::splat(4);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_sllv_epi64() {
        let a = i64x2::splat(2);
        let b = i64x2::splat(1);
        let r = avx2::_mm_sllv_epi64(a, b);
        let e = i64x2::splat(4);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sllv_epi64() {
        let a = i64x4::splat(2);
        let b = i64x4::splat(1);
        let r = avx2::_mm256_sllv_epi64(a, b);
        let e = i64x4::splat(4);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sra_epi16() {
        let a = i16x16::splat(-1);
        let b = i16x8::new(1, 0, 0, 0, 0, 0, 0, 0);
        let r = avx2::_mm256_sra_epi16(a, b);
        assert_eq!(r, i16x16::splat(-1));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sra_epi32() {
        let a = i32x8::splat(-1);
        let b = i32x4::splat(0).replace(0, 1);
        let r = avx2::_mm256_sra_epi32(a, b);
        assert_eq!(r, i32x8::splat(-1));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_srai_epi16() {
        assert_eq!(
            avx2::_mm256_srai_epi16(i16x16::splat(-1), 1),
            i16x16::splat(-1)
        );
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_srai_epi32() {
        assert_eq!(
            avx2::_mm256_srai_epi32(i32x8::splat(-1), 1),
            i32x8::splat(-1)
        );
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_srav_epi32() {
        let a = i32x4::splat(4);
        let count = i32x4::splat(1);
        let r = avx2::_mm_srav_epi32(a, count);
        let e = i32x4::splat(2);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_srav_epi32() {
        let a = i32x8::splat(4);
        let count = i32x8::splat(1);
        let r = avx2::_mm256_srav_epi32(a, count);
        let e = i32x8::splat(2);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_srl_epi16() {
        let a = i16x16::splat(0xFF);
        let b = i16x8::splat(0).replace(0, 4);
        let r = avx2::_mm256_srl_epi16(a, b);
        assert_eq!(r, i16x16::splat(0xF));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_srl_epi32() {
        let a = i32x8::splat(0xFFFF);
        let b = i32x4::splat(0).replace(0, 4);
        let r = avx2::_mm256_srl_epi32(a, b);
        assert_eq!(r, i32x8::splat(0xFFF));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_srl_epi64() {
        let a = i64x4::splat(0xFFFFFFFF);
        let b = i64x2::splat(0).replace(0, 4);
        let r = avx2::_mm256_srl_epi64(a, b);
        assert_eq!(r, i64x4::splat(0xFFFFFFF));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_srli_epi16() {
        assert_eq!(
            avx2::_mm256_srli_epi16(i16x16::splat(0xFF), 4),
            i16x16::splat(0xF)
        );
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_srli_epi32() {
        assert_eq!(
            avx2::_mm256_srli_epi32(i32x8::splat(0xFFFF), 4),
            i32x8::splat(0xFFF)
        );
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_srli_epi64() {
        assert_eq!(
            avx2::_mm256_srli_epi64(i64x4::splat(0xFFFFFFFF), 4),
            i64x4::splat(0xFFFFFFF)
        );
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_srlv_epi32() {
        let a = i32x4::splat(2);
        let count = i32x4::splat(1);
        let r = avx2::_mm_srlv_epi32(a, count);
        let e = i32x4::splat(1);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_srlv_epi32() {
        let a = i32x8::splat(2);
        let count = i32x8::splat(1);
        let r = avx2::_mm256_srlv_epi32(a, count);
        let e = i32x8::splat(1);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_srlv_epi64() {
        let a = i64x2::splat(2);
        let count = i64x2::splat(1);
        let r = avx2::_mm_srlv_epi64(a, count);
        let e = i64x2::splat(1);
        assert_eq!(r, e);
    }


    #[simd_test = "avx2"]
    unsafe fn _mm256_srlv_epi64() {
        let a = i64x4::splat(2);
        let count = i64x4::splat(1);
        let r = avx2::_mm256_srlv_epi64(a, count);
        let e = i64x4::splat(1);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sub_epi16() {
        let a = i16x16::splat(4);
        let b = i16x16::splat(2);
        let r = avx2::_mm256_sub_epi16(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sub_epi32() {
        let a = i32x8::splat(4);
        let b = i32x8::splat(2);
        let r = avx2::_mm256_sub_epi32(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sub_epi64() {
        let a = i64x4::splat(4);
        let b = i64x4::splat(2);
        let r = avx2::_mm256_sub_epi64(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_sub_epi8() {
        let a = i8x32::splat(4);
        let b = i8x32::splat(2);
        let r = avx2::_mm256_sub_epi8(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_subs_epi16() {
        let a = i16x16::splat(4);
        let b = i16x16::splat(2);
        let r = avx2::_mm256_subs_epi16(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_subs_epi8() {
        let a = i8x32::splat(4);
        let b = i8x32::splat(2);
        let r = avx2::_mm256_subs_epi8(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_subs_epu16() {
        let a = u16x16::splat(4);
        let b = u16x16::splat(2);
        let r = avx2::_mm256_subs_epu16(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_subs_epu8() {
        let a = u8x32::splat(4);
        let b = u8x32::splat(2);
        let r = avx2::_mm256_subs_epu8(a, b);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_xor_si256() {
        let a = __m256i::splat(5);
        let b = __m256i::splat(3);
        let r = avx2::_mm256_xor_si256(a, b);
        assert_eq!(r, __m256i::splat(6));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_alignr_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = i8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = i8x32::new(
            -1, -2, -3, -4, -5, -6, -7, -8,
            -9, -10, -11, -12, -13, -14, -15, -16,
            -17, -18, -19, -20, -21, -22, -23, -24,
            -25, -26, -27, -28, -29, -30, -31, -32,
        );
        let r = avx2::_mm256_alignr_epi8(a, b, 33);
        assert_eq!(r, i8x32::splat(0));

        let r = avx2::_mm256_alignr_epi8(a, b, 17);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = i8x32::new(
            2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 0,
        );
        assert_eq!(r, expected);

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = i8x32::new(
            -17, -18, -19, -20, -21, -22, -23, -24,
            -25, -26, -27, -28, -29, -30, -31, -32,
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        );
        let r = avx2::_mm256_alignr_epi8(a, b, 16);
        assert_eq!(r, expected);

        let r = avx2::_mm256_alignr_epi8(a, b, 15);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = i8x32::new(
            -16, -17, -18, -19, -20, -21, -22, -23,
            -24, -25, -26, -27, -28, -29, -30, -31,
            -32, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        assert_eq!(r, expected);

        let r = avx2::_mm256_alignr_epi8(a, b, 0);
        assert_eq!(r, b);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_shuffle_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = u8x32::new(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = u8x32::new(
            4, 128, 4, 3, 24, 12, 6, 19,
            12, 5, 5, 10, 4, 1, 8, 0,
            4, 128, 4, 3, 24, 12, 6, 19,
            12, 5, 5, 10, 4, 1, 8, 0,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = u8x32::new(
            5, 0, 5, 4, 9, 13, 7, 4,
            13, 6, 6, 11, 5, 2, 9, 1,
            21, 0, 21, 20, 25, 29, 23, 20,
            29, 22, 22, 27, 21, 18, 25, 17,
        );
        let r = avx2::_mm256_shuffle_epi8(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_permutevar8x32_epi32() {
        let a = u32x8::new(100, 200, 300, 400, 500, 600, 700, 800);
        let b = u32x8::new(5, 0, 5, 1, 7, 6, 3, 4);
        let expected = u32x8::new(600, 100, 600, 200, 800, 700, 400, 500);
        let r = avx2::_mm256_permutevar8x32_epi32(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_permute4x64_epi64() {
        let a = i64x4::new(100, 200, 300, 400);
        let expected = i64x4::new(400, 100, 200, 100);
        let r = avx2::_mm256_permute4x64_epi64(a, 0b00010011);
        assert_eq!(r, expected);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_permute2x128_si256() {
        let a = __m256i::from(i64x4::new(100, 200, 500, 600));
        let b = __m256i::from(i64x4::new(300, 400, 700, 800));
        let r = avx2::_mm256_permute2x128_si256(a, b, 0b00_01_00_11);
        let e = i64x4::new(700, 800, 500, 600);
        assert_eq!(i64x4::from(r), e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_permute4x64_pd() {
        let a = f64x4::new(1., 2., 3., 4.);
        let r = avx2::_mm256_permute4x64_pd(a, 0b00_01_00_11);
        let e = f64x4::new(4., 1., 2., 1.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_permutevar8x32_ps() {
        let a = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = i32x8::new(5, 0, 5, 1, 7, 6, 3, 4);
        let r = avx2::_mm256_permutevar8x32_ps(a, b);
        let e = f32x8::new(6., 1., 6., 2., 8., 7., 4., 5.);
        assert_eq!(r, e);
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_i32gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = avx2::_mm_i32gather_epi32(
            arr.as_ptr(),
            i32x4::new(0, 16, 32, 48),
            4,
        );
        assert_eq!(r, i32x4::new(0, 16, 32, 48));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_mask_i32gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = avx2::_mm_mask_i32gather_epi32(
            i32x4::splat(256),
            arr.as_ptr(),
            i32x4::new(0, 16, 64, 96),
            i32x4::new(-1, -1, -1, 0),
            4,
        );
        assert_eq!(r, i32x4::new(0, 16, 64, 256));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_i32gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = avx2::_mm256_i32gather_epi32(
            arr.as_ptr(),
            i32x8::new(0, 16, 32, 48, 1, 2, 3, 4),
            4,
        );
        assert_eq!(r, i32x8::new(0, 16, 32, 48, 1, 2, 3, 4));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mask_i32gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = avx2::_mm256_mask_i32gather_epi32(
            i32x8::splat(256),
            arr.as_ptr(),
            i32x8::new(0, 16, 64, 96, 0, 0, 0, 0),
            i32x8::new(-1, -1, -1, 0, 0, 0, 0, 0),
            4,
        );
        assert_eq!(r, i32x8::new(0, 16, 64, 256, 256, 256, 256, 256));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_i32gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r =
            avx2::_mm_i32gather_ps(arr.as_ptr(), i32x4::new(0, 16, 32, 48), 4);
        assert_eq!(r, f32x4::new(0.0, 16.0, 32.0, 48.0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_mask_i32gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r = avx2::_mm_mask_i32gather_ps(
            f32x4::splat(256.0),
            arr.as_ptr(),
            i32x4::new(0, 16, 64, 96),
            f32x4::new(-1.0, -1.0, -1.0, 0.0),
            4,
        );
        assert_eq!(r, f32x4::new(0.0, 16.0, 64.0, 256.0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_i32gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r = avx2::_mm256_i32gather_ps(
            arr.as_ptr(),
            i32x8::new(0, 16, 32, 48, 1, 2, 3, 4),
            4,
        );
        assert_eq!(r, f32x8::new(0.0, 16.0, 32.0, 48.0, 1.0, 2.0, 3.0, 4.0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mask_i32gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r = avx2::_mm256_mask_i32gather_ps(
            f32x8::splat(256.0),
            arr.as_ptr(),
            i32x8::new(0, 16, 64, 96, 0, 0, 0, 0),
            f32x8::new(-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            4,
        );
        assert_eq!(
            r,
            f32x8::new(0.0, 16.0, 64.0, 256.0, 256.0, 256.0, 256.0, 256.0)
        );
    }


    #[simd_test = "avx2"]
    unsafe fn _mm_i32gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = avx2::_mm_i32gather_epi64(
            arr.as_ptr(),
            i32x4::new(0, 16, 0, 0),
            8,
        );
        assert_eq!(r, i64x2::new(0, 16));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_mask_i32gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = avx2::_mm_mask_i32gather_epi64(
            i64x2::splat(256),
            arr.as_ptr(),
            i32x4::new(16, 16, 16, 16),
            i64x2::new(-1, 0),
            8,
        );
        assert_eq!(r, i64x2::new(16, 256));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_i32gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = avx2::_mm256_i32gather_epi64(
            arr.as_ptr(),
            i32x4::new(0, 16, 32, 48),
            8,
        );
        assert_eq!(r, i64x4::new(0, 16, 32, 48));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mask_i32gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = avx2::_mm256_mask_i32gather_epi64(
            i64x4::splat(256),
            arr.as_ptr(),
            i32x4::new(0, 16, 64, 96),
            i64x4::new(-1, -1, -1, 0),
            8,
        );
        assert_eq!(r, i64x4::new(0, 16, 64, 256));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_i32gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r =
            avx2::_mm_i32gather_pd(arr.as_ptr(), i32x4::new(0, 16, 0, 0), 8);
        assert_eq!(r, f64x2::new(0.0, 16.0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_mask_i32gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = avx2::_mm_mask_i32gather_pd(
            f64x2::splat(256.0),
            arr.as_ptr(),
            i32x4::new(16, 16, 16, 16),
            f64x2::new(-1.0, 0.0),
            8,
        );
        assert_eq!(r, f64x2::new(16.0, 256.0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_i32gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = avx2::_mm256_i32gather_pd(
            arr.as_ptr(),
            i32x4::new(0, 16, 32, 48),
            8,
        );
        assert_eq!(r, f64x4::new(0.0, 16.0, 32.0, 48.0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mask_i32gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = avx2::_mm256_mask_i32gather_pd(
            f64x4::splat(256.0),
            arr.as_ptr(),
            i32x4::new(0, 16, 64, 96),
            f64x4::new(-1.0, -1.0, -1.0, 0.0),
            8,
        );
        assert_eq!(r, f64x4::new(0.0, 16.0, 64.0, 256.0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_i64gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = avx2::_mm_i64gather_epi32(arr.as_ptr(), i64x2::new(0, 16), 4);
        assert_eq!(r, i32x4::new(0, 16, 0, 0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_mask_i64gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = avx2::_mm_mask_i64gather_epi32(
            i32x4::splat(256),
            arr.as_ptr(),
            i64x2::new(0, 16),
            i32x4::new(-1, 0, -1, 0),
            4,
        );
        assert_eq!(r, i32x4::new(0, 256, 0, 0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_i64gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = avx2::_mm256_i64gather_epi32(
            arr.as_ptr(),
            i64x4::new(0, 16, 32, 48),
            4,
        );
        assert_eq!(r, i32x4::new(0, 16, 32, 48));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mask_i64gather_epi32() {
        let mut arr = [0i32; 128];
        for i in 0..128i32 {
            arr[i as usize] = i;
        }
        // A multiplier of 4 is word-addressing
        let r = avx2::_mm256_mask_i64gather_epi32(
            i32x4::splat(256),
            arr.as_ptr(),
            i64x4::new(0, 16, 64, 96),
            i32x4::new(-1, -1, -1, 0),
            4,
        );
        assert_eq!(r, i32x4::new(0, 16, 64, 256));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_i64gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r = avx2::_mm_i64gather_ps(arr.as_ptr(), i64x2::new(0, 16), 4);
        assert_eq!(r, f32x4::new(0.0, 16.0, 0.0, 0.0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_mask_i64gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r = avx2::_mm_mask_i64gather_ps(
            f32x4::splat(256.0),
            arr.as_ptr(),
            i64x2::new(0, 16),
            f32x4::new(-1.0, 0.0, -1.0, 0.0),
            4,
        );
        assert_eq!(r, f32x4::new(0.0, 256.0, 0.0, 0.0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_i64gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r = avx2::_mm256_i64gather_ps(
            arr.as_ptr(),
            i64x4::new(0, 16, 32, 48),
            4,
        );
        assert_eq!(r, f32x4::new(0.0, 16.0, 32.0, 48.0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mask_i64gather_ps() {
        let mut arr = [0.0f32; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 4 is word-addressing for f32s
        let r = avx2::_mm256_mask_i64gather_ps(
            f32x4::splat(256.0),
            arr.as_ptr(),
            i64x4::new(0, 16, 64, 96),
            f32x4::new(-1.0, -1.0, -1.0, 0.0),
            4,
        );
        assert_eq!(r, f32x4::new(0.0, 16.0, 64.0, 256.0));
    }


    #[simd_test = "avx2"]
    unsafe fn _mm_i64gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = avx2::_mm_i64gather_epi64(arr.as_ptr(), i64x2::new(0, 16), 8);
        assert_eq!(r, i64x2::new(0, 16));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_mask_i64gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = avx2::_mm_mask_i64gather_epi64(
            i64x2::splat(256),
            arr.as_ptr(),
            i64x2::new(16, 16),
            i64x2::new(-1, 0),
            8,
        );
        assert_eq!(r, i64x2::new(16, 256));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_i64gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = avx2::_mm256_i64gather_epi64(
            arr.as_ptr(),
            i64x4::new(0, 16, 32, 48),
            8,
        );
        assert_eq!(r, i64x4::new(0, 16, 32, 48));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mask_i64gather_epi64() {
        let mut arr = [0i64; 128];
        for i in 0..128i64 {
            arr[i as usize] = i;
        }
        // A multiplier of 8 is word-addressing for i64s
        let r = avx2::_mm256_mask_i64gather_epi64(
            i64x4::splat(256),
            arr.as_ptr(),
            i64x4::new(0, 16, 64, 96),
            i64x4::new(-1, -1, -1, 0),
            8,
        );
        assert_eq!(r, i64x4::new(0, 16, 64, 256));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_i64gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = avx2::_mm_i64gather_pd(arr.as_ptr(), i64x2::new(0, 16), 8);
        assert_eq!(r, f64x2::new(0.0, 16.0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm_mask_i64gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = avx2::_mm_mask_i64gather_pd(
            f64x2::splat(256.0),
            arr.as_ptr(),
            i64x2::new(16, 16),
            f64x2::new(-1.0, 0.0),
            8,
        );
        assert_eq!(r, f64x2::new(16.0, 256.0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_i64gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = avx2::_mm256_i64gather_pd(
            arr.as_ptr(),
            i64x4::new(0, 16, 32, 48),
            8,
        );
        assert_eq!(r, f64x4::new(0.0, 16.0, 32.0, 48.0));
    }

    #[simd_test = "avx2"]
    unsafe fn _mm256_mask_i64gather_pd() {
        let mut arr = [0.0f64; 128];
        let mut j = 0.0;
        for i in 0..128usize {
            arr[i] = j;
            j += 1.0;
        }
        // A multiplier of 8 is word-addressing for f64s
        let r = avx2::_mm256_mask_i64gather_pd(
            f64x4::splat(256.0),
            arr.as_ptr(),
            i64x4::new(0, 16, 64, 96),
            f64x4::new(-1.0, -1.0, -1.0, 0.0),
            8,
        );
        assert_eq!(r, f64x4::new(0.0, 16.0, 64.0, 256.0));
    }

}
