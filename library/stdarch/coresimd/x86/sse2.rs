//! Streaming SIMD Extensions 2 (SSE2)

#[cfg(test)]
use stdsimd_test::assert_instr;

use coresimd::simd::*;
use coresimd::simd_llvm::*;
use coresimd::x86::*;
use intrinsics;
use mem;
use ptr;

/// Provide a hint to the processor that the code sequence is a spin-wait loop.
///
/// This can help improve the performance and power consumption of spin-wait
/// loops.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pause))]
pub unsafe fn _mm_pause() {
    pause()
}

/// Invalidate and flush the cache line that contains `p` from all levels of
/// the cache hierarchy.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(clflush))]
pub unsafe fn _mm_clflush(p: *mut u8) {
    clflush(p)
}

/// Perform a serializing operation on all load-from-memory instructions
/// that were issued prior to this instruction.
///
/// Guarantees that every load instruction that precedes, in program order, is
/// globally visible before any load instruction which follows the fence in
/// program order.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(lfence))]
pub unsafe fn _mm_lfence() {
    lfence()
}

/// Perform a serializing operation on all load-from-memory and store-to-memory
/// instructions that were issued prior to this instruction.
///
/// Guarantees that every memory access that precedes, in program order, the
/// memory fence instruction is globally visible before any memory instruction
/// which follows the fence in program order.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(mfence))]
pub unsafe fn _mm_mfence() {
    mfence()
}

/// Add packed 8-bit integers in `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(paddb))]
pub unsafe fn _mm_add_epi8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(simd_add(a.as_i8x16(), b.as_i8x16()))
}

/// Add packed 16-bit integers in `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(paddw))]
pub unsafe fn _mm_add_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(simd_add(a.as_i16x8(), b.as_i16x8()))
}

/// Add packed 32-bit integers in `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(paddd))]
pub unsafe fn _mm_add_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(simd_add(a.as_i32x4(), b.as_i32x4()))
}

/// Add packed 64-bit integers in `a` and "b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(paddq))]
pub unsafe fn _mm_add_epi64(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(simd_add(a.as_i64x2(), b.as_i64x2()))
}

/// Add packed 8-bit integers in `a` and `b` using saturation.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(paddsb))]
pub unsafe fn _mm_adds_epi8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(paddsb(a.as_i8x16(), b.as_i8x16()))
}

/// Add packed 16-bit integers in `a` and `b` using saturation.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(paddsw))]
pub unsafe fn _mm_adds_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(paddsw(a.as_i16x8(), b.as_i16x8()))
}

/// Add packed unsigned 8-bit integers in `a` and `b` using saturation.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(paddusb))]
pub unsafe fn _mm_adds_epu8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(paddsub(a.as_u8x16(), b.as_u8x16()))
}

/// Add packed unsigned 16-bit integers in `a` and `b` using saturation.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(paddusw))]
pub unsafe fn _mm_adds_epu16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(paddsuw(a.as_u16x8(), b.as_u16x8()))
}

/// Average packed unsigned 8-bit integers in `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pavgb))]
pub unsafe fn _mm_avg_epu8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pavgb(a.as_u8x16(), b.as_u8x16()))
}

/// Average packed unsigned 16-bit integers in `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pavgw))]
pub unsafe fn _mm_avg_epu16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pavgw(a.as_u16x8(), b.as_u16x8()))
}

/// Multiply and then horizontally add signed 16 bit integers in `a` and `b`.
///
/// Multiply packed signed 16-bit integers in `a` and `b`, producing
/// intermediate signed 32-bit integers. Horizontally add adjacent pairs of
/// intermediate 32-bit integers.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pmaddwd))]
pub unsafe fn _mm_madd_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pmaddwd(a.as_i16x8(), b.as_i16x8()))
}

/// Compare packed 16-bit integers in `a` and `b`, and return the packed
/// maximum values.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pmaxsw))]
pub unsafe fn _mm_max_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pmaxsw(a.as_i16x8(), b.as_i16x8()))
}

/// Compare packed unsigned 8-bit integers in `a` and `b`, and return the
/// packed maximum values.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pmaxub))]
pub unsafe fn _mm_max_epu8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pmaxub(a.as_u8x16(), b.as_u8x16()))
}

/// Compare packed 16-bit integers in `a` and `b`, and return the packed
/// minimum values.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pminsw))]
pub unsafe fn _mm_min_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pminsw(a.as_i16x8(), b.as_i16x8()))
}

/// Compare packed unsigned 8-bit integers in `a` and `b`, and return the
/// packed minimum values.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pminub))]
pub unsafe fn _mm_min_epu8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pminub(a.as_u8x16(), b.as_u8x16()))
}

/// Multiply the packed 16-bit integers in `a` and `b`.
///
/// The multiplication produces intermediate 32-bit integers, and returns the
/// high 16 bits of the intermediate integers.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pmulhw))]
pub unsafe fn _mm_mulhi_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pmulhw(a.as_i16x8(), b.as_i16x8()))
}

/// Multiply the packed unsigned 16-bit integers in `a` and `b`.
///
/// The multiplication produces intermediate 32-bit integers, and returns the
/// high 16 bits of the intermediate integers.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pmulhuw))]
pub unsafe fn _mm_mulhi_epu16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pmulhuw(a.as_u16x8(), b.as_u16x8()))
}

/// Multiply the packed 16-bit integers in `a` and `b`.
///
/// The multiplication produces intermediate 32-bit integers, and returns the
/// low 16 bits of the intermediate integers.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pmullw))]
pub unsafe fn _mm_mullo_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(simd_mul(a.as_i16x8(), b.as_i16x8()))
}

/// Multiply the low unsigned 32-bit integers from each packed 64-bit element
/// in `a` and `b`.
///
/// Return the unsigned 64-bit results.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pmuludq))]
pub unsafe fn _mm_mul_epu32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pmuludq(a.as_u32x4(), b.as_u32x4()))
}

/// Sum the absolute differences of packed unsigned 8-bit integers.
///
/// Compute the absolute differences of packed unsigned 8-bit integers in `a`
/// and `b`, then horizontally sum each consecutive 8 differences to produce
/// two unsigned 16-bit integers, and pack these unsigned 16-bit integers in
/// the low 16 bits of 64-bit elements returned.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psadbw))]
pub unsafe fn _mm_sad_epu8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(psadbw(a.as_u8x16(), b.as_u8x16()))
}

/// Subtract packed 8-bit integers in `b` from packed 8-bit integers in `a`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psubb))]
pub unsafe fn _mm_sub_epi8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(simd_sub(a.as_i8x16(), b.as_i8x16()))
}

/// Subtract packed 16-bit integers in `b` from packed 16-bit integers in `a`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psubw))]
pub unsafe fn _mm_sub_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(simd_sub(a.as_i16x8(), b.as_i16x8()))
}

/// Subtract packed 32-bit integers in `b` from packed 32-bit integers in `a`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psubd))]
pub unsafe fn _mm_sub_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(simd_sub(a.as_i32x4(), b.as_i32x4()))
}

/// Subtract packed 64-bit integers in `b` from packed 64-bit integers in `a`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psubq))]
pub unsafe fn _mm_sub_epi64(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(simd_sub(a.as_i64x2(), b.as_i64x2()))
}

/// Subtract packed 8-bit integers in `b` from packed 8-bit integers in `a`
/// using saturation.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psubsb))]
pub unsafe fn _mm_subs_epi8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(psubsb(a.as_i8x16(), b.as_i8x16()))
}

/// Subtract packed 16-bit integers in `b` from packed 16-bit integers in `a`
/// using saturation.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psubsw))]
pub unsafe fn _mm_subs_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(psubsw(a.as_i16x8(), b.as_i16x8()))
}

/// Subtract packed unsigned 8-bit integers in `b` from packed unsigned 8-bit
/// integers in `a` using saturation.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psubusb))]
pub unsafe fn _mm_subs_epu8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(psubusb(a.as_u8x16(), b.as_u8x16()))
}

/// Subtract packed unsigned 16-bit integers in `b` from packed unsigned 16-bit
/// integers in `a` using saturation.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psubusw))]
pub unsafe fn _mm_subs_epu16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(psubusw(a.as_u16x8(), b.as_u16x8()))
}

/// Shift `a` left by `imm8` bytes while shifting in zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pslldq, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_slli_si128(a: __m128i, imm8: i32) -> __m128i {
    _mm_slli_si128_impl(a, imm8)
}

/// Implementation detail: converts the immediate argument of the
/// `_mm_slli_si128` intrinsic into a compile-time constant.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn _mm_slli_si128_impl(a: __m128i, imm8: i32) -> __m128i {
    let (zero, imm8) = (_mm_set1_epi8(0).as_i8x16(), imm8 as u32);
    let a = a.as_i8x16();
    macro_rules! shuffle {
        ($shift: expr) => {
            simd_shuffle16::<i8x16, i8x16>(
                zero,
                a,
                [
                    16 - $shift,
                    17 - $shift,
                    18 - $shift,
                    19 - $shift,
                    20 - $shift,
                    21 - $shift,
                    22 - $shift,
                    23 - $shift,
                    24 - $shift,
                    25 - $shift,
                    26 - $shift,
                    27 - $shift,
                    28 - $shift,
                    29 - $shift,
                    30 - $shift,
                    31 - $shift,
                ],
            )
        };
    }
    let x = match imm8 {
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
    mem::transmute(x)
}

/// Shift `a` left by `imm8` bytes while shifting in zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pslldq, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_bslli_si128(a: __m128i, imm8: i32) -> __m128i {
    _mm_slli_si128_impl(a, imm8)
}

/// Shift `a` right by `imm8` bytes while shifting in zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psrldq, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_bsrli_si128(a: __m128i, imm8: i32) -> __m128i {
    _mm_srli_si128_impl(a, imm8)
}

/// Shift packed 16-bit integers in `a` left by `imm8` while shifting in zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psllw, imm8 = 7))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_slli_epi16(a: __m128i, imm8: i32) -> __m128i {
    mem::transmute(pslliw(a.as_i16x8(), imm8))
}

/// Shift packed 16-bit integers in `a` left by `count` while shifting in
/// zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psllw))]
pub unsafe fn _mm_sll_epi16(a: __m128i, count: __m128i) -> __m128i {
    mem::transmute(psllw(a.as_i16x8(), count.as_i16x8()))
}

/// Shift packed 32-bit integers in `a` left by `imm8` while shifting in zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pslld, imm8 = 7))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_slli_epi32(a: __m128i, imm8: i32) -> __m128i {
    mem::transmute(psllid(a.as_i32x4(), imm8))
}

/// Shift packed 32-bit integers in `a` left by `count` while shifting in
/// zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pslld))]
pub unsafe fn _mm_sll_epi32(a: __m128i, count: __m128i) -> __m128i {
    mem::transmute(pslld(a.as_i32x4(), count.as_i32x4()))
}

/// Shift packed 64-bit integers in `a` left by `imm8` while shifting in zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psllq, imm8 = 7))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_slli_epi64(a: __m128i, imm8: i32) -> __m128i {
    mem::transmute(pslliq(a.as_i64x2(), imm8))
}

/// Shift packed 64-bit integers in `a` left by `count` while shifting in
/// zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psllq))]
pub unsafe fn _mm_sll_epi64(a: __m128i, count: __m128i) -> __m128i {
    mem::transmute(psllq(a.as_i64x2(), count.as_i64x2()))
}

/// Shift packed 16-bit integers in `a` right by `imm8` while shifting in sign
/// bits.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psraw, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_srai_epi16(a: __m128i, imm8: i32) -> __m128i {
    mem::transmute(psraiw(a.as_i16x8(), imm8))
}

/// Shift packed 16-bit integers in `a` right by `count` while shifting in sign
/// bits.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psraw))]
pub unsafe fn _mm_sra_epi16(a: __m128i, count: __m128i) -> __m128i {
    mem::transmute(psraw(a.as_i16x8(), count.as_i16x8()))
}

/// Shift packed 32-bit integers in `a` right by `imm8` while shifting in sign
/// bits.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psrad, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_srai_epi32(a: __m128i, imm8: i32) -> __m128i {
    mem::transmute(psraid(a.as_i32x4(), imm8))
}

/// Shift packed 32-bit integers in `a` right by `count` while shifting in sign
/// bits.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psrad))]
pub unsafe fn _mm_sra_epi32(a: __m128i, count: __m128i) -> __m128i {
    mem::transmute(psrad(a.as_i32x4(), count.as_i32x4()))
}

/// Shift `a` right by `imm8` bytes while shifting in zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psrldq, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_srli_si128(a: __m128i, imm8: i32) -> __m128i {
    _mm_srli_si128_impl(a, imm8)
}

/// Implementation detail: converts the immediate argument of the
/// `_mm_srli_si128` intrinsic into a compile-time constant.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn _mm_srli_si128_impl(a: __m128i, imm8: i32) -> __m128i {
    let (zero, imm8) = (_mm_set1_epi8(0).as_i8x16(), imm8 as u32);
    let a = a.as_i8x16();
    macro_rules! shuffle {
        ($shift: expr) => {
            simd_shuffle16(
                a,
                zero,
                [
                    0 + $shift,
                    1 + $shift,
                    2 + $shift,
                    3 + $shift,
                    4 + $shift,
                    5 + $shift,
                    6 + $shift,
                    7 + $shift,
                    8 + $shift,
                    9 + $shift,
                    10 + $shift,
                    11 + $shift,
                    12 + $shift,
                    13 + $shift,
                    14 + $shift,
                    15 + $shift,
                ],
            )
        };
    }
    let x: i8x16 = match imm8 {
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
    mem::transmute(x)
}

/// Shift packed 16-bit integers in `a` right by `imm8` while shifting in
/// zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psrlw, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_srli_epi16(a: __m128i, imm8: i32) -> __m128i {
    mem::transmute(psrliw(a.as_i16x8(), imm8))
}

/// Shift packed 16-bit integers in `a` right by `count` while shifting in
/// zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psrlw))]
pub unsafe fn _mm_srl_epi16(a: __m128i, count: __m128i) -> __m128i {
    mem::transmute(psrlw(a.as_i16x8(), count.as_i16x8()))
}

/// Shift packed 32-bit integers in `a` right by `imm8` while shifting in
/// zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psrld, imm8 = 8))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_srli_epi32(a: __m128i, imm8: i32) -> __m128i {
    mem::transmute(psrlid(a.as_i32x4(), imm8))
}

/// Shift packed 32-bit integers in `a` right by `count` while shifting in
/// zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psrld))]
pub unsafe fn _mm_srl_epi32(a: __m128i, count: __m128i) -> __m128i {
    mem::transmute(psrld(a.as_i32x4(), count.as_i32x4()))
}

/// Shift packed 64-bit integers in `a` right by `imm8` while shifting in
/// zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psrlq, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_srli_epi64(a: __m128i, imm8: i32) -> __m128i {
    mem::transmute(psrliq(a.as_i64x2(), imm8))
}

/// Shift packed 64-bit integers in `a` right by `count` while shifting in
/// zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(psrlq))]
pub unsafe fn _mm_srl_epi64(a: __m128i, count: __m128i) -> __m128i {
    mem::transmute(psrlq(a.as_i64x2(), count.as_i64x2()))
}

/// Compute the bitwise AND of 128 bits (representing integer data) in `a` and
/// `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(andps))]
pub unsafe fn _mm_and_si128(a: __m128i, b: __m128i) -> __m128i {
    simd_and(a, b)
}

/// Compute the bitwise NOT of 128 bits (representing integer data) in `a` and
/// then AND with `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(andnps))]
pub unsafe fn _mm_andnot_si128(a: __m128i, b: __m128i) -> __m128i {
    simd_and(simd_xor(_mm_set1_epi8(-1), a), b)
}

/// Compute the bitwise OR of 128 bits (representing integer data) in `a` and
/// `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(orps))]
pub unsafe fn _mm_or_si128(a: __m128i, b: __m128i) -> __m128i {
    simd_or(a, b)
}

/// Compute the bitwise XOR of 128 bits (representing integer data) in `a` and
/// `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(xorps))]
pub unsafe fn _mm_xor_si128(a: __m128i, b: __m128i) -> __m128i {
    simd_xor(a, b)
}

/// Compare packed 8-bit integers in `a` and `b` for equality.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pcmpeqb))]
pub unsafe fn _mm_cmpeq_epi8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i8x16, _>(simd_eq(a.as_i8x16(), b.as_i8x16()))
}

/// Compare packed 16-bit integers in `a` and `b` for equality.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pcmpeqw))]
pub unsafe fn _mm_cmpeq_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i16x8, _>(simd_eq(a.as_i16x8(), b.as_i16x8()))
}

/// Compare packed 32-bit integers in `a` and `b` for equality.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pcmpeqd))]
pub unsafe fn _mm_cmpeq_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i32x4, _>(simd_eq(a.as_i32x4(), b.as_i32x4()))
}

/// Compare packed 8-bit integers in `a` and `b` for greater-than.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pcmpgtb))]
pub unsafe fn _mm_cmpgt_epi8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i8x16, _>(simd_gt(a.as_i8x16(), b.as_i8x16()))
}

/// Compare packed 16-bit integers in `a` and `b` for greater-than.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pcmpgtw))]
pub unsafe fn _mm_cmpgt_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i16x8, _>(simd_gt(a.as_i16x8(), b.as_i16x8()))
}

/// Compare packed 32-bit integers in `a` and `b` for greater-than.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pcmpgtd))]
pub unsafe fn _mm_cmpgt_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i32x4, _>(simd_gt(a.as_i32x4(), b.as_i32x4()))
}

/// Compare packed 8-bit integers in `a` and `b` for less-than.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pcmpgtb))]
pub unsafe fn _mm_cmplt_epi8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i8x16, _>(simd_lt(a.as_i8x16(), b.as_i8x16()))
}

/// Compare packed 16-bit integers in `a` and `b` for less-than.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pcmpgtw))]
pub unsafe fn _mm_cmplt_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i16x8, _>(simd_lt(a.as_i16x8(), b.as_i16x8()))
}

/// Compare packed 32-bit integers in `a` and `b` for less-than.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pcmpgtd))]
pub unsafe fn _mm_cmplt_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i32x4, _>(simd_lt(a.as_i32x4(), b.as_i32x4()))
}

/// Convert the lower two packed 32-bit integers in `a` to packed
/// double-precision (64-bit) floating-point elements.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvtdq2pd))]
pub unsafe fn _mm_cvtepi32_pd(a: __m128i) -> __m128d {
    let a = a.as_i32x4();
    simd_cast::<i32x2, __m128d>(simd_shuffle2(a, a, [0, 1]))
}

/// Return `a` with its lower element replaced by `b` after converting it to
/// an `f64`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvtsi2sd))]
pub unsafe fn _mm_cvtsi32_sd(a: __m128d, b: i32) -> __m128d {
    simd_insert(a, 0, b as f64)
}

/// Convert packed 32-bit integers in `a` to packed single-precision (32-bit)
/// floating-point elements.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvtdq2ps))]
pub unsafe fn _mm_cvtepi32_ps(a: __m128i) -> __m128 {
    cvtdq2ps(a.as_i32x4())
}

/// Convert packed single-precision (32-bit) floating-point elements in `a`
/// to packed 32-bit integers.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvtps2dq))]
pub unsafe fn _mm_cvtps_epi32(a: __m128) -> __m128i {
    mem::transmute(cvtps2dq(a))
}

/// Return a vector whose lowest element is `a` and all higher elements are
/// `0`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(all(test, target_arch = "x86_64"), assert_instr(movd))]
pub unsafe fn _mm_cvtsi32_si128(a: i32) -> __m128i {
    mem::transmute(i32x4::new(a, 0, 0, 0))
}

/// Return the lowest element of `a`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movd))]
pub unsafe fn _mm_cvtsi128_si32(a: __m128i) -> i32 {
    simd_extract(a.as_i32x4(), 0)
}

/// Set packed 64-bit integers with the supplied values, from highest to
/// lowest.
#[inline]
#[target_feature(enable = "sse2")]
// no particular instruction to test
pub unsafe fn _mm_set_epi64x(e1: i64, e0: i64) -> __m128i {
    mem::transmute(i64x2::new(e0, e1))
}

/// Set packed 32-bit integers with the supplied values.
#[inline]
#[target_feature(enable = "sse2")]
// no particular instruction to test
pub unsafe fn _mm_set_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> __m128i {
    mem::transmute(i32x4::new(e0, e1, e2, e3))
}

/// Set packed 16-bit integers with the supplied values.
#[inline]
#[target_feature(enable = "sse2")]
// no particular instruction to test
pub unsafe fn _mm_set_epi16(
    e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16
) -> __m128i {
    mem::transmute(i16x8::new(e0, e1, e2, e3, e4, e5, e6, e7))
}

/// Set packed 8-bit integers with the supplied values.
#[inline]
#[target_feature(enable = "sse2")]
// no particular instruction to test
pub unsafe fn _mm_set_epi8(
    e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8,
    e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8,
) -> __m128i {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    mem::transmute(i8x16::new(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
    ))
}

/// Broadcast 64-bit integer `a` to all elements.
#[inline]
#[target_feature(enable = "sse2")]
// no particular instruction to test
pub unsafe fn _mm_set1_epi64x(a: i64) -> __m128i {
    _mm_set_epi64x(a, a)
}

/// Broadcast 32-bit integer `a` to all elements.
#[inline]
#[target_feature(enable = "sse2")]
// no particular instruction to test
pub unsafe fn _mm_set1_epi32(a: i32) -> __m128i {
    _mm_set_epi32(a, a, a, a)
}

/// Broadcast 16-bit integer `a` to all elements.
#[inline]
#[target_feature(enable = "sse2")]
// no particular instruction to test
pub unsafe fn _mm_set1_epi16(a: i16) -> __m128i {
    _mm_set_epi16(a, a, a, a, a, a, a, a)
}

/// Broadcast 8-bit integer `a` to all elements.
#[inline]
#[target_feature(enable = "sse2")]
// no particular instruction to test
pub unsafe fn _mm_set1_epi8(a: i8) -> __m128i {
    _mm_set_epi8(a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a)
}

/// Set packed 32-bit integers with the supplied values in reverse order.
#[inline]
#[target_feature(enable = "sse2")]
// no particular instruction to test
pub unsafe fn _mm_setr_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> __m128i {
    _mm_set_epi32(e0, e1, e2, e3)
}

/// Set packed 16-bit integers with the supplied values in reverse order.
#[inline]
#[target_feature(enable = "sse2")]
// no particular instruction to test
pub unsafe fn _mm_setr_epi16(
    e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16
) -> __m128i {
    _mm_set_epi16(e0, e1, e2, e3, e4, e5, e6, e7)
}

/// Set packed 8-bit integers with the supplied values in reverse order.
#[inline]
#[target_feature(enable = "sse2")]
// no particular instruction to test
pub unsafe fn _mm_setr_epi8(
    e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8,
    e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8,
) -> __m128i {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    _mm_set_epi8(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
    )
}

/// Returns a vector with all elements set to zero.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(xorps))]
pub unsafe fn _mm_setzero_si128() -> __m128i {
    _mm_set1_epi64x(0)
}

/// Load 64-bit integer from memory into first element of returned vector.
#[inline]
#[target_feature(enable = "sse2")]
// FIXME movsd on windows
#[cfg_attr(all(test, not(windows),
               not(all(target_os = "linux", target_arch = "x86_64")),
               target_arch = "x86_64"),
           assert_instr(movq))]
pub unsafe fn _mm_loadl_epi64(mem_addr: *const __m128i) -> __m128i {
    _mm_set_epi64x(0, simd_extract((*mem_addr).as_i64x2(), 0))
}

/// Load 128-bits of integer data from memory into a new vector.
///
/// `mem_addr` must be aligned on a 16-byte boundary.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movaps))]
pub unsafe fn _mm_load_si128(mem_addr: *const __m128i) -> __m128i {
    *mem_addr
}

/// Load 128-bits of integer data from memory into a new vector.
///
/// `mem_addr` does not need to be aligned on any particular boundary.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movups))]
pub unsafe fn _mm_loadu_si128(mem_addr: *const __m128i) -> __m128i {
    let mut dst: __m128i = _mm_undefined_si128();
    ptr::copy_nonoverlapping(
        mem_addr as *const u8,
        &mut dst as *mut __m128i as *mut u8,
        mem::size_of::<__m128i>(),
    );
    dst
}

/// Conditionally store 8-bit integer elements from `a` into memory using
/// `mask`.
///
/// Elements are not stored when the highest bit is not set in the
/// corresponding element.
///
/// `mem_addr` should correspond to a 128-bit memory location and does not need
/// to be aligned on any particular boundary.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(maskmovdqu))]
pub unsafe fn _mm_maskmoveu_si128(
    a: __m128i, mask: __m128i, mem_addr: *mut i8
) {
    maskmovdqu(a.as_i8x16(), mask.as_i8x16(), mem_addr)
}

/// Store 128-bits of integer data from `a` into memory.
///
/// `mem_addr` must be aligned on a 16-byte boundary.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movaps))]
pub unsafe fn _mm_store_si128(mem_addr: *mut __m128i, a: __m128i) {
    *mem_addr = a;
}

/// Store 128-bits of integer data from `a` into memory.
///
/// `mem_addr` does not need to be aligned on any particular boundary.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movups))] // FIXME movdqu expected
pub unsafe fn _mm_storeu_si128(mem_addr: *mut __m128i, a: __m128i) {
    storeudq(mem_addr as *mut i8, a);
}

/// Store the lower 64-bit integer `a` to a memory location.
///
/// `mem_addr` does not need to be aligned on any particular boundary.
#[inline]
#[target_feature(enable = "sse2")]
// FIXME mov on windows, movlps on i686
#[cfg_attr(all(test, not(windows),
               not(all(target_os = "linux", target_arch = "x86_64")),
               target_arch = "x86_64"),
           assert_instr(movq))]
pub unsafe fn _mm_storel_epi64(mem_addr: *mut __m128i, a: __m128i) {
    ptr::copy_nonoverlapping(
        &a as *const _ as *const u8,
        mem_addr as *mut u8,
        8,
    );
}

/// Stores a 128-bit integer vector to a 128-bit aligned memory location.
/// To minimize caching, the data is flagged as non-temporal (unlikely to be
/// used again soon).
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movntps))] // FIXME movntdq
pub unsafe fn _mm_stream_si128(mem_addr: *mut __m128i, a: __m128i) {
    ::intrinsics::nontemporal_store(mem_addr, a);
}

/// Stores a 32-bit integer value in the specified memory location.
/// To minimize caching, the data is flagged as non-temporal (unlikely to be
/// used again soon).
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movnti))]
pub unsafe fn _mm_stream_si32(mem_addr: *mut i32, a: i32) {
    ::intrinsics::nontemporal_store(mem_addr, a);
}

/// Return a vector where the low element is extracted from `a` and its upper
/// element is zero.
#[inline]
#[target_feature(enable = "sse2")]
// FIXME movd on windows, movd on i686
#[cfg_attr(all(test, not(windows), target_arch = "x86_64"),
           assert_instr(movq))]
pub unsafe fn _mm_move_epi64(a: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let r: i64x2 = simd_shuffle2(a.as_i64x2(), zero.as_i64x2(), [0, 2]);
    mem::transmute(r)
}

/// Convert packed 16-bit integers from `a` and `b` to packed 8-bit integers
/// using signed saturation.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(packsswb))]
pub unsafe fn _mm_packs_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(packsswb(a.as_i16x8(), b.as_i16x8()))
}

/// Convert packed 32-bit integers from `a` and `b` to packed 16-bit integers
/// using signed saturation.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(packssdw))]
pub unsafe fn _mm_packs_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(packssdw(a.as_i32x4(), b.as_i32x4()))
}

/// Convert packed 16-bit integers from `a` and `b` to packed 8-bit integers
/// using unsigned saturation.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(packuswb))]
pub unsafe fn _mm_packus_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(packuswb(a.as_i16x8(), b.as_i16x8()))
}

/// Return the `imm8` element of `a`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pextrw, imm8 = 9))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_extract_epi16(a: __m128i, imm8: i32) -> i32 {
    simd_extract::<_, i16>(a.as_i16x8(), (imm8 & 7) as u32) as i32
}

/// Return a new vector where the `imm8` element of `a` is replaced with `i`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pinsrw, imm8 = 9))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_insert_epi16(a: __m128i, i: i32, imm8: i32) -> __m128i {
    mem::transmute(simd_insert(
        a.as_i16x8(),
        (imm8 & 7) as u32,
        i as i16,
    ))
}

/// Return a mask of the most significant bit of each element in `a`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pmovmskb))]
pub unsafe fn _mm_movemask_epi8(a: __m128i) -> i32 {
    pmovmskb(a.as_i8x16())
}

/// Shuffle 32-bit integers in `a` using the control in `imm8`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pshufd, imm8 = 9))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_shuffle_epi32(a: __m128i, imm8: i32) -> __m128i {
    // simd_shuffleX requires that its selector parameter be made up of
    // constant values, but we can't enforce that here. In spirit, we need
    // to write a `match` on all possible values of a byte, and for each value,
    // hard-code the correct `simd_shuffleX` call using only constants. We
    // then hope for LLVM to do the rest.
    //
    // Of course, that's... awful. So we try to use macros to do it for us.
    let imm8 = (imm8 & 0xFF) as u8;
    let a = a.as_i32x4();

    macro_rules! shuffle_done {
        ($x01: expr, $x23: expr, $x45: expr, $x67: expr) => {
            simd_shuffle4(a, a, [$x01, $x23, $x45, $x67])
        };
    }
    macro_rules! shuffle_x67 {
        ($x01: expr, $x23: expr, $x45: expr) => {
            match (imm8 >> 6) & 0b11 {
                0b00 => shuffle_done!($x01, $x23, $x45, 0),
                0b01 => shuffle_done!($x01, $x23, $x45, 1),
                0b10 => shuffle_done!($x01, $x23, $x45, 2),
                _ => shuffle_done!($x01, $x23, $x45, 3),
            }
        };
    }
    macro_rules! shuffle_x45 {
        ($x01: expr, $x23: expr) => {
            match (imm8 >> 4) & 0b11 {
                0b00 => shuffle_x67!($x01, $x23, 0),
                0b01 => shuffle_x67!($x01, $x23, 1),
                0b10 => shuffle_x67!($x01, $x23, 2),
                _ => shuffle_x67!($x01, $x23, 3),
            }
        };
    }
    macro_rules! shuffle_x23 {
        ($x01: expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => shuffle_x45!($x01, 0),
                0b01 => shuffle_x45!($x01, 1),
                0b10 => shuffle_x45!($x01, 2),
                _ => shuffle_x45!($x01, 3),
            }
        };
    }
    let x: i32x4 = match imm8 & 0b11 {
        0b00 => shuffle_x23!(0),
        0b01 => shuffle_x23!(1),
        0b10 => shuffle_x23!(2),
        _ => shuffle_x23!(3),
    };
    mem::transmute(x)
}

/// Shuffle 16-bit integers in the high 64 bits of `a` using the control in
/// `imm8`.
///
/// Put the results in the high 64 bits of the returned vector, with the low 64
/// bits being copied from from `a`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pshufhw, imm8 = 9))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_shufflehi_epi16(a: __m128i, imm8: i32) -> __m128i {
    // See _mm_shuffle_epi32.
    let imm8 = (imm8 & 0xFF) as u8;
    let a = a.as_i16x8();
    macro_rules! shuffle_done {
        ($x01: expr, $x23: expr, $x45: expr, $x67: expr) => {
            simd_shuffle8(
                a,
                a,
                [
                    0,
                    1,
                    2,
                    3,
                    $x01 + 4,
                    $x23 + 4,
                    $x45 + 4,
                    $x67 + 4,
                ],
            )
        };
    }
    macro_rules! shuffle_x67 {
        ($x01: expr, $x23: expr, $x45: expr) => {
            match (imm8 >> 6) & 0b11 {
                0b00 => shuffle_done!($x01, $x23, $x45, 0),
                0b01 => shuffle_done!($x01, $x23, $x45, 1),
                0b10 => shuffle_done!($x01, $x23, $x45, 2),
                _ => shuffle_done!($x01, $x23, $x45, 3),
            }
        };
    }
    macro_rules! shuffle_x45 {
        ($x01: expr, $x23: expr) => {
            match (imm8 >> 4) & 0b11 {
                0b00 => shuffle_x67!($x01, $x23, 0),
                0b01 => shuffle_x67!($x01, $x23, 1),
                0b10 => shuffle_x67!($x01, $x23, 2),
                _ => shuffle_x67!($x01, $x23, 3),
            }
        };
    }
    macro_rules! shuffle_x23 {
        ($x01: expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => shuffle_x45!($x01, 0),
                0b01 => shuffle_x45!($x01, 1),
                0b10 => shuffle_x45!($x01, 2),
                _ => shuffle_x45!($x01, 3),
            }
        };
    }
    let x: i16x8 = match imm8 & 0b11 {
        0b00 => shuffle_x23!(0),
        0b01 => shuffle_x23!(1),
        0b10 => shuffle_x23!(2),
        _ => shuffle_x23!(3),
    };
    mem::transmute(x)
}

/// Shuffle 16-bit integers in the low 64 bits of `a` using the control in
/// `imm8`.
///
/// Put the results in the low 64 bits of the returned vector, with the high 64
/// bits being copied from from `a`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(pshuflw, imm8 = 9))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_shufflelo_epi16(a: __m128i, imm8: i32) -> __m128i {
    // See _mm_shuffle_epi32.
    let imm8 = (imm8 & 0xFF) as u8;
    let a = a.as_i16x8();

    macro_rules! shuffle_done {
        ($x01: expr, $x23: expr, $x45: expr, $x67: expr) => {
            simd_shuffle8(a, a, [$x01, $x23, $x45, $x67, 4, 5, 6, 7])
        };
    }
    macro_rules! shuffle_x67 {
        ($x01: expr, $x23: expr, $x45: expr) => {
            match (imm8 >> 6) & 0b11 {
                0b00 => shuffle_done!($x01, $x23, $x45, 0),
                0b01 => shuffle_done!($x01, $x23, $x45, 1),
                0b10 => shuffle_done!($x01, $x23, $x45, 2),
                _ => shuffle_done!($x01, $x23, $x45, 3),
            }
        };
    }
    macro_rules! shuffle_x45 {
        ($x01: expr, $x23: expr) => {
            match (imm8 >> 4) & 0b11 {
                0b00 => shuffle_x67!($x01, $x23, 0),
                0b01 => shuffle_x67!($x01, $x23, 1),
                0b10 => shuffle_x67!($x01, $x23, 2),
                _ => shuffle_x67!($x01, $x23, 3),
            }
        };
    }
    macro_rules! shuffle_x23 {
        ($x01: expr) => {
            match (imm8 >> 2) & 0b11 {
                0b00 => shuffle_x45!($x01, 0),
                0b01 => shuffle_x45!($x01, 1),
                0b10 => shuffle_x45!($x01, 2),
                _ => shuffle_x45!($x01, 3),
            }
        };
    }
    let x: i16x8 = match imm8 & 0b11 {
        0b00 => shuffle_x23!(0),
        0b01 => shuffle_x23!(1),
        0b10 => shuffle_x23!(2),
        _ => shuffle_x23!(3),
    };
    mem::transmute(x)
}

/// Unpack and interleave 8-bit integers from the high half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(punpckhbw))]
pub unsafe fn _mm_unpackhi_epi8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i8x16, _>(simd_shuffle16(
        a.as_i8x16(),
        b.as_i8x16(),
        [
            8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
        ],
    ))
}

/// Unpack and interleave 16-bit integers from the high half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(punpckhwd))]
pub unsafe fn _mm_unpackhi_epi16(a: __m128i, b: __m128i) -> __m128i {
    let x = simd_shuffle8(
        a.as_i16x8(),
        b.as_i16x8(),
        [4, 12, 5, 13, 6, 14, 7, 15],
    );
    mem::transmute::<i16x8, _>(x)
}

/// Unpack and interleave 32-bit integers from the high half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(unpckhps))]
pub unsafe fn _mm_unpackhi_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i32x4, _>(simd_shuffle4(
        a.as_i32x4(),
        b.as_i32x4(),
        [2, 6, 3, 7],
    ))
}

/// Unpack and interleave 64-bit integers from the high half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(unpckhpd))]
pub unsafe fn _mm_unpackhi_epi64(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i64x2, _>(simd_shuffle2(
        a.as_i64x2(),
        b.as_i64x2(),
        [1, 3],
    ))
}

/// Unpack and interleave 8-bit integers from the low half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(punpcklbw))]
pub unsafe fn _mm_unpacklo_epi8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i8x16, _>(simd_shuffle16(
        a.as_i8x16(),
        b.as_i8x16(),
        [
            0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23
        ],
    ))
}

/// Unpack and interleave 16-bit integers from the low half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(punpcklwd))]
pub unsafe fn _mm_unpacklo_epi16(a: __m128i, b: __m128i) -> __m128i {
    let x = simd_shuffle8(
        a.as_i16x8(),
        b.as_i16x8(),
        [0, 8, 1, 9, 2, 10, 3, 11],
    );
    mem::transmute::<i16x8, _>(x)
}

/// Unpack and interleave 32-bit integers from the low half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(unpcklps))]
pub unsafe fn _mm_unpacklo_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i32x4, _>(simd_shuffle4(
        a.as_i32x4(),
        b.as_i32x4(),
        [0, 4, 1, 5],
    ))
}

/// Unpack and interleave 64-bit integers from the low half of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movlhps))]
pub unsafe fn _mm_unpacklo_epi64(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute::<i64x2, _>(simd_shuffle2(
        a.as_i64x2(),
        b.as_i64x2(),
        [0, 2],
    ))
}

/// Return a new vector with the low element of `a` replaced by the sum of the
/// low elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(addsd))]
pub unsafe fn _mm_add_sd(a: __m128d, b: __m128d) -> __m128d {
    simd_insert(a, 0, _mm_cvtsd_f64(a) + _mm_cvtsd_f64(b))
}

/// Add packed double-precision (64-bit) floating-point elements in `a` and
/// `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(addpd))]
pub unsafe fn _mm_add_pd(a: __m128d, b: __m128d) -> __m128d {
    simd_add(a, b)
}

/// Return a new vector with the low element of `a` replaced by the result of
/// diving the lower element of `a` by the lower element of `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(divsd))]
pub unsafe fn _mm_div_sd(a: __m128d, b: __m128d) -> __m128d {
    simd_insert(a, 0, _mm_cvtsd_f64(a) / _mm_cvtsd_f64(b))
}

/// Divide packed double-precision (64-bit) floating-point elements in `a` by
/// packed elements in `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(divpd))]
pub unsafe fn _mm_div_pd(a: __m128d, b: __m128d) -> __m128d {
    simd_div(a, b)
}

/// Return a new vector with the low element of `a` replaced by the maximum
/// of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(maxsd))]
pub unsafe fn _mm_max_sd(a: __m128d, b: __m128d) -> __m128d {
    maxsd(a, b)
}

/// Return a new vector with the maximum values from corresponding elements in
/// `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(maxpd))]
pub unsafe fn _mm_max_pd(a: __m128d, b: __m128d) -> __m128d {
    maxpd(a, b)
}

/// Return a new vector with the low element of `a` replaced by the minimum
/// of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(minsd))]
pub unsafe fn _mm_min_sd(a: __m128d, b: __m128d) -> __m128d {
    minsd(a, b)
}

/// Return a new vector with the minimum values from corresponding elements in
/// `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(minpd))]
pub unsafe fn _mm_min_pd(a: __m128d, b: __m128d) -> __m128d {
    minpd(a, b)
}

/// Return a new vector with the low element of `a` replaced by multiplying the
/// low elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(mulsd))]
pub unsafe fn _mm_mul_sd(a: __m128d, b: __m128d) -> __m128d {
    simd_insert(a, 0, _mm_cvtsd_f64(a) * _mm_cvtsd_f64(b))
}

/// Multiply packed double-precision (64-bit) floating-point elements in `a`
/// and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(mulpd))]
pub unsafe fn _mm_mul_pd(a: __m128d, b: __m128d) -> __m128d {
    simd_mul(a, b)
}

/// Return a new vector with the low element of `a` replaced by the square
/// root of the lower element `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(sqrtsd))]
pub unsafe fn _mm_sqrt_sd(a: __m128d, b: __m128d) -> __m128d {
    simd_insert(a, 0, _mm_cvtsd_f64(sqrtsd(b)))
}

/// Return a new vector with the square root of each of the values in `a`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(sqrtpd))]
pub unsafe fn _mm_sqrt_pd(a: __m128d) -> __m128d {
    sqrtpd(a)
}

/// Return a new vector with the low element of `a` replaced by subtracting the
/// low element by `b` from the low element of `a`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(subsd))]
pub unsafe fn _mm_sub_sd(a: __m128d, b: __m128d) -> __m128d {
    simd_insert(a, 0, _mm_cvtsd_f64(a) - _mm_cvtsd_f64(b))
}

/// Subtract packed double-precision (64-bit) floating-point elements in `b`
/// from `a`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(subpd))]
pub unsafe fn _mm_sub_pd(a: __m128d, b: __m128d) -> __m128d {
    simd_sub(a, b)
}

/// Compute the bitwise AND of packed double-precision (64-bit) floating-point
/// elements in `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(andps))]
pub unsafe fn _mm_and_pd(a: __m128d, b: __m128d) -> __m128d {
    let a: __m128i = mem::transmute(a);
    let b: __m128i = mem::transmute(b);
    mem::transmute(_mm_and_si128(a, b))
}

/// Compute the bitwise NOT of `a` and then AND with `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(andnps))]
pub unsafe fn _mm_andnot_pd(a: __m128d, b: __m128d) -> __m128d {
    let a: __m128i = mem::transmute(a);
    let b: __m128i = mem::transmute(b);
    mem::transmute(_mm_andnot_si128(a, b))
}

/// Compute the bitwise OR of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(orps))]
pub unsafe fn _mm_or_pd(a: __m128d, b: __m128d) -> __m128d {
    let a: __m128i = mem::transmute(a);
    let b: __m128i = mem::transmute(b);
    mem::transmute(_mm_or_si128(a, b))
}

/// Compute the bitwise OR of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(xorps))]
pub unsafe fn _mm_xor_pd(a: __m128d, b: __m128d) -> __m128d {
    let a: __m128i = mem::transmute(a);
    let b: __m128i = mem::transmute(b);
    mem::transmute(_mm_xor_si128(a, b))
}

/// Return a new vector with the low element of `a` replaced by the equality
/// comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpeqsd))]
pub unsafe fn _mm_cmpeq_sd(a: __m128d, b: __m128d) -> __m128d {
    cmpsd(a, b, 0)
}

/// Return a new vector with the low element of `a` replaced by the less-than
/// comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpltsd))]
pub unsafe fn _mm_cmplt_sd(a: __m128d, b: __m128d) -> __m128d {
    cmpsd(a, b, 1)
}

/// Return a new vector with the low element of `a` replaced by the
/// less-than-or-equal comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmplesd))]
pub unsafe fn _mm_cmple_sd(a: __m128d, b: __m128d) -> __m128d {
    cmpsd(a, b, 2)
}

/// Return a new vector with the low element of `a` replaced by the
/// greater-than comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpltsd))]
pub unsafe fn _mm_cmpgt_sd(a: __m128d, b: __m128d) -> __m128d {
    simd_insert(
        _mm_cmplt_sd(b, a),
        1,
        simd_extract::<_, f64>(a, 1),
    )
}

/// Return a new vector with the low element of `a` replaced by the
/// greater-than-or-equal comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmplesd))]
pub unsafe fn _mm_cmpge_sd(a: __m128d, b: __m128d) -> __m128d {
    simd_insert(
        _mm_cmple_sd(b, a),
        1,
        simd_extract::<_, f64>(a, 1),
    )
}

/// Return a new vector with the low element of `a` replaced by the result
/// of comparing both of the lower elements of `a` and `b` to `NaN`. If
/// neither are equal to `NaN` then `0xFFFFFFFFFFFFFFFF` is used and `0`
/// otherwise.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpordsd))]
pub unsafe fn _mm_cmpord_sd(a: __m128d, b: __m128d) -> __m128d {
    cmpsd(a, b, 7)
}

/// Return a new vector with the low element of `a` replaced by the result of
/// comparing both of the lower elements of `a` and `b` to `NaN`. If either is
/// equal to `NaN` then `0xFFFFFFFFFFFFFFFF` is used and `0` otherwise.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpunordsd))]
pub unsafe fn _mm_cmpunord_sd(a: __m128d, b: __m128d) -> __m128d {
    cmpsd(a, b, 3)
}

/// Return a new vector with the low element of `a` replaced by the not-equal
/// comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpneqsd))]
pub unsafe fn _mm_cmpneq_sd(a: __m128d, b: __m128d) -> __m128d {
    cmpsd(a, b, 4)
}

/// Return a new vector with the low element of `a` replaced by the
/// not-less-than comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpnltsd))]
pub unsafe fn _mm_cmpnlt_sd(a: __m128d, b: __m128d) -> __m128d {
    cmpsd(a, b, 5)
}

/// Return a new vector with the low element of `a` replaced by the
/// not-less-than-or-equal comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpnlesd))]
pub unsafe fn _mm_cmpnle_sd(a: __m128d, b: __m128d) -> __m128d {
    cmpsd(a, b, 6)
}

/// Return a new vector with the low element of `a` replaced by the
/// not-greater-than comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpnltsd))]
pub unsafe fn _mm_cmpngt_sd(a: __m128d, b: __m128d) -> __m128d {
    simd_insert(
        _mm_cmpnlt_sd(b, a),
        1,
        simd_extract::<_, f64>(a, 1),
    )
}

/// Return a new vector with the low element of `a` replaced by the
/// not-greater-than-or-equal comparison of the lower elements of `a` and `b`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpnlesd))]
pub unsafe fn _mm_cmpnge_sd(a: __m128d, b: __m128d) -> __m128d {
    simd_insert(
        _mm_cmpnle_sd(b, a),
        1,
        simd_extract::<_, f64>(a, 1),
    )
}

/// Compare corresponding elements in `a` and `b` for equality.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpeqpd))]
pub unsafe fn _mm_cmpeq_pd(a: __m128d, b: __m128d) -> __m128d {
    cmppd(a, b, 0)
}

/// Compare corresponding elements in `a` and `b` for less-than.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpltpd))]
pub unsafe fn _mm_cmplt_pd(a: __m128d, b: __m128d) -> __m128d {
    cmppd(a, b, 1)
}

/// Compare corresponding elements in `a` and `b` for less-than-or-equal
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmplepd))]
pub unsafe fn _mm_cmple_pd(a: __m128d, b: __m128d) -> __m128d {
    cmppd(a, b, 2)
}

/// Compare corresponding elements in `a` and `b` for greater-than.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpltpd))]
pub unsafe fn _mm_cmpgt_pd(a: __m128d, b: __m128d) -> __m128d {
    _mm_cmplt_pd(b, a)
}

/// Compare corresponding elements in `a` and `b` for greater-than-or-equal.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmplepd))]
pub unsafe fn _mm_cmpge_pd(a: __m128d, b: __m128d) -> __m128d {
    _mm_cmple_pd(b, a)
}

/// Compare corresponding elements in `a` and `b` to see if neither is `NaN`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpordpd))]
pub unsafe fn _mm_cmpord_pd(a: __m128d, b: __m128d) -> __m128d {
    cmppd(a, b, 7)
}

/// Compare corresponding elements in `a` and `b` to see if either is `NaN`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpunordpd))]
pub unsafe fn _mm_cmpunord_pd(a: __m128d, b: __m128d) -> __m128d {
    cmppd(a, b, 3)
}

/// Compare corresponding elements in `a` and `b` for not-equal.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpneqpd))]
pub unsafe fn _mm_cmpneq_pd(a: __m128d, b: __m128d) -> __m128d {
    cmppd(a, b, 4)
}

/// Compare corresponding elements in `a` and `b` for not-less-than.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpnltpd))]
pub unsafe fn _mm_cmpnlt_pd(a: __m128d, b: __m128d) -> __m128d {
    cmppd(a, b, 5)
}

/// Compare corresponding elements in `a` and `b` for not-less-than-or-equal.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpnlepd))]
pub unsafe fn _mm_cmpnle_pd(a: __m128d, b: __m128d) -> __m128d {
    cmppd(a, b, 6)
}

/// Compare corresponding elements in `a` and `b` for not-greater-than.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpnltpd))]
pub unsafe fn _mm_cmpngt_pd(a: __m128d, b: __m128d) -> __m128d {
    _mm_cmpnlt_pd(b, a)
}

/// Compare corresponding elements in `a` and `b` for
/// not-greater-than-or-equal.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cmpnlepd))]
pub unsafe fn _mm_cmpnge_pd(a: __m128d, b: __m128d) -> __m128d {
    _mm_cmpnle_pd(b, a)
}

/// Compare the lower element of `a` and `b` for equality.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(comisd))]
pub unsafe fn _mm_comieq_sd(a: __m128d, b: __m128d) -> i32 {
    comieqsd(a, b)
}

/// Compare the lower element of `a` and `b` for less-than.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(comisd))]
pub unsafe fn _mm_comilt_sd(a: __m128d, b: __m128d) -> i32 {
    comiltsd(a, b)
}

/// Compare the lower element of `a` and `b` for less-than-or-equal.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(comisd))]
pub unsafe fn _mm_comile_sd(a: __m128d, b: __m128d) -> i32 {
    comilesd(a, b)
}

/// Compare the lower element of `a` and `b` for greater-than.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(comisd))]
pub unsafe fn _mm_comigt_sd(a: __m128d, b: __m128d) -> i32 {
    comigtsd(a, b)
}

/// Compare the lower element of `a` and `b` for greater-than-or-equal.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(comisd))]
pub unsafe fn _mm_comige_sd(a: __m128d, b: __m128d) -> i32 {
    comigesd(a, b)
}

/// Compare the lower element of `a` and `b` for not-equal.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(comisd))]
pub unsafe fn _mm_comineq_sd(a: __m128d, b: __m128d) -> i32 {
    comineqsd(a, b)
}

/// Compare the lower element of `a` and `b` for equality.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(ucomisd))]
pub unsafe fn _mm_ucomieq_sd(a: __m128d, b: __m128d) -> i32 {
    ucomieqsd(a, b)
}

/// Compare the lower element of `a` and `b` for less-than.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(ucomisd))]
pub unsafe fn _mm_ucomilt_sd(a: __m128d, b: __m128d) -> i32 {
    ucomiltsd(a, b)
}

/// Compare the lower element of `a` and `b` for less-than-or-equal.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(ucomisd))]
pub unsafe fn _mm_ucomile_sd(a: __m128d, b: __m128d) -> i32 {
    ucomilesd(a, b)
}

/// Compare the lower element of `a` and `b` for greater-than.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(ucomisd))]
pub unsafe fn _mm_ucomigt_sd(a: __m128d, b: __m128d) -> i32 {
    ucomigtsd(a, b)
}

/// Compare the lower element of `a` and `b` for greater-than-or-equal.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(ucomisd))]
pub unsafe fn _mm_ucomige_sd(a: __m128d, b: __m128d) -> i32 {
    ucomigesd(a, b)
}

/// Compare the lower element of `a` and `b` for not-equal.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(ucomisd))]
pub unsafe fn _mm_ucomineq_sd(a: __m128d, b: __m128d) -> i32 {
    ucomineqsd(a, b)
}

/// Convert packed double-precision (64-bit) floating-point elements in "a" to
/// packed single-precision (32-bit) floating-point elements
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvtpd2ps))]
pub unsafe fn _mm_cvtpd_ps(a: __m128d) -> __m128 {
    cvtpd2ps(a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to
/// packed
/// double-precision (64-bit) floating-point elements.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvtps2pd))]
pub unsafe fn _mm_cvtps_pd(a: __m128) -> __m128d {
    cvtps2pd(a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to
/// packed 32-bit integers.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvtpd2dq))]
pub unsafe fn _mm_cvtpd_epi32(a: __m128d) -> __m128i {
    mem::transmute(cvtpd2dq(a))
}

/// Convert the lower double-precision (64-bit) floating-point element in a to
/// a 32-bit integer.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvtsd2si))]
pub unsafe fn _mm_cvtsd_si32(a: __m128d) -> i32 {
    cvtsd2si(a)
}

/// Convert the lower double-precision (64-bit) floating-point element in `b`
/// to a single-precision (32-bit) floating-point element, store the result in
/// the lower element of the return value, and copy the upper element from `a`
/// to the upper element the return value.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvtsd2ss))]
pub unsafe fn _mm_cvtsd_ss(a: __m128, b: __m128d) -> __m128 {
    cvtsd2ss(a, b)
}

/// Return the lower double-precision (64-bit) floating-point element of "a".
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_cvtsd_f64(a: __m128d) -> f64 {
    simd_extract(a, 0)
}

/// Convert the lower single-precision (32-bit) floating-point element in `b`
/// to a double-precision (64-bit) floating-point element, store the result in
/// the lower element of the return value, and copy the upper element from `a`
/// to the upper element the return value.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvtss2sd))]
pub unsafe fn _mm_cvtss_sd(a: __m128d, b: __m128) -> __m128d {
    cvtss2sd(a, b)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to
/// packed 32-bit integers with truncation.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvttpd2dq))]
pub unsafe fn _mm_cvttpd_epi32(a: __m128d) -> __m128i {
    mem::transmute(cvttpd2dq(a))
}

/// Convert the lower double-precision (64-bit) floating-point element in `a`
/// to a 32-bit integer with truncation.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvttsd2si))]
pub unsafe fn _mm_cvttsd_si32(a: __m128d) -> i32 {
    cvttsd2si(a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to
/// packed 32-bit integers with truncation.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(cvttps2dq))]
pub unsafe fn _mm_cvttps_epi32(a: __m128) -> __m128i {
    mem::transmute(cvttps2dq(a))
}

/// Copy double-precision (64-bit) floating-point element `a` to the lower
/// element of the packed 64-bit return value.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set_sd(a: f64) -> __m128d {
    _mm_set_pd(0.0, a)
}

/// Broadcast double-precision (64-bit) floating-point value a to all elements
/// of the return value.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set1_pd(a: f64) -> __m128d {
    _mm_set_pd(a, a)
}

/// Broadcast double-precision (64-bit) floating-point value a to all elements
/// of the return value.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set_pd1(a: f64) -> __m128d {
    _mm_set_pd(a, a)
}

/// Set packed double-precision (64-bit) floating-point elements in the return
/// value with the supplied values.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_set_pd(a: f64, b: f64) -> __m128d {
    __m128d(b, a)
}

/// Set packed double-precision (64-bit) floating-point elements in the return
/// value with the supplied values in reverse order.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_setr_pd(a: f64, b: f64) -> __m128d {
    _mm_set_pd(b, a)
}

/// Returns packed double-precision (64-bit) floating-point elements with all
/// zeros.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(xorps))] // FIXME xorpd expected
pub unsafe fn _mm_setzero_pd() -> __m128d {
    _mm_set_pd(0.0, 0.0)
}

/// Return a mask of the most significant bit of each element in `a`.
///
/// The mask is stored in the 2 least significant bits of the return value.
/// All other bits are set to `0`.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movmskpd))]
pub unsafe fn _mm_movemask_pd(a: __m128d) -> i32 {
    movmskpd(a)
}

/// Load 128-bits (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from memory into the returned vector.
/// `mem_addr` must be aligned on a 16-byte boundary or a general-protection
/// exception may be generated.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movaps))]
pub unsafe fn _mm_load_pd(mem_addr: *const f64) -> __m128d {
    *(mem_addr as *const __m128d)
}

/// Loads a 64-bit double-precision value to the low element of a
/// 128-bit integer vector and clears the upper element.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movsd))]
pub unsafe fn _mm_load_sd(mem_addr: *const f64) -> __m128d {
    _mm_setr_pd(*mem_addr, 0.)
}

/// Loads a double-precision value into the high-order bits of a 128-bit
/// vector of [2 x double]. The low-order bits are copied from the low-order
/// bits of the first operand.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movhpd))]
pub unsafe fn _mm_loadh_pd(a: __m128d, mem_addr: *const f64) -> __m128d {
    _mm_setr_pd(simd_extract(a, 0), *mem_addr)
}

/// Loads a double-precision value into the low-order bits of a 128-bit
/// vector of [2 x double]. The high-order bits are copied from the
/// high-order bits of the first operand.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movlpd))]
pub unsafe fn _mm_loadl_pd(a: __m128d, mem_addr: *const f64) -> __m128d {
    _mm_setr_pd(*mem_addr, simd_extract(a, 1))
}

/// Stores a 128-bit floating point vector of [2 x double] to a 128-bit
/// aligned memory location.
/// To minimize caching, the data is flagged as non-temporal (unlikely to be
/// used again soon).
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movntps))] // FIXME movntpd
pub unsafe fn _mm_stream_pd(mem_addr: *mut f64, a: __m128d) {
    intrinsics::nontemporal_store(mem::transmute(mem_addr), a);
}

/// Stores the lower 64 bits of a 128-bit vector of [2 x double] to a
/// memory location.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movlps))]
pub unsafe fn _mm_store_sd(mem_addr: *mut f64, a: __m128d) {
    *mem_addr = simd_extract(a, 0)
}

/// Store 128-bits (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory. `mem_addr` must be aligned
/// on a 16-byte boundary or a general-protection exception may be generated.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movaps))]
pub unsafe fn _mm_store_pd(mem_addr: *mut f64, a: __m128d) {
    *(mem_addr as *mut __m128d) = a;
}

/// Store 128-bits (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movups))] // FIXME movupd expected
pub unsafe fn _mm_storeu_pd(mem_addr: *mut f64, a: __m128d) {
    storeupd(mem_addr as *mut i8, a);
}

/// Store the lower double-precision (64-bit) floating-point element from `a`
/// into 2 contiguous elements in memory. `mem_addr` must be aligned on a
/// 16-byte boundary or a general-protection exception may be generated.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_store1_pd(mem_addr: *mut f64, a: __m128d) {
    let b: __m128d = simd_shuffle2(a, a, [0, 0]);
    *(mem_addr as *mut __m128d) = b;
}

/// Store the lower double-precision (64-bit) floating-point element from `a`
/// into 2 contiguous elements in memory. `mem_addr` must be aligned on a
/// 16-byte boundary or a general-protection exception may be generated.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_store_pd1(mem_addr: *mut f64, a: __m128d) {
    let b: __m128d = simd_shuffle2(a, a, [0, 0]);
    *(mem_addr as *mut __m128d) = b;
}

/// Store 2 double-precision (64-bit) floating-point elements from `a` into
/// memory in reverse order.
/// `mem_addr` must be aligned on a 16-byte boundary or a general-protection
/// exception may be generated.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_storer_pd(mem_addr: *mut f64, a: __m128d) {
    let b: __m128d = simd_shuffle2(a, a, [1, 0]);
    *(mem_addr as *mut __m128d) = b;
}

/// Stores the upper 64 bits of a 128-bit vector of [2 x double] to a
/// memory location.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movhpd))]
pub unsafe fn _mm_storeh_pd(mem_addr: *mut f64, a: __m128d) {
    *mem_addr = simd_extract(a, 1);
}

/// Stores the lower 64 bits of a 128-bit vector of [2 x double] to a
/// memory location.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movlps))] // FIXME movlpd
pub unsafe fn _mm_storel_pd(mem_addr: *mut f64, a: __m128d) {
    *mem_addr = simd_extract(a, 0);
}

/// Load a double-precision (64-bit) floating-point element from memory
/// into both elements of returned vector.
#[inline]
#[target_feature(enable = "sse2")]
// #[cfg_attr(test, assert_instr(movapd))] // FIXME LLVM uses different codegen
pub unsafe fn _mm_load1_pd(mem_addr: *const f64) -> __m128d {
    let d = *mem_addr;
    _mm_setr_pd(d, d)
}

/// Load a double-precision (64-bit) floating-point element from memory
/// into both elements of returned vector.
#[inline]
#[target_feature(enable = "sse2")]
// #[cfg_attr(test, assert_instr(movapd))] // FIXME same as _mm_load1_pd
pub unsafe fn _mm_load_pd1(mem_addr: *const f64) -> __m128d {
    _mm_load1_pd(mem_addr)
}

/// Load 2 double-precision (64-bit) floating-point elements from memory into
/// the returned vector in reverse order. `mem_addr` must be aligned on a
/// 16-byte boundary or a general-protection exception may be generated.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movapd))]
pub unsafe fn _mm_loadr_pd(mem_addr: *const f64) -> __m128d {
    let a = _mm_load_pd(mem_addr);
    simd_shuffle2(a, a, [1, 0])
}

/// Load 128-bits (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from memory into the returned vector.
/// `mem_addr` does not need to be aligned on any particular boundary.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movups))]
pub unsafe fn _mm_loadu_pd(mem_addr: *const f64) -> __m128d {
    let mut dst = _mm_undefined_pd();
    ptr::copy_nonoverlapping(
        mem_addr as *const u8,
        &mut dst as *mut __m128d as *mut u8,
        mem::size_of::<__m128d>(),
    );
    dst
}

/// Constructs a 128-bit floating-point vector of [2 x double] from two
/// 128-bit vector parameters of [2 x double], using the immediate-value
/// parameter as a specifier.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(shufpd, imm8 = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_shuffle_pd(a: __m128d, b: __m128d, imm8: i32) -> __m128d {
    match imm8 & 0b11 {
        0b00 => simd_shuffle2(a, b, [0, 2]),
        0b01 => simd_shuffle2(a, b, [1, 2]),
        0b10 => simd_shuffle2(a, b, [0, 3]),
        _ => simd_shuffle2(a, b, [1, 3]),
    }
}

/// Constructs a 128-bit floating-point vector of [2 x double]. The lower
/// 64 bits are set to the lower 64 bits of the second parameter. The upper
/// 64 bits are set to the upper 64 bits of the first parameter.
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movsd))]
pub unsafe fn _mm_move_sd(a: __m128d, b: __m128d) -> __m128d {
    _mm_setr_pd(simd_extract(b, 0), simd_extract(a, 1))
}

/// Casts a 128-bit floating-point vector of [2 x double] into a 128-bit
/// floating-point vector of [4 x float].
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_castpd_ps(a: __m128d) -> __m128 {
    mem::transmute(a)
}

/// Casts a 128-bit floating-point vector of [2 x double] into a 128-bit
/// integer vector.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_castpd_si128(a: __m128d) -> __m128i {
    mem::transmute::<i64x2, _>(simd_cast(a))
}

/// Casts a 128-bit floating-point vector of [4 x float] into a 128-bit
/// floating-point vector of [2 x double].
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_castps_pd(a: __m128) -> __m128d {
    mem::transmute(a)
}

/// Casts a 128-bit floating-point vector of [4 x float] into a 128-bit
/// integer vector.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_castps_si128(a: __m128) -> __m128i {
    mem::transmute(a)
}

/// Casts a 128-bit integer vector into a 128-bit floating-point vector
/// of [2 x double].
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_castsi128_pd(a: __m128i) -> __m128d {
    simd_cast(a.as_i64x2())
}

/// Casts a 128-bit integer vector into a 128-bit floating-point vector
/// of [4 x float].
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_castsi128_ps(a: __m128i) -> __m128 {
    mem::transmute(a)
}

/// Return vector of type __m128d with undefined elements.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_undefined_pd() -> __m128d {
    _mm_set1_pd(mem::uninitialized())
}

/// Return vector of type __m128i with undefined elements.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn _mm_undefined_si128() -> __m128i {
    _mm_set1_epi8(mem::uninitialized())
}

/// The resulting `__m128d` element is composed by the low-order values of
/// the two `__m128d` interleaved input elements, i.e.:
///
/// * The [127:64] bits are copied from the [127:64] bits of the second input
/// * The [63:0] bits are copied from the [127:64] bits of the first input
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(unpckhpd))]
pub unsafe fn _mm_unpackhi_pd(a: __m128d, b: __m128d) -> __m128d {
    simd_shuffle2(a, b, [1, 3])
}

/// The resulting `__m128d` element is composed by the high-order values of
/// the two `__m128d` interleaved input elements, i.e.:
///
/// * The [127:64] bits are copied from the [63:0] bits of the second input
/// * The [63:0] bits are copied from the [63:0] bits of the first input
#[inline]
#[target_feature(enable = "sse2")]
#[cfg_attr(test, assert_instr(movlhps))]
pub unsafe fn _mm_unpacklo_pd(a: __m128d, b: __m128d) -> __m128d {
    simd_shuffle2(a, b, [0, 2])
}

/// Adds two signed or unsigned 64-bit integer values, returning the
/// lower 64 bits of the sum.
#[inline]
#[target_feature(enable = "sse2,mmx")]
#[cfg_attr(test, assert_instr(paddq))]
pub unsafe fn _mm_add_si64(a: __m64, b: __m64) -> __m64 {
    paddq(a, b)
}

/// Multiplies 32-bit unsigned integer values contained in the lower bits
/// of the two 64-bit integer vectors and returns the 64-bit unsigned
/// product.
#[inline]
#[target_feature(enable = "sse2,mmx")]
#[cfg_attr(test, assert_instr(pmuludq))]
pub unsafe fn _mm_mul_su32(a: __m64, b: __m64) -> __m64 {
    pmuludq2(a, b)
}

/// Subtracts signed or unsigned 64-bit integer values and writes the
/// difference to the corresponding bits in the destination.
#[inline]
#[target_feature(enable = "sse2,mmx")]
#[cfg_attr(test, assert_instr(psubq))]
pub unsafe fn _mm_sub_si64(a: __m64, b: __m64) -> __m64 {
    psubq(a, b)
}

/// Converts the two signed 32-bit integer elements of a 64-bit vector of
/// [2 x i32] into two double-precision floating-point values, returned in a
/// 128-bit vector of [2 x double].
#[inline]
#[target_feature(enable = "sse2,mmx")]
#[cfg_attr(test, assert_instr(cvtpi2pd))]
pub unsafe fn _mm_cvtpi32_pd(a: __m64) -> __m128d {
    cvtpi2pd(a)
}

/// Initializes both 64-bit values in a 128-bit vector of [2 x i64] with
/// the specified 64-bit integer values.
#[inline]
#[target_feature(enable = "sse2,mmx")]
// no particular instruction to test
pub unsafe fn _mm_set_epi64(e1: __m64, e0: __m64) -> __m128i {
    _mm_set_epi64x(mem::transmute(e1), mem::transmute(e0))
}

/// Initializes both values in a 128-bit vector of [2 x i64] with the
/// specified 64-bit value.
#[inline]
#[target_feature(enable = "sse2,mmx")]
// no particular instruction to test
pub unsafe fn _mm_set1_epi64(a: __m64) -> __m128i {
    _mm_set_epi64x(mem::transmute(a), mem::transmute(a))
}

/// Constructs a 128-bit integer vector, initialized in reverse order
/// with the specified 64-bit integral values.
#[inline]
#[target_feature(enable = "sse2,mmx")]
// no particular instruction to test
pub unsafe fn _mm_setr_epi64(e1: __m64, e0: __m64) -> __m128i {
    _mm_set_epi64x(mem::transmute(e0), mem::transmute(e1))
}

/// Returns the lower 64 bits of a 128-bit integer vector as a 64-bit
/// integer.
#[inline]
#[target_feature(enable = "sse2,mmx")]
// #[cfg_attr(test, assert_instr(movdq2q))] // FIXME: llvm codegens wrong
// instr?
pub unsafe fn _mm_movepi64_pi64(a: __m128i) -> __m64 {
    mem::transmute(simd_extract::<_, i64>(a.as_i64x2(), 0))
}

/// Moves the 64-bit operand to a 128-bit integer vector, zeroing the
/// upper bits.
#[inline]
#[target_feature(enable = "sse2,mmx")]
// #[cfg_attr(test, assert_instr(movq2dq))] // FIXME: llvm codegens wrong
// instr?
pub unsafe fn _mm_movpi64_epi64(a: __m64) -> __m128i {
    _mm_set_epi64x(0, mem::transmute(a))
}

/// Converts the two double-precision floating-point elements of a
/// 128-bit vector of [2 x double] into two signed 32-bit integer values,
/// returned in a 64-bit vector of [2 x i32].
#[inline]
#[target_feature(enable = "sse2,mmx")]
#[cfg_attr(test, assert_instr(cvtpd2pi))]
pub unsafe fn _mm_cvtpd_pi32(a: __m128d) -> __m64 {
    cvtpd2pi(a)
}

/// Converts the two double-precision floating-point elements of a
/// 128-bit vector of [2 x double] into two signed 32-bit integer values,
/// returned in a 64-bit vector of [2 x i32].
/// If the result of either conversion is inexact, the result is truncated
/// (rounded towards zero) regardless of the current MXCSR setting.
#[inline]
#[target_feature(enable = "sse2,mmx")]
#[cfg_attr(test, assert_instr(cvttpd2pi))]
pub unsafe fn _mm_cvttpd_pi32(a: __m128d) -> __m64 {
    cvttpd2pi(a)
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.sse2.pause"]
    fn pause();
    #[link_name = "llvm.x86.sse2.clflush"]
    fn clflush(p: *mut u8);
    #[link_name = "llvm.x86.sse2.lfence"]
    fn lfence();
    #[link_name = "llvm.x86.sse2.mfence"]
    fn mfence();
    #[link_name = "llvm.x86.sse2.padds.b"]
    fn paddsb(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.x86.sse2.padds.w"]
    fn paddsw(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.x86.sse2.paddus.b"]
    fn paddsub(a: u8x16, b: u8x16) -> u8x16;
    #[link_name = "llvm.x86.sse2.paddus.w"]
    fn paddsuw(a: u16x8, b: u16x8) -> u16x8;
    #[link_name = "llvm.x86.sse2.pavg.b"]
    fn pavgb(a: u8x16, b: u8x16) -> u8x16;
    #[link_name = "llvm.x86.sse2.pavg.w"]
    fn pavgw(a: u16x8, b: u16x8) -> u16x8;
    #[link_name = "llvm.x86.sse2.pmadd.wd"]
    fn pmaddwd(a: i16x8, b: i16x8) -> i32x4;
    #[link_name = "llvm.x86.sse2.pmaxs.w"]
    fn pmaxsw(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.x86.sse2.pmaxu.b"]
    fn pmaxub(a: u8x16, b: u8x16) -> u8x16;
    #[link_name = "llvm.x86.sse2.pmins.w"]
    fn pminsw(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.x86.sse2.pminu.b"]
    fn pminub(a: u8x16, b: u8x16) -> u8x16;
    #[link_name = "llvm.x86.sse2.pmulh.w"]
    fn pmulhw(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.x86.sse2.pmulhu.w"]
    fn pmulhuw(a: u16x8, b: u16x8) -> u16x8;
    #[link_name = "llvm.x86.sse2.pmulu.dq"]
    fn pmuludq(a: u32x4, b: u32x4) -> u64x2;
    #[link_name = "llvm.x86.sse2.psad.bw"]
    fn psadbw(a: u8x16, b: u8x16) -> u64x2;
    #[link_name = "llvm.x86.sse2.psubs.b"]
    fn psubsb(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.x86.sse2.psubs.w"]
    fn psubsw(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.x86.sse2.psubus.b"]
    fn psubusb(a: u8x16, b: u8x16) -> u8x16;
    #[link_name = "llvm.x86.sse2.psubus.w"]
    fn psubusw(a: u16x8, b: u16x8) -> u16x8;
    #[link_name = "llvm.x86.sse2.pslli.w"]
    fn pslliw(a: i16x8, imm8: i32) -> i16x8;
    #[link_name = "llvm.x86.sse2.psll.w"]
    fn psllw(a: i16x8, count: i16x8) -> i16x8;
    #[link_name = "llvm.x86.sse2.pslli.d"]
    fn psllid(a: i32x4, imm8: i32) -> i32x4;
    #[link_name = "llvm.x86.sse2.psll.d"]
    fn pslld(a: i32x4, count: i32x4) -> i32x4;
    #[link_name = "llvm.x86.sse2.pslli.q"]
    fn pslliq(a: i64x2, imm8: i32) -> i64x2;
    #[link_name = "llvm.x86.sse2.psll.q"]
    fn psllq(a: i64x2, count: i64x2) -> i64x2;
    #[link_name = "llvm.x86.sse2.psrai.w"]
    fn psraiw(a: i16x8, imm8: i32) -> i16x8;
    #[link_name = "llvm.x86.sse2.psra.w"]
    fn psraw(a: i16x8, count: i16x8) -> i16x8;
    #[link_name = "llvm.x86.sse2.psrai.d"]
    fn psraid(a: i32x4, imm8: i32) -> i32x4;
    #[link_name = "llvm.x86.sse2.psra.d"]
    fn psrad(a: i32x4, count: i32x4) -> i32x4;
    #[link_name = "llvm.x86.sse2.psrli.w"]
    fn psrliw(a: i16x8, imm8: i32) -> i16x8;
    #[link_name = "llvm.x86.sse2.psrl.w"]
    fn psrlw(a: i16x8, count: i16x8) -> i16x8;
    #[link_name = "llvm.x86.sse2.psrli.d"]
    fn psrlid(a: i32x4, imm8: i32) -> i32x4;
    #[link_name = "llvm.x86.sse2.psrl.d"]
    fn psrld(a: i32x4, count: i32x4) -> i32x4;
    #[link_name = "llvm.x86.sse2.psrli.q"]
    fn psrliq(a: i64x2, imm8: i32) -> i64x2;
    #[link_name = "llvm.x86.sse2.psrl.q"]
    fn psrlq(a: i64x2, count: i64x2) -> i64x2;
    #[link_name = "llvm.x86.sse2.cvtdq2ps"]
    fn cvtdq2ps(a: i32x4) -> __m128;
    #[link_name = "llvm.x86.sse2.cvtps2dq"]
    fn cvtps2dq(a: __m128) -> i32x4;
    #[link_name = "llvm.x86.sse2.maskmov.dqu"]
    fn maskmovdqu(a: i8x16, mask: i8x16, mem_addr: *mut i8);
    #[link_name = "llvm.x86.sse2.packsswb.128"]
    fn packsswb(a: i16x8, b: i16x8) -> i8x16;
    #[link_name = "llvm.x86.sse2.packssdw.128"]
    fn packssdw(a: i32x4, b: i32x4) -> i16x8;
    #[link_name = "llvm.x86.sse2.packuswb.128"]
    fn packuswb(a: i16x8, b: i16x8) -> u8x16;
    #[link_name = "llvm.x86.sse2.pmovmskb.128"]
    fn pmovmskb(a: i8x16) -> i32;
    #[link_name = "llvm.x86.sse2.max.sd"]
    fn maxsd(a: __m128d, b: __m128d) -> __m128d;
    #[link_name = "llvm.x86.sse2.max.pd"]
    fn maxpd(a: __m128d, b: __m128d) -> __m128d;
    #[link_name = "llvm.x86.sse2.min.sd"]
    fn minsd(a: __m128d, b: __m128d) -> __m128d;
    #[link_name = "llvm.x86.sse2.min.pd"]
    fn minpd(a: __m128d, b: __m128d) -> __m128d;
    #[link_name = "llvm.x86.sse2.sqrt.sd"]
    fn sqrtsd(a: __m128d) -> __m128d;
    #[link_name = "llvm.x86.sse2.sqrt.pd"]
    fn sqrtpd(a: __m128d) -> __m128d;
    #[link_name = "llvm.x86.sse2.cmp.sd"]
    fn cmpsd(a: __m128d, b: __m128d, imm8: i8) -> __m128d;
    #[link_name = "llvm.x86.sse2.cmp.pd"]
    fn cmppd(a: __m128d, b: __m128d, imm8: i8) -> __m128d;
    #[link_name = "llvm.x86.sse2.comieq.sd"]
    fn comieqsd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.comilt.sd"]
    fn comiltsd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.comile.sd"]
    fn comilesd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.comigt.sd"]
    fn comigtsd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.comige.sd"]
    fn comigesd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.comineq.sd"]
    fn comineqsd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.ucomieq.sd"]
    fn ucomieqsd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.ucomilt.sd"]
    fn ucomiltsd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.ucomile.sd"]
    fn ucomilesd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.ucomigt.sd"]
    fn ucomigtsd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.ucomige.sd"]
    fn ucomigesd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.ucomineq.sd"]
    fn ucomineqsd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.movmsk.pd"]
    fn movmskpd(a: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.cvtpd2ps"]
    fn cvtpd2ps(a: __m128d) -> __m128;
    #[link_name = "llvm.x86.sse2.cvtps2pd"]
    fn cvtps2pd(a: __m128) -> __m128d;
    #[link_name = "llvm.x86.sse2.cvtpd2dq"]
    fn cvtpd2dq(a: __m128d) -> i32x4;
    #[link_name = "llvm.x86.sse2.cvtsd2si"]
    fn cvtsd2si(a: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.cvtsd2ss"]
    fn cvtsd2ss(a: __m128, b: __m128d) -> __m128;
    #[link_name = "llvm.x86.sse2.cvtss2sd"]
    fn cvtss2sd(a: __m128d, b: __m128) -> __m128d;
    #[link_name = "llvm.x86.sse2.cvttpd2dq"]
    fn cvttpd2dq(a: __m128d) -> i32x4;
    #[link_name = "llvm.x86.sse2.cvttsd2si"]
    fn cvttsd2si(a: __m128d) -> i32;
    #[link_name = "llvm.x86.sse2.cvttps2dq"]
    fn cvttps2dq(a: __m128) -> i32x4;
    #[link_name = "llvm.x86.sse2.storeu.dq"]
    fn storeudq(mem_addr: *mut i8, a: __m128i);
    #[link_name = "llvm.x86.sse2.storeu.pd"]
    fn storeupd(mem_addr: *mut i8, a: __m128d);
    #[link_name = "llvm.x86.mmx.padd.q"]
    fn paddq(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pmulu.dq"]
    fn pmuludq2(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.psub.q"]
    fn psubq(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.sse.cvtpi2pd"]
    fn cvtpi2pd(a: __m64) -> __m128d;
    #[link_name = "llvm.x86.sse.cvtpd2pi"]
    fn cvtpd2pi(a: __m128d) -> __m64;
    #[link_name = "llvm.x86.sse.cvttpd2pi"]
    fn cvttpd2pi(a: __m128d) -> __m64;
}

#[cfg(test)]
mod tests {
    use std::f32;
    use std::f64::{self, NAN};
    use std::i32;
    use std::mem::{self, transmute};

    use coresimd::simd::*;
    use coresimd::x86::*;
    use stdsimd_test::simd_test;
    use test::black_box; // Used to inhibit constant-folding.

    #[simd_test = "sse2"]
    unsafe fn test_mm_pause() {
        _mm_pause();
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_clflush() {
        let x = 0;
        _mm_clflush(&x as *const _ as *mut u8);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_lfence() {
        _mm_lfence();
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_mfence() {
        _mm_mfence();
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_add_epi8() {
        let a = _mm_setr_epi8(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm_setr_epi8(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        let r = _mm_add_epi8(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm_setr_epi8(
            16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_add_epi8_overflow() {
        let a = _mm_set1_epi8(0x7F);
        let b = _mm_set1_epi8(1);
        let r = _mm_add_epi8(a, b);
        assert_eq_m128i(r, _mm_set1_epi8(-128));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_add_epi16() {
        let a = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm_setr_epi16(8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm_add_epi16(a, b);
        let e = _mm_setr_epi16(8, 10, 12, 14, 16, 18, 20, 22);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_add_epi32() {
        let a = _mm_setr_epi32(0, 1, 2, 3);
        let b = _mm_setr_epi32(4, 5, 6, 7);
        let r = _mm_add_epi32(a, b);
        let e = _mm_setr_epi32(4, 6, 8, 10);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_add_epi64() {
        let a = _mm_setr_epi64x(0, 1);
        let b = _mm_setr_epi64x(2, 3);
        let r = _mm_add_epi64(a, b);
        let e = _mm_setr_epi64x(2, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_adds_epi8() {
        let a = _mm_setr_epi8(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm_setr_epi8(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        let r = _mm_adds_epi8(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm_setr_epi8(
            16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_adds_epi8_saturate_positive() {
        let a = _mm_set1_epi8(0x7F);
        let b = _mm_set1_epi8(1);
        let r = _mm_adds_epi8(a, b);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_adds_epi8_saturate_negative() {
        let a = _mm_set1_epi8(-0x80);
        let b = _mm_set1_epi8(-1);
        let r = _mm_adds_epi8(a, b);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_adds_epi16() {
        let a = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm_setr_epi16(8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm_adds_epi16(a, b);
        let e = _mm_setr_epi16(8, 10, 12, 14, 16, 18, 20, 22);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_adds_epi16_saturate_positive() {
        let a = _mm_set1_epi16(0x7FFF);
        let b = _mm_set1_epi16(1);
        let r = _mm_adds_epi16(a, b);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_adds_epi16_saturate_negative() {
        let a = _mm_set1_epi16(-0x8000);
        let b = _mm_set1_epi16(-1);
        let r = _mm_adds_epi16(a, b);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_adds_epu8() {
        let a = _mm_setr_epi8(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm_setr_epi8(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        let r = _mm_adds_epu8(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm_setr_epi8(
            16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_adds_epu8_saturate() {
        let a = _mm_set1_epi8(!0);
        let b = _mm_set1_epi8(1);
        let r = _mm_adds_epu8(a, b);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_adds_epu16() {
        let a = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm_setr_epi16(8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm_adds_epu16(a, b);
        let e = _mm_setr_epi16(8, 10, 12, 14, 16, 18, 20, 22);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_adds_epu16_saturate() {
        let a = _mm_set1_epi16(!0);
        let b = _mm_set1_epi16(1);
        let r = _mm_adds_epu16(a, b);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_avg_epu8() {
        let (a, b) = (_mm_set1_epi8(3), _mm_set1_epi8(9));
        let r = _mm_avg_epu8(a, b);
        assert_eq_m128i(r, _mm_set1_epi8(6));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_avg_epu16() {
        let (a, b) = (_mm_set1_epi16(3), _mm_set1_epi16(9));
        let r = _mm_avg_epu16(a, b);
        assert_eq_m128i(r, _mm_set1_epi16(6));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_madd_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm_madd_epi16(a, b);
        let e = _mm_setr_epi32(29, 81, 149, 233);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_max_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(-1);
        let r = _mm_max_epi16(a, b);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_max_epu8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(!0);
        let r = _mm_max_epu8(a, b);
        assert_eq_m128i(r, b);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_min_epi16() {
        let a = _mm_set1_epi16(1);
        let b = _mm_set1_epi16(-1);
        let r = _mm_min_epi16(a, b);
        assert_eq_m128i(r, b);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_min_epu8() {
        let a = _mm_set1_epi8(1);
        let b = _mm_set1_epi8(!0);
        let r = _mm_min_epu8(a, b);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_mulhi_epi16() {
        let (a, b) = (_mm_set1_epi16(1000), _mm_set1_epi16(-1001));
        let r = _mm_mulhi_epi16(a, b);
        assert_eq_m128i(r, _mm_set1_epi16(-16));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_mulhi_epu16() {
        let (a, b) = (_mm_set1_epi16(1000), _mm_set1_epi16(1001));
        let r = _mm_mulhi_epu16(a, b);
        assert_eq_m128i(r, _mm_set1_epi16(15));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_mullo_epi16() {
        let (a, b) = (_mm_set1_epi16(1000), _mm_set1_epi16(-1001));
        let r = _mm_mullo_epi16(a, b);
        assert_eq_m128i(r, _mm_set1_epi16(-17960));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_mul_epu32() {
        let a = _mm_setr_epi64x(1_000_000_000, 1 << 34);
        let b = _mm_setr_epi64x(1_000_000_000, 1 << 35);
        let r = _mm_mul_epu32(a, b);
        let e = _mm_setr_epi64x(1_000_000_000 * 1_000_000_000, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sad_epu8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            255u8 as i8, 254u8 as i8, 253u8 as i8, 252u8 as i8,
            1, 2, 3, 4,
            155u8 as i8, 154u8 as i8, 153u8 as i8, 152u8 as i8,
            1, 2, 3, 4,
        );
        let b = _mm_setr_epi8(0, 0, 0, 0, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2);
        let r = _mm_sad_epu8(a, b);
        let e = _mm_setr_epi64x(1020, 614);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sub_epi8() {
        let (a, b) = (_mm_set1_epi8(5), _mm_set1_epi8(6));
        let r = _mm_sub_epi8(a, b);
        assert_eq_m128i(r, _mm_set1_epi8(-1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sub_epi16() {
        let (a, b) = (_mm_set1_epi16(5), _mm_set1_epi16(6));
        let r = _mm_sub_epi16(a, b);
        assert_eq_m128i(r, _mm_set1_epi16(-1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sub_epi32() {
        let (a, b) = (_mm_set1_epi32(5), _mm_set1_epi32(6));
        let r = _mm_sub_epi32(a, b);
        assert_eq_m128i(r, _mm_set1_epi32(-1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sub_epi64() {
        let (a, b) = (_mm_set1_epi64x(5), _mm_set1_epi64x(6));
        let r = _mm_sub_epi64(a, b);
        assert_eq_m128i(r, _mm_set1_epi64x(-1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_subs_epi8() {
        let (a, b) = (_mm_set1_epi8(5), _mm_set1_epi8(2));
        let r = _mm_subs_epi8(a, b);
        assert_eq_m128i(r, _mm_set1_epi8(3));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_subs_epi8_saturate_positive() {
        let a = _mm_set1_epi8(0x7F);
        let b = _mm_set1_epi8(-1);
        let r = _mm_subs_epi8(a, b);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_subs_epi8_saturate_negative() {
        let a = _mm_set1_epi8(-0x80);
        let b = _mm_set1_epi8(1);
        let r = _mm_subs_epi8(a, b);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_subs_epi16() {
        let (a, b) = (_mm_set1_epi16(5), _mm_set1_epi16(2));
        let r = _mm_subs_epi16(a, b);
        assert_eq_m128i(r, _mm_set1_epi16(3));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_subs_epi16_saturate_positive() {
        let a = _mm_set1_epi16(0x7FFF);
        let b = _mm_set1_epi16(-1);
        let r = _mm_subs_epi16(a, b);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_subs_epi16_saturate_negative() {
        let a = _mm_set1_epi16(-0x8000);
        let b = _mm_set1_epi16(1);
        let r = _mm_subs_epi16(a, b);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_subs_epu8() {
        let (a, b) = (_mm_set1_epi8(5), _mm_set1_epi8(2));
        let r = _mm_subs_epu8(a, b);
        assert_eq_m128i(r, _mm_set1_epi8(3));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_subs_epu8_saturate() {
        let a = _mm_set1_epi8(0);
        let b = _mm_set1_epi8(1);
        let r = _mm_subs_epu8(a, b);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_subs_epu16() {
        let (a, b) = (_mm_set1_epi16(5), _mm_set1_epi16(2));
        let r = _mm_subs_epu16(a, b);
        assert_eq_m128i(r, _mm_set1_epi16(3));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_subs_epu16_saturate() {
        let a = _mm_set1_epi16(0);
        let b = _mm_set1_epi16(1);
        let r = _mm_subs_epu16(a, b);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_slli_si128() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        );
        let r = _mm_slli_si128(a, 1);
        let e = _mm_setr_epi8(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        );
        assert_eq_m128i(r, e);

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        );
        let r = _mm_slli_si128(a, 15);
        let e = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
        assert_eq_m128i(r, e);

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        );
        let r = _mm_slli_si128(a, 16);
        assert_eq_m128i(r, _mm_set1_epi8(0));

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        );
        let r = _mm_slli_si128(a, -1);
        assert_eq_m128i(_mm_set1_epi8(0), r);

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        );
        let r = _mm_slli_si128(a, -0x80000000);
        assert_eq_m128i(r, _mm_set1_epi8(0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_slli_epi16() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi16(
            0xFFFF as u16 as i16, 0x0FFF, 0x00FF, 0x000F, 0, 0, 0, 0,
        );
        let r = _mm_slli_epi16(a, 4);

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm_setr_epi16(
            0xFFF0 as u16 as i16, 0xFFF0 as u16 as i16, 0x0FF0, 0x00F0,
            0, 0, 0, 0,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sll_epi16() {
        let a = _mm_setr_epi16(0xFF, 0, 0, 0, 0, 0, 0, 0);
        let r = _mm_sll_epi16(a, _mm_setr_epi16(4, 0, 0, 0, 0, 0, 0, 0));
        assert_eq_m128i(r, _mm_setr_epi16(0xFF0, 0, 0, 0, 0, 0, 0, 0));
        let r = _mm_sll_epi16(a, _mm_setr_epi16(0, 0, 0, 0, 4, 0, 0, 0));
        assert_eq_m128i(r, _mm_setr_epi16(0xFF, 0, 0, 0, 0, 0, 0, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_slli_epi32() {
        let r = _mm_slli_epi32(_mm_set1_epi32(0xFFFF), 4);
        assert_eq_m128i(r, _mm_set1_epi32(0xFFFF0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sll_epi32() {
        let a = _mm_set1_epi32(0xFFFF);
        let b = _mm_setr_epi32(4, 0, 0, 0);
        let r = _mm_sll_epi32(a, b);
        assert_eq_m128i(r, _mm_set1_epi32(0xFFFF0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_slli_epi64() {
        let r = _mm_slli_epi64(_mm_set1_epi64x(0xFFFFFFFF), 4);
        assert_eq_m128i(r, _mm_set1_epi64x(0xFFFFFFFF0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sll_epi64() {
        let a = _mm_set1_epi64x(0xFFFFFFFF);
        let b = _mm_setr_epi64x(4, 0);
        let r = _mm_sll_epi64(a, b);
        assert_eq_m128i(r, _mm_set1_epi64x(0xFFFFFFFF0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_srai_epi16() {
        let r = _mm_srai_epi16(_mm_set1_epi16(-1), 1);
        assert_eq_m128i(r, _mm_set1_epi16(-1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sra_epi16() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_setr_epi16(1, 0, 0, 0, 0, 0, 0, 0);
        let r = _mm_sra_epi16(a, b);
        assert_eq_m128i(r, _mm_set1_epi16(-1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_srai_epi32() {
        let r = _mm_srai_epi32(_mm_set1_epi32(-1), 1);
        assert_eq_m128i(r, _mm_set1_epi32(-1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sra_epi32() {
        let a = _mm_set1_epi32(-1);
        let b = _mm_setr_epi32(1, 0, 0, 0);
        let r = _mm_sra_epi32(a, b);
        assert_eq_m128i(r, _mm_set1_epi32(-1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_srli_si128() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        );
        let r = _mm_srli_si128(a, 1);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm_setr_epi8(
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0,
        );
        assert_eq_m128i(r, e);

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        );
        let r = _mm_srli_si128(a, 15);
        let e = _mm_setr_epi8(16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        );
        let r = _mm_srli_si128(a, 16);
        assert_eq_m128i(r, _mm_set1_epi8(0));

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        );
        let r = _mm_srli_si128(a, -1);
        assert_eq_m128i(r, _mm_set1_epi8(0));

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        );
        let r = _mm_srli_si128(a, -0x80000000);
        assert_eq_m128i(r, _mm_set1_epi8(0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_srli_epi16() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi16(
            0xFFFF as u16 as i16, 0x0FFF, 0x00FF, 0x000F, 0, 0, 0, 0,
        );
        let r = _mm_srli_epi16(a, 4);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm_setr_epi16(
            0xFFF as u16 as i16, 0xFF as u16 as i16, 0xF, 0, 0, 0, 0, 0,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_srl_epi16() {
        let a = _mm_setr_epi16(0xFF, 0, 0, 0, 0, 0, 0, 0);
        let r = _mm_srl_epi16(a, _mm_setr_epi16(4, 0, 0, 0, 0, 0, 0, 0));
        assert_eq_m128i(r, _mm_setr_epi16(0xF, 0, 0, 0, 0, 0, 0, 0));
        let r = _mm_srl_epi16(a, _mm_setr_epi16(0, 0, 0, 0, 4, 0, 0, 0));
        assert_eq_m128i(r, _mm_setr_epi16(0xFF, 0, 0, 0, 0, 0, 0, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_srli_epi32() {
        let r = _mm_srli_epi32(_mm_set1_epi32(0xFFFF), 4);
        assert_eq_m128i(r, _mm_set1_epi32(0xFFF));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_srl_epi32() {
        let a = _mm_set1_epi32(0xFFFF);
        let b = _mm_setr_epi32(4, 0, 0, 0);
        let r = _mm_srl_epi32(a, b);
        assert_eq_m128i(r, _mm_set1_epi32(0xFFF));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_srli_epi64() {
        let r = _mm_srli_epi64(_mm_set1_epi64x(0xFFFFFFFF), 4);
        assert_eq_m128i(r, _mm_set1_epi64x(0xFFFFFFF));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_srl_epi64() {
        let a = _mm_set1_epi64x(0xFFFFFFFF);
        let b = _mm_setr_epi64x(4, 0);
        let r = _mm_srl_epi64(a, b);
        assert_eq_m128i(r, _mm_set1_epi64x(0xFFFFFFF));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_and_si128() {
        let a = _mm_set1_epi8(5);
        let b = _mm_set1_epi8(3);
        let r = _mm_and_si128(a, b);
        assert_eq_m128i(r, _mm_set1_epi8(1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_andnot_si128() {
        let a = _mm_set1_epi8(5);
        let b = _mm_set1_epi8(3);
        let r = _mm_andnot_si128(a, b);
        assert_eq_m128i(r, _mm_set1_epi8(2));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_or_si128() {
        let a = _mm_set1_epi8(5);
        let b = _mm_set1_epi8(3);
        let r = _mm_or_si128(a, b);
        assert_eq_m128i(r, _mm_set1_epi8(7));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_xor_si128() {
        let a = _mm_set1_epi8(5);
        let b = _mm_set1_epi8(3);
        let r = _mm_xor_si128(a, b);
        assert_eq_m128i(r, _mm_set1_epi8(6));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpeq_epi8() {
        let a = _mm_setr_epi8(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        );
        let b = _mm_setr_epi8(
            15,
            14,
            2,
            12,
            11,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            0,
        );
        let r = _mm_cmpeq_epi8(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        assert_eq_m128i(
            r,
            _mm_setr_epi8(
                0, 0, 0xFFu8 as i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            )
        );
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpeq_epi16() {
        let a = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm_setr_epi16(7, 6, 2, 4, 3, 2, 1, 0);
        let r = _mm_cmpeq_epi16(a, b);
        assert_eq_m128i(r, _mm_setr_epi16(0, 0, !0, 0, 0, 0, 0, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpeq_epi32() {
        let a = _mm_setr_epi32(0, 1, 2, 3);
        let b = _mm_setr_epi32(3, 2, 2, 0);
        let r = _mm_cmpeq_epi32(a, b);
        assert_eq_m128i(r, _mm_setr_epi32(0, 0, !0, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpgt_epi8() {
        let a = _mm_set_epi8(5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let b = _mm_set1_epi8(0);
        let r = _mm_cmpgt_epi8(a, b);
        let e = _mm_set_epi8(!0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpgt_epi16() {
        let a = _mm_set_epi16(5, 0, 0, 0, 0, 0, 0, 0);
        let b = _mm_set1_epi16(0);
        let r = _mm_cmpgt_epi16(a, b);
        let e = _mm_set_epi16(!0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpgt_epi32() {
        let a = _mm_set_epi32(5, 0, 0, 0);
        let b = _mm_set1_epi32(0);
        let r = _mm_cmpgt_epi32(a, b);
        assert_eq_m128i(r, _mm_set_epi32(!0, 0, 0, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmplt_epi8() {
        let a = _mm_set1_epi8(0);
        let b = _mm_set_epi8(5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r = _mm_cmplt_epi8(a, b);
        let e = _mm_set_epi8(!0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmplt_epi16() {
        let a = _mm_set1_epi16(0);
        let b = _mm_set_epi16(5, 0, 0, 0, 0, 0, 0, 0);
        let r = _mm_cmplt_epi16(a, b);
        let e = _mm_set_epi16(!0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmplt_epi32() {
        let a = _mm_set1_epi32(0);
        let b = _mm_set_epi32(5, 0, 0, 0);
        let r = _mm_cmplt_epi32(a, b);
        assert_eq_m128i(r, _mm_set_epi32(!0, 0, 0, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtepi32_pd() {
        let a = _mm_set_epi32(35, 25, 15, 5);
        let r = _mm_cvtepi32_pd(a);
        assert_eq_m128d(r, _mm_setr_pd(5.0, 15.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtsi32_sd() {
        let a = _mm_set1_pd(3.5);
        let r = _mm_cvtsi32_sd(a, 5);
        assert_eq_m128d(r, _mm_setr_pd(5.0, 3.5));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtepi32_ps() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let r = _mm_cvtepi32_ps(a);
        assert_eq_m128(r, _mm_setr_ps(1.0, 2.0, 3.0, 4.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtps_epi32() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let r = _mm_cvtps_epi32(a);
        assert_eq_m128i(r, _mm_setr_epi32(1, 2, 3, 4));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtsi32_si128() {
        let r = _mm_cvtsi32_si128(5);
        assert_eq_m128i(r, _mm_setr_epi32(5, 0, 0, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtsi128_si32() {
        let r = _mm_cvtsi128_si32(_mm_setr_epi32(5, 0, 0, 0));
        assert_eq!(r, 5);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_set_epi64x() {
        let r = _mm_set_epi64x(0, 1);
        assert_eq_m128i(r, _mm_setr_epi64x(1, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_set_epi32() {
        let r = _mm_set_epi32(0, 1, 2, 3);
        assert_eq_m128i(r, _mm_setr_epi32(3, 2, 1, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_set_epi16() {
        let r = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq_m128i(r, _mm_setr_epi16(7, 6, 5, 4, 3, 2, 1, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_set_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let r = _mm_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm_setr_epi8(
            15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_set1_epi64x() {
        let r = _mm_set1_epi64x(1);
        assert_eq_m128i(r, _mm_set1_epi64x(1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_set1_epi32() {
        let r = _mm_set1_epi32(1);
        assert_eq_m128i(r, _mm_set1_epi32(1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_set1_epi16() {
        let r = _mm_set1_epi16(1);
        assert_eq_m128i(r, _mm_set1_epi16(1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_set1_epi8() {
        let r = _mm_set1_epi8(1);
        assert_eq_m128i(r, _mm_set1_epi8(1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_setr_epi32() {
        let r = _mm_setr_epi32(0, 1, 2, 3);
        assert_eq_m128i(r, _mm_setr_epi32(0, 1, 2, 3));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_setr_epi16() {
        let r = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq_m128i(r, _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_setr_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let r = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_setzero_si128() {
        let r = _mm_setzero_si128();
        assert_eq_m128i(r, _mm_set1_epi64x(0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_loadl_epi64() {
        let a = _mm_setr_epi64x(6, 5);
        let r = _mm_loadl_epi64(&a as *const _);
        assert_eq_m128i(r, _mm_setr_epi64x(6, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_load_si128() {
        let a = _mm_set_epi64x(5, 6);
        let r = _mm_load_si128(&a as *const _ as *const _);
        assert_eq_m128i(a, r);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_loadu_si128() {
        let a = _mm_set_epi64x(5, 6);
        let r = _mm_loadu_si128(&a as *const _ as *const _);
        assert_eq_m128i(a, r);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_maskmoveu_si128() {
        let a = _mm_set1_epi8(9);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let mask = _mm_set_epi8(
            0, 0, 0x80u8 as i8, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        );
        let mut r = _mm_set1_epi8(0);
        _mm_maskmoveu_si128(a, mask, &mut r as *mut _ as *mut i8);
        let e = _mm_set_epi8(0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_store_si128() {
        let a = _mm_set1_epi8(9);
        let mut r = _mm_set1_epi8(0);
        _mm_store_si128(&mut r as *mut _ as *mut __m128i, a);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_storeu_si128() {
        let a = _mm_set1_epi8(9);
        let mut r = _mm_set1_epi8(0);
        _mm_storeu_si128(&mut r as *mut _ as *mut __m128i, a);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_storel_epi64() {
        let a = _mm_setr_epi64x(2, 9);
        let mut r = _mm_set1_epi8(0);
        _mm_storel_epi64(&mut r as *mut _ as *mut __m128i, a);
        assert_eq_m128i(r, _mm_setr_epi64x(2, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_stream_si128() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let mut r = _mm_undefined_si128();
        _mm_stream_si128(&mut r as *mut _, a);
        assert_eq_m128i(r, a);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_stream_si32() {
        let a: i32 = 7;
        let mut mem = ::std::boxed::Box::<i32>::new(-1);
        _mm_stream_si32(&mut *mem as *mut i32, a);
        assert_eq!(a, *mem);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_move_epi64() {
        let a = _mm_setr_epi64x(5, 6);
        let r = _mm_move_epi64(a);
        assert_eq_m128i(r, _mm_setr_epi64x(5, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_packs_epi16() {
        let a = _mm_setr_epi16(0x80, -0x81, 0, 0, 0, 0, 0, 0);
        let b = _mm_setr_epi16(0, 0, 0, 0, 0, 0, -0x81, 0x80);
        let r = _mm_packs_epi16(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        assert_eq_m128i(
            r,
            _mm_setr_epi8(
                0x7F, -0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0x80, 0x7F
            )
        );
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_packs_epi32() {
        let a = _mm_setr_epi32(0x8000, -0x8001, 0, 0);
        let b = _mm_setr_epi32(0, 0, -0x8001, 0x8000);
        let r = _mm_packs_epi32(a, b);
        assert_eq_m128i(
            r,
            _mm_setr_epi16(0x7FFF, -0x8000, 0, 0, 0, 0, -0x8000, 0x7FFF),
        );
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_packus_epi16() {
        let a = _mm_setr_epi16(0x100, -1, 0, 0, 0, 0, 0, 0);
        let b = _mm_setr_epi16(0, 0, 0, 0, 0, 0, -1, 0x100);
        let r = _mm_packus_epi16(a, b);
        assert_eq_m128i(
            r,
            _mm_setr_epi8(!0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, !0),
        );
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_extract_epi16() {
        let a = _mm_setr_epi16(-1, 1, 2, 3, 4, 5, 6, 7);
        let r1 = _mm_extract_epi16(a, 0);
        let r2 = _mm_extract_epi16(a, 11);
        assert_eq!(r1, -1);
        assert_eq!(r2, 3);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_insert_epi16() {
        let a = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_insert_epi16(a, 9, 0);
        let e = _mm_setr_epi16(9, 1, 2, 3, 4, 5, 6, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_movemask_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            0b1000_0000u8 as i8, 0b0, 0b1000_0000u8 as i8, 0b01,
            0b0101, 0b1111_0000u8 as i8, 0, 0,
            0, 0, 0b1111_0000u8 as i8, 0b0101,
            0b01, 0b1000_0000u8 as i8, 0b0, 0b1000_0000u8 as i8,
        );
        let r = _mm_movemask_epi8(a);
        assert_eq!(r, 0b10100100_00100101);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_shuffle_epi32() {
        let a = _mm_setr_epi32(5, 10, 15, 20);
        let r = _mm_shuffle_epi32(a, 0b00_01_01_11);
        let e = _mm_setr_epi32(20, 10, 10, 5);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_shufflehi_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 10, 15, 20);
        let r = _mm_shufflehi_epi16(a, 0b00_01_01_11);
        let e = _mm_setr_epi16(1, 2, 3, 4, 20, 10, 10, 5);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_shufflelo_epi16() {
        let a = _mm_setr_epi16(5, 10, 15, 20, 1, 2, 3, 4);
        let r = _mm_shufflelo_epi16(a, 0b00_01_01_11);
        let e = _mm_setr_epi16(20, 10, 10, 5, 1, 2, 3, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_unpackhi_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm_setr_epi8(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        let r = _mm_unpackhi_epi8(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm_setr_epi8(
            8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_unpackhi_epi16() {
        let a = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm_setr_epi16(8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm_unpackhi_epi16(a, b);
        let e = _mm_setr_epi16(4, 12, 5, 13, 6, 14, 7, 15);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_unpackhi_epi32() {
        let a = _mm_setr_epi32(0, 1, 2, 3);
        let b = _mm_setr_epi32(4, 5, 6, 7);
        let r = _mm_unpackhi_epi32(a, b);
        let e = _mm_setr_epi32(2, 6, 3, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_unpackhi_epi64() {
        let a = _mm_setr_epi64x(0, 1);
        let b = _mm_setr_epi64x(2, 3);
        let r = _mm_unpackhi_epi64(a, b);
        let e = _mm_setr_epi64x(1, 3);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_unpacklo_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm_setr_epi8(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        let r = _mm_unpacklo_epi8(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm_setr_epi8(
            0, 16, 1, 17, 2, 18, 3, 19,
            4, 20, 5, 21, 6, 22, 7, 23,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_unpacklo_epi16() {
        let a = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let b = _mm_setr_epi16(8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm_unpacklo_epi16(a, b);
        let e = _mm_setr_epi16(0, 8, 1, 9, 2, 10, 3, 11);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_unpacklo_epi32() {
        let a = _mm_setr_epi32(0, 1, 2, 3);
        let b = _mm_setr_epi32(4, 5, 6, 7);
        let r = _mm_unpacklo_epi32(a, b);
        let e = _mm_setr_epi32(0, 4, 1, 5);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_unpacklo_epi64() {
        let a = _mm_setr_epi64x(0, 1);
        let b = _mm_setr_epi64x(2, 3);
        let r = _mm_unpacklo_epi64(a, b);
        let e = _mm_setr_epi64x(0, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_add_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_add_sd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(6.0, 2.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_add_pd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_add_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(6.0, 12.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_div_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_div_sd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(0.2, 2.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_div_pd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_div_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(0.2, 0.2));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_max_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_max_sd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(5.0, 2.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_max_pd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_max_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(5.0, 10.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_min_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_min_sd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(1.0, 2.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_min_pd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_min_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(1.0, 2.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_mul_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_mul_sd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(5.0, 2.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_mul_pd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_mul_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(5.0, 20.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sqrt_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_sqrt_sd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(5.0f64.sqrt(), 2.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sqrt_pd() {
        let r = _mm_sqrt_pd(_mm_setr_pd(1.0, 2.0));
        assert_eq_m128d(r, _mm_setr_pd(1.0f64.sqrt(), 2.0f64.sqrt()));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sub_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_sub_sd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(-4.0, 2.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_sub_pd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_sub_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(-4.0, -8.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_and_pd() {
        let a = transmute(u64x2::splat(5));
        let b = transmute(u64x2::splat(3));
        let r = _mm_and_pd(a, b);
        let e = transmute(u64x2::splat(1));
        assert_eq_m128d(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_andnot_pd() {
        let a = transmute(u64x2::splat(5));
        let b = transmute(u64x2::splat(3));
        let r = _mm_andnot_pd(a, b);
        let e = transmute(u64x2::splat(2));
        assert_eq_m128d(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_or_pd() {
        let a = transmute(u64x2::splat(5));
        let b = transmute(u64x2::splat(3));
        let r = _mm_or_pd(a, b);
        let e = transmute(u64x2::splat(7));
        assert_eq_m128d(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_xor_pd() {
        let a = transmute(u64x2::splat(5));
        let b = transmute(u64x2::splat(3));
        let r = _mm_xor_pd(a, b);
        let e = transmute(u64x2::splat(6));
        assert_eq_m128d(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpeq_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(!0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpeq_sd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmplt_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(!0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmplt_sd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmple_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(!0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmple_sd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpgt_sd() {
        let (a, b) = (_mm_setr_pd(5.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(!0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpgt_sd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpge_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(!0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpge_sd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpord_sd() {
        let (a, b) = (_mm_setr_pd(NAN, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpord_sd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpunord_sd() {
        let (a, b) = (_mm_setr_pd(NAN, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(!0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpunord_sd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpneq_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(!0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpneq_sd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpnlt_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpnlt_sd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpnle_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpnle_sd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpngt_sd() {
        let (a, b) = (_mm_setr_pd(5.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpngt_sd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpnge_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpnge_sd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpeq_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(!0, 0);
        let r = transmute::<_, __m128i>(_mm_cmpeq_pd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmplt_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, !0);
        let r = transmute::<_, __m128i>(_mm_cmplt_pd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmple_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(!0, !0);
        let r = transmute::<_, __m128i>(_mm_cmple_pd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpgt_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, 0);
        let r = transmute::<_, __m128i>(_mm_cmpgt_pd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpge_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(!0, 0);
        let r = transmute::<_, __m128i>(_mm_cmpge_pd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpord_pd() {
        let (a, b) = (_mm_setr_pd(NAN, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(0, !0);
        let r = transmute::<_, __m128i>(_mm_cmpord_pd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpunord_pd() {
        let (a, b) = (_mm_setr_pd(NAN, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(!0, 0);
        let r = transmute::<_, __m128i>(_mm_cmpunord_pd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpneq_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(!0, !0);
        let r = transmute::<_, __m128i>(_mm_cmpneq_pd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpnlt_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(0, 0);
        let r = transmute::<_, __m128i>(_mm_cmpnlt_pd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpnle_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, 0);
        let r = transmute::<_, __m128i>(_mm_cmpnle_pd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpngt_pd() {
        let (a, b) = (_mm_setr_pd(5.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, !0);
        let r = transmute::<_, __m128i>(_mm_cmpngt_pd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cmpnge_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, !0);
        let r = transmute::<_, __m128i>(_mm_cmpnge_pd(a, b));
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_comieq_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_comieq_sd(a, b) != 0);

        let (a, b) = (_mm_setr_pd(NAN, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_comieq_sd(a, b) == 0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_comilt_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_comilt_sd(a, b) == 0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_comile_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_comile_sd(a, b) != 0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_comigt_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_comigt_sd(a, b) == 0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_comige_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_comige_sd(a, b) != 0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_comineq_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_comineq_sd(a, b) == 0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_ucomieq_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_ucomieq_sd(a, b) != 0);

        let (a, b) = (_mm_setr_pd(NAN, 2.0), _mm_setr_pd(NAN, 3.0));
        assert!(_mm_ucomieq_sd(a, b) == 0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_ucomilt_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_ucomilt_sd(a, b) == 0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_ucomile_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_ucomile_sd(a, b) != 0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_ucomigt_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_ucomigt_sd(a, b) == 0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_ucomige_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_ucomige_sd(a, b) != 0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_ucomineq_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_ucomineq_sd(a, b) == 0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_movemask_pd() {
        let r = _mm_movemask_pd(_mm_setr_pd(-1.0, 5.0));
        assert_eq!(r, 0b01);

        let r = _mm_movemask_pd(_mm_setr_pd(-1.0, -5.0));
        assert_eq!(r, 0b11);
    }

    #[repr(align(16))]
    struct Memory {
        data: [f64; 4],
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_load_pd() {
        let mem = Memory {
            data: [1.0f64, 2.0, 3.0, 4.0],
        };
        let vals = &mem.data;
        let d = vals.as_ptr();

        let r = _mm_load_pd(d);
        assert_eq_m128d(r, _mm_setr_pd(1.0, 2.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_load_sd() {
        let a = 1.;
        let expected = _mm_setr_pd(a, 0.);
        let r = _mm_load_sd(&a);
        assert_eq_m128d(r, expected);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_loadh_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = 3.;
        let expected = _mm_setr_pd(_mm_cvtsd_f64(a), 3.);
        let r = _mm_loadh_pd(a, &b);
        assert_eq_m128d(r, expected);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_loadl_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = 3.;
        let expected = _mm_setr_pd(3., get_m128d(a, 1));
        let r = _mm_loadl_pd(a, &b);
        assert_eq_m128d(r, expected);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_stream_pd() {
        #[repr(align(128))]
        struct Memory {
            pub data: [f64; 2],
        }
        let a = _mm_set1_pd(7.0);
        let mut mem = Memory {
            data: [-1.0; 2],
        };

        _mm_stream_pd(&mut mem.data[0] as *mut f64, a);
        for i in 0..2 {
            assert_eq!(mem.data[i], get_m128d(a, i));
        }
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_store_sd() {
        let mut dest = 0.;
        let a = _mm_setr_pd(1., 2.);
        _mm_store_sd(&mut dest, a);
        assert_eq!(dest, _mm_cvtsd_f64(a));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_store_pd() {
        let mut mem = Memory {
            data: [0.0f64; 4],
        };
        let vals = &mut mem.data;
        let a = _mm_setr_pd(1.0, 2.0);
        let d = vals.as_mut_ptr();

        _mm_store_pd(d, *black_box(&a));
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 2.0);
    }

    #[simd_test = "sse"]
    unsafe fn test_mm_storeu_pd() {
        let mut mem = Memory {
            data: [0.0f64; 4],
        };
        let vals = &mut mem.data;
        let a = _mm_setr_pd(1.0, 2.0);

        let mut ofs = 0;
        let mut p = vals.as_mut_ptr();

        // Make sure p is *not* aligned to 16-byte boundary
        if (p as usize) & 0xf == 0 {
            ofs = 1;
            p = p.offset(1);
        }

        _mm_storeu_pd(p, *black_box(&a));

        if ofs > 0 {
            assert_eq!(vals[ofs - 1], 0.0);
        }
        assert_eq!(vals[ofs + 0], 1.0);
        assert_eq!(vals[ofs + 1], 2.0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_store1_pd() {
        let mut mem = Memory {
            data: [0.0f64; 4],
        };
        let vals = &mut mem.data;
        let a = _mm_setr_pd(1.0, 2.0);
        let d = vals.as_mut_ptr();

        _mm_store1_pd(d, *black_box(&a));
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 1.0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_store_pd1() {
        let mut mem = Memory {
            data: [0.0f64; 4],
        };
        let vals = &mut mem.data;
        let a = _mm_setr_pd(1.0, 2.0);
        let d = vals.as_mut_ptr();

        _mm_store_pd1(d, *black_box(&a));
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 1.0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_storer_pd() {
        let mut mem = Memory {
            data: [0.0f64; 4],
        };
        let vals = &mut mem.data;
        let a = _mm_setr_pd(1.0, 2.0);
        let d = vals.as_mut_ptr();

        _mm_storer_pd(d, *black_box(&a));
        assert_eq!(vals[0], 2.0);
        assert_eq!(vals[1], 1.0);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_storeh_pd() {
        let mut dest = 0.;
        let a = _mm_setr_pd(1., 2.);
        _mm_storeh_pd(&mut dest, a);
        assert_eq!(dest, get_m128d(a, 1));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_storel_pd() {
        let mut dest = 0.;
        let a = _mm_setr_pd(1., 2.);
        _mm_storel_pd(&mut dest, a);
        assert_eq!(dest, _mm_cvtsd_f64(a));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_loadr_pd() {
        let mut mem = Memory {
            data: [1.0f64, 2.0, 3.0, 4.0],
        };
        let vals = &mut mem.data;
        let d = vals.as_ptr();

        let r = _mm_loadr_pd(d);
        assert_eq_m128d(r, _mm_setr_pd(2.0, 1.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_loadu_pd() {
        let mut mem = Memory {
            data: [1.0f64, 2.0, 3.0, 4.0],
        };
        let vals = &mut mem.data;
        let mut d = vals.as_ptr();

        // make sure d is not aligned to 16-byte boundary
        let mut offset = 0;
        if (d as usize) & 0xf == 0 {
            offset = 1;
            d = d.offset(offset as isize);
        }

        let r = _mm_loadu_pd(d);
        let e = _mm_add_pd(
            _mm_setr_pd(1.0, 2.0),
            _mm_set1_pd(offset as f64),
        );
        assert_eq_m128d(r, e);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtpd_ps() {
        let r = _mm_cvtpd_ps(_mm_setr_pd(-1.0, 5.0));
        assert_eq_m128(r, _mm_setr_ps(-1.0, 5.0, 0.0, 0.0));

        let r = _mm_cvtpd_ps(_mm_setr_pd(-1.0, -5.0));
        assert_eq_m128(r, _mm_setr_ps(-1.0, -5.0, 0.0, 0.0));

        let r = _mm_cvtpd_ps(_mm_setr_pd(f64::MAX, f64::MIN));
        assert_eq_m128(
            r,
            _mm_setr_ps(f32::INFINITY, f32::NEG_INFINITY, 0.0, 0.0),
        );

        let r = _mm_cvtpd_ps(_mm_setr_pd(f32::MAX as f64, f32::MIN as f64));
        assert_eq_m128(r, _mm_setr_ps(f32::MAX, f32::MIN, 0.0, 0.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtps_pd() {
        let r = _mm_cvtps_pd(_mm_setr_ps(-1.0, 2.0, -3.0, 5.0));
        assert_eq_m128d(r, _mm_setr_pd(-1.0, 2.0));

        let r = _mm_cvtps_pd(_mm_setr_ps(
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN,
        ));
        assert_eq_m128d(r, _mm_setr_pd(f32::MAX as f64, f64::INFINITY));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtpd_epi32() {
        let r = _mm_cvtpd_epi32(_mm_setr_pd(-1.0, 5.0));
        assert_eq_m128i(r, _mm_setr_epi32(-1, 5, 0, 0));

        let r = _mm_cvtpd_epi32(_mm_setr_pd(-1.0, -5.0));
        assert_eq_m128i(r, _mm_setr_epi32(-1, -5, 0, 0));

        let r = _mm_cvtpd_epi32(_mm_setr_pd(f64::MAX, f64::MIN));
        assert_eq_m128i(r, _mm_setr_epi32(i32::MIN, i32::MIN, 0, 0));

        let r = _mm_cvtpd_epi32(_mm_setr_pd(f64::INFINITY, f64::NEG_INFINITY));
        assert_eq_m128i(r, _mm_setr_epi32(i32::MIN, i32::MIN, 0, 0));

        let r = _mm_cvtpd_epi32(_mm_setr_pd(f64::NAN, f64::NAN));
        assert_eq_m128i(r, _mm_setr_epi32(i32::MIN, i32::MIN, 0, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtsd_si32() {
        let r = _mm_cvtsd_si32(_mm_setr_pd(-2.0, 5.0));
        assert_eq!(r, -2);

        let r = _mm_cvtsd_si32(_mm_setr_pd(f64::MAX, f64::MIN));
        assert_eq!(r, i32::MIN);

        let r = _mm_cvtsd_si32(_mm_setr_pd(f64::NAN, f64::NAN));
        assert_eq!(r, i32::MIN);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtsd_ss() {
        let a = _mm_setr_ps(-1.1, -2.2, 3.3, 4.4);
        let b = _mm_setr_pd(2.0, -5.0);

        let r = _mm_cvtsd_ss(a, b);

        assert_eq_m128(r, _mm_setr_ps(2.0, -2.2, 3.3, 4.4));

        let a = _mm_setr_ps(
            -1.1,
            f32::NEG_INFINITY,
            f32::MAX,
            f32::NEG_INFINITY,
        );
        let b = _mm_setr_pd(f64::INFINITY, -5.0);

        let r = _mm_cvtsd_ss(a, b);

        assert_eq_m128(
            r,
            _mm_setr_ps(
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::MAX,
                f32::NEG_INFINITY,
            ),
        );
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtsd_f64() {
        let r = _mm_cvtsd_f64(_mm_setr_pd(-1.1, 2.2));
        assert_eq!(r, -1.1);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvtss_sd() {
        let a = _mm_setr_pd(-1.1, 2.2);
        let b = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);

        let r = _mm_cvtss_sd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(1.0, 2.2));

        let a = _mm_setr_pd(-1.1, f64::INFINITY);
        let b = _mm_setr_ps(f32::NEG_INFINITY, 2.0, 3.0, 4.0);

        let r = _mm_cvtss_sd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(f64::NEG_INFINITY, f64::INFINITY));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvttpd_epi32() {
        let a = _mm_setr_pd(-1.1, 2.2);
        let r = _mm_cvttpd_epi32(a);
        assert_eq_m128i(r, _mm_setr_epi32(-1, 2, 0, 0));

        let a = _mm_setr_pd(f64::NEG_INFINITY, f64::NAN);
        let r = _mm_cvttpd_epi32(a);
        assert_eq_m128i(r, _mm_setr_epi32(i32::MIN, i32::MIN, 0, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvttsd_si32() {
        let a = _mm_setr_pd(-1.1, 2.2);
        let r = _mm_cvttsd_si32(a);
        assert_eq!(r, -1);

        let a = _mm_setr_pd(f64::NEG_INFINITY, f64::NAN);
        let r = _mm_cvttsd_si32(a);
        assert_eq!(r, i32::MIN);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_cvttps_epi32() {
        let a = _mm_setr_ps(-1.1, 2.2, -3.3, 6.6);
        let r = _mm_cvttps_epi32(a);
        assert_eq_m128i(r, _mm_setr_epi32(-1, 2, -3, 6));

        let a = _mm_setr_ps(
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::MIN,
            f32::MAX,
        );
        let r = _mm_cvttps_epi32(a);
        assert_eq_m128i(
            r,
            _mm_setr_epi32(i32::MIN, i32::MIN, i32::MIN, i32::MIN),
        );
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_set_sd() {
        let r = _mm_set_sd(-1.0_f64);
        assert_eq_m128d(r, _mm_setr_pd(-1.0_f64, 0_f64));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_set1_pd() {
        let r = _mm_set1_pd(-1.0_f64);
        assert_eq_m128d(r, _mm_setr_pd(-1.0_f64, -1.0_f64));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_set_pd1() {
        let r = _mm_set_pd1(-2.0_f64);
        assert_eq_m128d(r, _mm_setr_pd(-2.0_f64, -2.0_f64));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_set_pd() {
        let r = _mm_set_pd(1.0_f64, 5.0_f64);
        assert_eq_m128d(r, _mm_setr_pd(5.0_f64, 1.0_f64));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_setr_pd() {
        let r = _mm_setr_pd(1.0_f64, -5.0_f64);
        assert_eq_m128d(r, _mm_setr_pd(1.0_f64, -5.0_f64));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_setzero_pd() {
        let r = _mm_setzero_pd();
        assert_eq_m128d(r, _mm_setr_pd(0_f64, 0_f64));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_load1_pd() {
        let d = -5.0;
        let r = _mm_load1_pd(&d);
        assert_eq_m128d(r, _mm_setr_pd(d, d));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_load_pd1() {
        let d = -5.0;
        let r = _mm_load_pd1(&d);
        assert_eq_m128d(r, _mm_setr_pd(d, d));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_unpackhi_pd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(3.0, 4.0);
        let r = _mm_unpackhi_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(2.0, 4.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_unpacklo_pd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(3.0, 4.0);
        let r = _mm_unpacklo_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(1.0, 3.0));
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_shuffle_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(3., 4.);
        let expected = _mm_setr_pd(1., 3.);
        let r = _mm_shuffle_pd(a, b, 0);
        assert_eq_m128d(r, expected);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_move_sd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(3., 4.);
        let expected = _mm_setr_pd(3., 2.);
        let r = _mm_move_sd(a, b);
        assert_eq_m128d(r, expected);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_castpd_ps() {
        let a = _mm_set1_pd(0.);
        let expected = _mm_set1_ps(0.);
        let r = _mm_castpd_ps(a);
        assert_eq_m128(r, expected);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_castpd_si128() {
        let a = _mm_set1_pd(0.);
        let expected = _mm_set1_epi64x(0);
        let r = _mm_castpd_si128(a);
        assert_eq_m128i(r, expected);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_castps_pd() {
        let a = _mm_set1_ps(0.);
        let expected = _mm_set1_pd(0.);
        let r = _mm_castps_pd(a);
        assert_eq_m128d(r, expected);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_castps_si128() {
        let a = _mm_set1_ps(0.);
        let expected = _mm_set1_epi32(0);
        let r = _mm_castps_si128(a);
        assert_eq_m128i(r, expected);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_castsi128_pd() {
        let a = _mm_set1_epi64x(0);
        let expected = _mm_set1_pd(0.);
        let r = _mm_castsi128_pd(a);
        assert_eq_m128d(r, expected);
    }

    #[simd_test = "sse2"]
    unsafe fn test_mm_castsi128_ps() {
        let a = _mm_set1_epi32(0);
        let expected = _mm_set1_ps(0.);
        let r = _mm_castsi128_ps(a);
        assert_eq_m128(r, expected);
    }

    #[simd_test = "sse2,mmx"]
    unsafe fn test_mm_add_si64() {
        let a = 1i64;
        let b = 2i64;
        let expected = 3i64;
        let r = _mm_add_si64(mem::transmute(a), mem::transmute(b));
        assert_eq!(mem::transmute::<__m64, i64>(r), expected);
    }

    #[simd_test = "sse2,mmx"]
    unsafe fn test_mm_mul_su32() {
        let a = _mm_setr_pi32(1, 2);
        let b = _mm_setr_pi32(3, 4);
        let expected = 3u64;
        let r = _mm_mul_su32(a, b);
        assert_eq_m64(r, mem::transmute(expected));
    }

    #[simd_test = "sse2,mmx"]
    unsafe fn test_mm_sub_si64() {
        let a = 1i64;
        let b = 2i64;
        let expected = -1i64;
        let r = _mm_sub_si64(mem::transmute(a), mem::transmute(b));
        assert_eq!(mem::transmute::<__m64, i64>(r), expected);
    }

    #[simd_test = "sse2,mmx"]
    unsafe fn test_mm_cvtpi32_pd() {
        let a = _mm_setr_pi32(1, 2);
        let expected = _mm_setr_pd(1., 2.);
        let r = _mm_cvtpi32_pd(a);
        assert_eq_m128d(r, expected);
    }

    #[simd_test = "sse2,mmx"]
    unsafe fn test_mm_set_epi64() {
        let r = _mm_set_epi64(mem::transmute(1i64), mem::transmute(2i64));
        assert_eq_m128i(r, _mm_setr_epi64x(2, 1));
    }

    #[simd_test = "sse2,mmx"]
    unsafe fn test_mm_set1_epi64() {
        let r = _mm_set1_epi64(mem::transmute(1i64));
        assert_eq_m128i(r, _mm_setr_epi64x(1, 1));
    }

    #[simd_test = "sse2,mmx"]
    unsafe fn test_mm_setr_epi64() {
        let r = _mm_setr_epi64(mem::transmute(1i64), mem::transmute(2i64));
        assert_eq_m128i(r, _mm_setr_epi64x(1, 2));
    }

    #[simd_test = "sse2,mmx"]
    unsafe fn test_mm_movepi64_pi64() {
        let r = _mm_movepi64_pi64(_mm_setr_epi64x(5, 0));
        assert_eq_m64(r, _mm_setr_pi8(5, 0, 0, 0, 0, 0, 0, 0));
    }

    #[simd_test = "sse2,mmx"]
    unsafe fn test_mm_movpi64_epi64() {
        let r = _mm_movpi64_epi64(_mm_setr_pi8(5, 0, 0, 0, 0, 0, 0, 0));
        assert_eq_m128i(r, _mm_setr_epi64x(5, 0));
    }

    #[simd_test = "sse2,mmx"]
    unsafe fn test_mm_cvtpd_pi32() {
        let a = _mm_setr_pd(5., 0.);
        let r = _mm_cvtpd_pi32(a);
        assert_eq_m64(r, _mm_setr_pi32(5, 0));
    }

    #[simd_test = "sse2,mmx"]
    unsafe fn test_mm_cvttpd_pi32() {
        use std::{f64, i32};

        let a = _mm_setr_pd(5., 0.);
        let r = _mm_cvttpd_pi32(a);
        assert_eq_m64(r, _mm_setr_pi32(5, 0));

        let a = _mm_setr_pd(f64::NEG_INFINITY, f64::NAN);
        let r = _mm_cvttpd_pi32(a);
        assert_eq_m64(r, _mm_setr_pi32(i32::MIN, i32::MIN));
    }
}
