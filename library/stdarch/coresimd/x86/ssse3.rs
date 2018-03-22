//! Supplemental Streaming SIMD Extensions 3 (SSSE3)

use coresimd::simd_llvm::simd_shuffle16;
use coresimd::simd::*;
use coresimd::x86::*;
use mem;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Compute the absolute value of packed 8-bit signed integers in `a` and
/// return the unsigned results.
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pabsb))]
pub unsafe fn _mm_abs_epi8(a: __m128i) -> __m128i {
    mem::transmute(pabsb128(a.as_i8x16()))
}

/// Compute the absolute value of each of the packed 16-bit signed integers in
/// `a` and
/// return the 16-bit unsigned integer
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pabsw))]
pub unsafe fn _mm_abs_epi16(a: __m128i) -> __m128i {
    mem::transmute(pabsw128(a.as_i16x8()))
}

/// Compute the absolute value of each of the packed 32-bit signed integers in
/// `a` and
/// return the 32-bit unsigned integer
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pabsd))]
pub unsafe fn _mm_abs_epi32(a: __m128i) -> __m128i {
    mem::transmute(pabsd128(a.as_i32x4()))
}

/// Shuffle bytes from `a` according to the content of `b`.
///
/// The last 4 bits of each byte of `b` are used as addresses
/// into the 16 bytes of `a`.
///
/// In addition, if the highest significant bit of a byte of `b`
/// is set, the respective destination byte is set to 0.
///
/// Picturing `a` and `b` as `[u8; 16]`, `_mm_shuffle_epi8` is
/// logically equivalent to:
///
/// ```
/// fn mm_shuffle_epi8(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
///     let mut r = [0u8; 16];
///     for i in 0..16 {
///         // if the most significant bit of b is set,
///         // then the destination byte is set to 0.
///         if b[i] & 0x80 == 0u8 {
///             r[i] = a[(b[i] % 16) as usize];
///         }
///     }
///     r
/// }
/// ```
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pshufb))]
pub unsafe fn _mm_shuffle_epi8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pshufb128(a.as_u8x16(), b.as_u8x16()))
}

/// Concatenate 16-byte blocks in `a` and `b` into a 32-byte temporary result,
/// shift the result right by `n` bytes, and return the low 16 bytes.
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(palignr, n = 15))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_alignr_epi8(a: __m128i, b: __m128i, n: i32) -> __m128i {
    let n = n as u32;
    // If palignr is shifting the pair of vectors more than the size of two
    // lanes, emit zero.
    if n > 32 {
        return _mm_set1_epi8(0);
    }
    // If palignr is shifting the pair of input vectors more than one lane,
    // but less than two lanes, convert to shifting in zeroes.
    let (a, b, n) = if n > 16 {
        (_mm_set1_epi8(0), a, n - 16)
    } else {
        (a, b, n)
    };
    let a = a.as_i8x16();
    let b = b.as_i8x16();

    macro_rules! shuffle {
        ($shift: expr) => {
            simd_shuffle16(
                b,
                a,
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
    let r: i8x16 = match n {
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
    mem::transmute(r)
}

/// Horizontally add the adjacent pairs of values contained in 2 packed
/// 128-bit vectors of [8 x i16].
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phaddw))]
pub unsafe fn _mm_hadd_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(phaddw128(a.as_i16x8(), b.as_i16x8()))
}

/// Horizontally add the adjacent pairs of values contained in 2 packed
/// 128-bit vectors of [8 x i16]. Positive sums greater than 7FFFh are
/// saturated to 7FFFh. Negative sums less than 8000h are saturated to 8000h.
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phaddsw))]
pub unsafe fn _mm_hadds_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(phaddsw128(a.as_i16x8(), b.as_i16x8()))
}

/// Horizontally add the adjacent pairs of values contained in 2 packed
/// 128-bit vectors of [4 x i32].
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phaddd))]
pub unsafe fn _mm_hadd_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(phaddd128(a.as_i32x4(), b.as_i32x4()))
}

/// Horizontally subtract the adjacent pairs of values contained in 2
/// packed 128-bit vectors of [8 x i16].
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phsubw))]
pub unsafe fn _mm_hsub_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(phsubw128(a.as_i16x8(), b.as_i16x8()))
}

/// Horizontally subtract the adjacent pairs of values contained in 2
/// packed 128-bit vectors of [8 x i16]. Positive differences greater than
/// 7FFFh are saturated to 7FFFh. Negative differences less than 8000h are
/// saturated to 8000h.
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phsubsw))]
pub unsafe fn _mm_hsubs_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(phsubsw128(a.as_i16x8(), b.as_i16x8()))
}

/// Horizontally subtract the adjacent pairs of values contained in 2
/// packed 128-bit vectors of [4 x i32].
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phsubd))]
pub unsafe fn _mm_hsub_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(phsubd128(a.as_i32x4(), b.as_i32x4()))
}

/// Multiply corresponding pairs of packed 8-bit unsigned integer
/// values contained in the first source operand and packed 8-bit signed
/// integer values contained in the second source operand, add pairs of
/// contiguous products with signed saturation, and writes the 16-bit sums to
/// the corresponding bits in the destination.
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pmaddubsw))]
pub unsafe fn _mm_maddubs_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pmaddubsw128(a.as_u8x16(), b.as_i8x16()))
}

/// Multiply packed 16-bit signed integer values, truncate the 32-bit
/// product to the 18 most significant bits by right-shifting, round the
/// truncated value by adding 1, and write bits [16:1] to the destination.
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pmulhrsw))]
pub unsafe fn _mm_mulhrs_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pmulhrsw128(a.as_i16x8(), b.as_i16x8()))
}

/// Negate packed 8-bit integers in `a` when the corresponding signed 8-bit
/// integer in `b` is negative, and return the result.
/// Elements in result are zeroed out when the corresponding element in `b`
/// is zero.
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(psignb))]
pub unsafe fn _mm_sign_epi8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(psignb128(a.as_i8x16(), b.as_i8x16()))
}

/// Negate packed 16-bit integers in `a` when the corresponding signed 16-bit
/// integer in `b` is negative, and return the results.
/// Elements in result are zeroed out when the corresponding element in `b`
/// is zero.
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(psignw))]
pub unsafe fn _mm_sign_epi16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(psignw128(a.as_i16x8(), b.as_i16x8()))
}

/// Negate packed 32-bit integers in `a` when the corresponding signed 32-bit
/// integer in `b` is negative, and return the results.
/// Element in result are zeroed out when the corresponding element in `b`
/// is zero.
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(psignd))]
pub unsafe fn _mm_sign_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(psignd128(a.as_i32x4(), b.as_i32x4()))
}

/// Compute the absolute value of packed 8-bit integers in `a` and
/// return the unsigned results.
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(pabsb))]
pub unsafe fn _mm_abs_pi8(a: __m64) -> __m64 {
    pabsb(a)
}

/// Compute the absolute value of packed 8-bit integers in `a`, and return the
/// unsigned results.
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(pabsw))]
pub unsafe fn _mm_abs_pi16(a: __m64) -> __m64 {
    pabsw(a)
}

/// Compute the absolute value of packed 32-bit integers in `a`, and return the
/// unsigned results.
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(pabsd))]
pub unsafe fn _mm_abs_pi32(a: __m64) -> __m64 {
    pabsd(a)
}

/// Shuffle packed 8-bit integers in `a` according to shuffle control mask in
/// the corresponding 8-bit element of `b`, and return the results
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(pshufb))]
pub unsafe fn _mm_shuffle_pi8(a: __m64, b: __m64) -> __m64 {
    pshufb(a, b)
}

/// Concatenates the two 64-bit integer vector operands, and right-shifts
/// the result by the number of bytes specified in the immediate operand.
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(palignr, n = 15))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_alignr_pi8(a: __m64, b: __m64, n: i32) -> __m64 {
    macro_rules! call {
        ($imm8: expr) => {
            palignrb(a, b, $imm8)
        };
    }
    constify_imm8!(n, call)
}

/// Horizontally add the adjacent pairs of values contained in 2 packed
/// 64-bit vectors of [4 x i16].
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(phaddw))]
pub unsafe fn _mm_hadd_pi16(a: __m64, b: __m64) -> __m64 {
    phaddw(a, b)
}

/// Horizontally add the adjacent pairs of values contained in 2 packed
/// 64-bit vectors of [2 x i32].
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(phaddd))]
pub unsafe fn _mm_hadd_pi32(a: __m64, b: __m64) -> __m64 {
    phaddd(a, b)
}

/// Horizontally add the adjacent pairs of values contained in 2 packed
/// 64-bit vectors of [4 x i16]. Positive sums greater than 7FFFh are
/// saturated to 7FFFh. Negative sums less than 8000h are saturated to 8000h.
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(phaddsw))]
pub unsafe fn _mm_hadds_pi16(a: __m64, b: __m64) -> __m64 {
    phaddsw(a, b)
}

/// Horizontally subtracts the adjacent pairs of values contained in 2
/// packed 64-bit vectors of [4 x i16].
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(phsubw))]
pub unsafe fn _mm_hsub_pi16(a: __m64, b: __m64) -> __m64 {
    phsubw(a, b)
}

/// Horizontally subtracts the adjacent pairs of values contained in 2
/// packed 64-bit vectors of [2 x i32].
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(phsubd))]
pub unsafe fn _mm_hsub_pi32(a: __m64, b: __m64) -> __m64 {
    phsubd(a, b)
}

/// Horizontally subtracts the adjacent pairs of values contained in 2
/// packed 64-bit vectors of [4 x i16]. Positive differences greater than
/// 7FFFh are saturated to 7FFFh. Negative differences less than 8000h are
/// saturated to 8000h.
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(phsubsw))]
pub unsafe fn _mm_hsubs_pi16(a: __m64, b: __m64) -> __m64 {
    phsubsw(a, b)
}

/// Multiplies corresponding pairs of packed 8-bit unsigned integer
/// values contained in the first source operand and packed 8-bit signed
/// integer values contained in the second source operand, adds pairs of
/// contiguous products with signed saturation, and writes the 16-bit sums to
/// the corresponding bits in the destination.
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(pmaddubsw))]
pub unsafe fn _mm_maddubs_pi16(a: __m64, b: __m64) -> __m64 {
    pmaddubsw(a, b)
}

/// Multiplies packed 16-bit signed integer values, truncates the 32-bit
/// products to the 18 most significant bits by right-shifting, rounds the
/// truncated value by adding 1, and writes bits [16:1] to the destination.
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(pmulhrsw))]
pub unsafe fn _mm_mulhrs_pi16(a: __m64, b: __m64) -> __m64 {
    pmulhrsw(a, b)
}

/// Negate packed 8-bit integers in `a` when the corresponding signed 8-bit
/// integer in `b` is negative, and return the results.
/// Element in result are zeroed out when the corresponding element in `b` is
/// zero.
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(psignb))]
pub unsafe fn _mm_sign_pi8(a: __m64, b: __m64) -> __m64 {
    psignb(a, b)
}

/// Negate packed 16-bit integers in `a` when the corresponding signed 16-bit
/// integer in `b` is negative, and return the results.
/// Element in result are zeroed out when the corresponding element in `b` is
/// zero.
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(psignw))]
pub unsafe fn _mm_sign_pi16(a: __m64, b: __m64) -> __m64 {
    psignw(a, b)
}

/// Negate packed 32-bit integers in `a` when the corresponding signed 32-bit
/// integer in `b` is negative, and return the results.
/// Element in result are zeroed out when the corresponding element in `b` is
/// zero.
#[inline]
#[target_feature(enable = "ssse3,mmx")]
#[cfg_attr(test, assert_instr(psignd))]
pub unsafe fn _mm_sign_pi32(a: __m64, b: __m64) -> __m64 {
    psignd(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.ssse3.pabs.b.128"]
    fn pabsb128(a: i8x16) -> u8x16;

    #[link_name = "llvm.x86.ssse3.pabs.w.128"]
    fn pabsw128(a: i16x8) -> u16x8;

    #[link_name = "llvm.x86.ssse3.pabs.d.128"]
    fn pabsd128(a: i32x4) -> u32x4;

    #[link_name = "llvm.x86.ssse3.pshuf.b.128"]
    fn pshufb128(a: u8x16, b: u8x16) -> u8x16;

    #[link_name = "llvm.x86.ssse3.phadd.w.128"]
    fn phaddw128(a: i16x8, b: i16x8) -> i16x8;

    #[link_name = "llvm.x86.ssse3.phadd.sw.128"]
    fn phaddsw128(a: i16x8, b: i16x8) -> i16x8;

    #[link_name = "llvm.x86.ssse3.phadd.d.128"]
    fn phaddd128(a: i32x4, b: i32x4) -> i32x4;

    #[link_name = "llvm.x86.ssse3.phsub.w.128"]
    fn phsubw128(a: i16x8, b: i16x8) -> i16x8;

    #[link_name = "llvm.x86.ssse3.phsub.sw.128"]
    fn phsubsw128(a: i16x8, b: i16x8) -> i16x8;

    #[link_name = "llvm.x86.ssse3.phsub.d.128"]
    fn phsubd128(a: i32x4, b: i32x4) -> i32x4;

    #[link_name = "llvm.x86.ssse3.pmadd.ub.sw.128"]
    fn pmaddubsw128(a: u8x16, b: i8x16) -> i16x8;

    #[link_name = "llvm.x86.ssse3.pmul.hr.sw.128"]
    fn pmulhrsw128(a: i16x8, b: i16x8) -> i16x8;

    #[link_name = "llvm.x86.ssse3.psign.b.128"]
    fn psignb128(a: i8x16, b: i8x16) -> i8x16;

    #[link_name = "llvm.x86.ssse3.psign.w.128"]
    fn psignw128(a: i16x8, b: i16x8) -> i16x8;

    #[link_name = "llvm.x86.ssse3.psign.d.128"]
    fn psignd128(a: i32x4, b: i32x4) -> i32x4;

    #[link_name = "llvm.x86.ssse3.pabs.b"]
    fn pabsb(a: __m64) -> __m64;

    #[link_name = "llvm.x86.ssse3.pabs.w"]
    fn pabsw(a: __m64) -> __m64;

    #[link_name = "llvm.x86.ssse3.pabs.d"]
    fn pabsd(a: __m64) -> __m64;

    #[link_name = "llvm.x86.ssse3.pshuf.b"]
    fn pshufb(a: __m64, b: __m64) -> __m64;

    #[link_name = "llvm.x86.mmx.palignr.b"]
    fn palignrb(a: __m64, b: __m64, n: u8) -> __m64;

    #[link_name = "llvm.x86.ssse3.phadd.w"]
    fn phaddw(a: __m64, b: __m64) -> __m64;

    #[link_name = "llvm.x86.ssse3.phadd.d"]
    fn phaddd(a: __m64, b: __m64) -> __m64;

    #[link_name = "llvm.x86.ssse3.phadd.sw"]
    fn phaddsw(a: __m64, b: __m64) -> __m64;

    #[link_name = "llvm.x86.ssse3.phsub.w"]
    fn phsubw(a: __m64, b: __m64) -> __m64;

    #[link_name = "llvm.x86.ssse3.phsub.d"]
    fn phsubd(a: __m64, b: __m64) -> __m64;

    #[link_name = "llvm.x86.ssse3.phsub.sw"]
    fn phsubsw(a: __m64, b: __m64) -> __m64;

    #[link_name = "llvm.x86.ssse3.pmadd.ub.sw"]
    fn pmaddubsw(a: __m64, b: __m64) -> __m64;

    #[link_name = "llvm.x86.ssse3.pmul.hr.sw"]
    fn pmulhrsw(a: __m64, b: __m64) -> __m64;

    #[link_name = "llvm.x86.ssse3.psign.b"]
    fn psignb(a: __m64, b: __m64) -> __m64;

    #[link_name = "llvm.x86.ssse3.psign.w"]
    fn psignw(a: __m64, b: __m64) -> __m64;

    #[link_name = "llvm.x86.ssse3.psign.d"]
    fn psignd(a: __m64, b: __m64) -> __m64;
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use coresimd::x86::*;

    #[simd_test = "ssse3"]
    unsafe fn test_mm_abs_epi8() {
        let r = _mm_abs_epi8(_mm_set1_epi8(-5));
        assert_eq_m128i(r, _mm_set1_epi8(5));
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_abs_epi16() {
        let r = _mm_abs_epi16(_mm_set1_epi16(-5));
        assert_eq_m128i(r, _mm_set1_epi16(5));
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_abs_epi32() {
        let r = _mm_abs_epi32(_mm_set1_epi32(-5));
        assert_eq_m128i(r, _mm_set1_epi32(5));
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_shuffle_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm_setr_epi8(
            4, 128_u8 as i8, 4, 3,
            24, 12, 6, 19,
            12, 5, 5, 10,
            4, 1, 8, 0,
        );
        let expected = _mm_setr_epi8(
            5,
            0,
            5,
            4,
            9,
            13,
            7,
            4,
            13,
            6,
            6,
            11,
            5,
            2,
            9,
            1,
        );
        let r = _mm_shuffle_epi8(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_alignr_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm_setr_epi8(
            4, 63, 4, 3,
            24, 12, 6, 19,
            12, 5, 5, 10,
            4, 1, 8, 0,
        );
        let r = _mm_alignr_epi8(a, b, 33);
        assert_eq_m128i(r, _mm_set1_epi8(0));

        let r = _mm_alignr_epi8(a, b, 17);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = _mm_setr_epi8(
            2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 0,
        );
        assert_eq_m128i(r, expected);

        let r = _mm_alignr_epi8(a, b, 16);
        assert_eq_m128i(r, a);

        let r = _mm_alignr_epi8(a, b, 15);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        assert_eq_m128i(r, expected);

        let r = _mm_alignr_epi8(a, b, 0);
        assert_eq_m128i(r, b);
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_hadd_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 24, 12, 6, 19);
        let expected = _mm_setr_epi16(3, 7, 11, 15, 132, 7, 36, 25);
        let r = _mm_hadd_epi16(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_hadds_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 32767, 1, -32768, -1);
        let expected = _mm_setr_epi16(3, 7, 11, 15, 132, 7, 32767, -32768);
        let r = _mm_hadds_epi16(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_hadd_epi32() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let b = _mm_setr_epi32(4, 128, 4, 3);
        let expected = _mm_setr_epi32(3, 7, 132, 7);
        let r = _mm_hadd_epi32(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_hsub_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 24, 12, 6, 19);
        let expected = _mm_setr_epi16(-1, -1, -1, -1, -124, 1, 12, -13);
        let r = _mm_hsub_epi16(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_hsubs_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 32767, -1, -32768, 1);
        let expected = _mm_setr_epi16(-1, -1, -1, -1, -124, 1, 32767, -32768);
        let r = _mm_hsubs_epi16(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_hsub_epi32() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let b = _mm_setr_epi32(4, 128, 4, 3);
        let expected = _mm_setr_epi32(-1, -1, -124, 1);
        let r = _mm_hsub_epi32(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_maddubs_epi16() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm_setr_epi8(
            4, 63, 4, 3,
            24, 12, 6, 19,
            12, 5, 5, 10,
            4, 1, 8, 0,
        );
        let expected = _mm_setr_epi16(130, 24, 192, 194, 158, 175, 66, 120);
        let r = _mm_maddubs_epi16(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_mulhrs_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 32767, -1, -32768, 1);
        let expected = _mm_setr_epi16(0, 0, 0, 0, 5, 0, -7, 0);
        let r = _mm_mulhrs_epi16(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_sign_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, -14, -15, 16,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm_setr_epi8(
            4, 63, -4, 3, 24, 12, -6, -19,
            12, 5, -5, 10, 4, 1, -8, 0,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = _mm_setr_epi8(
            1, 2, -3, 4, 5, 6, -7, -8,
            9, 10, -11, 12, 13, -14, 15, 0,
        );
        let r = _mm_sign_epi8(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_sign_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, -5, -6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 0, 3, 1, -1, -2, 1);
        let expected = _mm_setr_epi16(1, 2, 0, 4, -5, 6, -7, 8);
        let r = _mm_sign_epi16(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn test_mm_sign_epi32() {
        let a = _mm_setr_epi32(-1, 2, 3, 4);
        let b = _mm_setr_epi32(1, -1, 1, 0);
        let expected = _mm_setr_epi32(-1, -2, 3, 0);
        let r = _mm_sign_epi32(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_abs_pi8() {
        let r = _mm_abs_pi8(_mm_set1_pi8(-5));
        assert_eq_m64(r, _mm_set1_pi8(5));
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_abs_pi16() {
        let r = _mm_abs_pi16(_mm_set1_pi16(-5));
        assert_eq_m64(r, _mm_set1_pi16(5));
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_abs_pi32() {
        let r = _mm_abs_pi32(_mm_set1_pi32(-5));
        assert_eq_m64(r, _mm_set1_pi32(5));
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_shuffle_pi8() {
        let a = _mm_setr_pi8(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_pi8(4, 128u8 as i8, 4, 3, 24, 12, 6, 19);
        let expected = _mm_setr_pi8(5, 0, 5, 4, 1, 5, 7, 4);
        let r = _mm_shuffle_pi8(a, b);
        assert_eq_m64(r, expected);
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_alignr_pi8() {
        let a = _mm_setr_pi32(0x89ABCDEF_u32 as i32, 0x01234567_u32 as i32);
        let b = _mm_setr_pi32(0xBBAA9988_u32 as i32, 0xFFDDEECC_u32 as i32);
        let r = _mm_alignr_pi8(a, b, 4);
        assert_eq_m64(r, ::std::mem::transmute(0x89abcdefffddeecc_u64));
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_hadd_pi16() {
        let a = _mm_setr_pi16(1, 2, 3, 4);
        let b = _mm_setr_pi16(4, 128, 4, 3);
        let expected = _mm_setr_pi16(3, 7, 132, 7);
        let r = _mm_hadd_pi16(a, b);
        assert_eq_m64(r, expected);
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_hadd_pi32() {
        let a = _mm_setr_pi32(1, 2);
        let b = _mm_setr_pi32(4, 128);
        let expected = _mm_setr_pi32(3, 132);
        let r = _mm_hadd_pi32(a, b);
        assert_eq_m64(r, expected);
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_hadds_pi16() {
        let a = _mm_setr_pi16(1, 2, 3, 4);
        let b = _mm_setr_pi16(32767, 1, -32768, -1);
        let expected = _mm_setr_pi16(3, 7, 32767, -32768);
        let r = _mm_hadds_pi16(a, b);
        assert_eq_m64(r, expected);
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_hsub_pi16() {
        let a = _mm_setr_pi16(1, 2, 3, 4);
        let b = _mm_setr_pi16(4, 128, 4, 3);
        let expected = _mm_setr_pi16(-1, -1, -124, 1);
        let r = _mm_hsub_pi16(a, b);
        assert_eq_m64(r, expected);
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_hsub_pi32() {
        let a = _mm_setr_pi32(1, 2);
        let b = _mm_setr_pi32(4, 128);
        let expected = _mm_setr_pi32(-1, -124);
        let r = _mm_hsub_pi32(a, b);
        assert_eq_m64(r, expected);
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_hsubs_pi16() {
        let a = _mm_setr_pi16(1, 2, 3, 4);
        let b = _mm_setr_pi16(4, 128, 4, 3);
        let expected = _mm_setr_pi16(-1, -1, -124, 1);
        let r = _mm_hsubs_pi16(a, b);
        assert_eq_m64(r, expected);
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_maddubs_pi16() {
        let a = _mm_setr_pi8(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_pi8(4, 63, 4, 3, 24, 12, 6, 19);
        let expected = _mm_setr_pi16(130, 24, 192, 194);
        let r = _mm_maddubs_pi16(a, b);
        assert_eq_m64(r, expected);
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_mulhrs_pi16() {
        let a = _mm_setr_pi16(1, 2, 3, 4);
        let b = _mm_setr_pi16(4, 32767, -1, -32768);
        let expected = _mm_setr_pi16(0, 2, 0, -4);
        let r = _mm_mulhrs_pi16(a, b);
        assert_eq_m64(r, expected);
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_sign_pi8() {
        let a = _mm_setr_pi8(1, 2, 3, 4, -5, -6, 7, 8);
        let b = _mm_setr_pi8(4, 64, 0, 3, 1, -1, -2, 1);
        let expected = _mm_setr_pi8(1, 2, 0, 4, -5, 6, -7, 8);
        let r = _mm_sign_pi8(a, b);
        assert_eq_m64(r, expected);
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_sign_pi16() {
        let a = _mm_setr_pi16(-1, 2, 3, 4);
        let b = _mm_setr_pi16(1, -1, 1, 0);
        let expected = _mm_setr_pi16(-1, -2, 3, 0);
        let r = _mm_sign_pi16(a, b);
        assert_eq_m64(r, expected);
    }

    #[simd_test = "ssse3,mmx"]
    unsafe fn test_mm_sign_pi32() {
        let a = _mm_setr_pi32(-1, 2);
        let b = _mm_setr_pi32(1, 0);
        let expected = _mm_setr_pi32(-1, 0);
        let r = _mm_sign_pi32(a, b);
        assert_eq_m64(r, expected);
    }
}
