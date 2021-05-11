//! Supplemental Streaming SIMD Extensions 3 (SSSE3)

use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem::transmute,
};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Computes the absolute value of packed 8-bit signed integers in `a` and
/// return the unsigned results.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_abs_epi8)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pabsb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_abs_epi8(a: __m128i) -> __m128i {
    transmute(pabsb128(a.as_i8x16()))
}

/// Computes the absolute value of each of the packed 16-bit signed integers in
/// `a` and
/// return the 16-bit unsigned integer
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_abs_epi16)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pabsw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_abs_epi16(a: __m128i) -> __m128i {
    transmute(pabsw128(a.as_i16x8()))
}

/// Computes the absolute value of each of the packed 32-bit signed integers in
/// `a` and
/// return the 32-bit unsigned integer
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_abs_epi32)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pabsd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_abs_epi32(a: __m128i) -> __m128i {
    transmute(pabsd128(a.as_i32x4()))
}

/// Shuffles bytes from `a` according to the content of `b`.
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
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_shuffle_epi8)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pshufb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_shuffle_epi8(a: __m128i, b: __m128i) -> __m128i {
    transmute(pshufb128(a.as_u8x16(), b.as_u8x16()))
}

/// Concatenate 16-byte blocks in `a` and `b` into a 32-byte temporary result,
/// shift the result right by `n` bytes, and returns the low 16 bytes.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_alignr_epi8)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(palignr, IMM8 = 15))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_alignr_epi8<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    static_assert_imm8!(IMM8);
    // If palignr is shifting the pair of vectors more than the size of two
    // lanes, emit zero.
    if IMM8 > 32 {
        return _mm_set1_epi8(0);
    }
    // If palignr is shifting the pair of input vectors more than one lane,
    // but less than two lanes, convert to shifting in zeroes.
    let (a, b) = if IMM8 > 16 {
        (_mm_set1_epi8(0), a)
    } else {
        (a, b)
    };
    const fn mask(shift: u32, i: u32) -> u32 {
        if shift > 32 {
            // Unused, but needs to be a valid index.
            i
        } else if shift > 16 {
            shift - 16 + i
        } else {
            shift + i
        }
    }
    let r: i8x16 = simd_shuffle16!(
        b.as_i8x16(),
        a.as_i8x16(),
        <const IMM8: i32> [
            mask(IMM8 as u32, 0),
            mask(IMM8 as u32, 1),
            mask(IMM8 as u32, 2),
            mask(IMM8 as u32, 3),
            mask(IMM8 as u32, 4),
            mask(IMM8 as u32, 5),
            mask(IMM8 as u32, 6),
            mask(IMM8 as u32, 7),
            mask(IMM8 as u32, 8),
            mask(IMM8 as u32, 9),
            mask(IMM8 as u32, 10),
            mask(IMM8 as u32, 11),
            mask(IMM8 as u32, 12),
            mask(IMM8 as u32, 13),
            mask(IMM8 as u32, 14),
            mask(IMM8 as u32, 15),
        ],
    );
    transmute(r)
}

/// Horizontally adds the adjacent pairs of values contained in 2 packed
/// 128-bit vectors of `[8 x i16]`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_hadd_epi16)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phaddw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_hadd_epi16(a: __m128i, b: __m128i) -> __m128i {
    transmute(phaddw128(a.as_i16x8(), b.as_i16x8()))
}

/// Horizontally adds the adjacent pairs of values contained in 2 packed
/// 128-bit vectors of `[8 x i16]`. Positive sums greater than 7FFFh are
/// saturated to 7FFFh. Negative sums less than 8000h are saturated to 8000h.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_hadds_epi16)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phaddsw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_hadds_epi16(a: __m128i, b: __m128i) -> __m128i {
    transmute(phaddsw128(a.as_i16x8(), b.as_i16x8()))
}

/// Horizontally adds the adjacent pairs of values contained in 2 packed
/// 128-bit vectors of `[4 x i32]`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_hadd_epi32)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phaddd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_hadd_epi32(a: __m128i, b: __m128i) -> __m128i {
    transmute(phaddd128(a.as_i32x4(), b.as_i32x4()))
}

/// Horizontally subtract the adjacent pairs of values contained in 2
/// packed 128-bit vectors of `[8 x i16]`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_hsub_epi16)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phsubw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_hsub_epi16(a: __m128i, b: __m128i) -> __m128i {
    transmute(phsubw128(a.as_i16x8(), b.as_i16x8()))
}

/// Horizontally subtract the adjacent pairs of values contained in 2
/// packed 128-bit vectors of `[8 x i16]`. Positive differences greater than
/// 7FFFh are saturated to 7FFFh. Negative differences less than 8000h are
/// saturated to 8000h.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_hsubs_epi16)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phsubsw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_hsubs_epi16(a: __m128i, b: __m128i) -> __m128i {
    transmute(phsubsw128(a.as_i16x8(), b.as_i16x8()))
}

/// Horizontally subtract the adjacent pairs of values contained in 2
/// packed 128-bit vectors of `[4 x i32]`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_hsub_epi32)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phsubd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_hsub_epi32(a: __m128i, b: __m128i) -> __m128i {
    transmute(phsubd128(a.as_i32x4(), b.as_i32x4()))
}

/// Multiplies corresponding pairs of packed 8-bit unsigned integer
/// values contained in the first source operand and packed 8-bit signed
/// integer values contained in the second source operand, add pairs of
/// contiguous products with signed saturation, and writes the 16-bit sums to
/// the corresponding bits in the destination.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maddubs_epi16)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pmaddubsw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_maddubs_epi16(a: __m128i, b: __m128i) -> __m128i {
    transmute(pmaddubsw128(a.as_u8x16(), b.as_i8x16()))
}

/// Multiplies packed 16-bit signed integer values, truncate the 32-bit
/// product to the 18 most significant bits by right-shifting, round the
/// truncated value by adding 1, and write bits `[16:1]` to the destination.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mulhrs_epi16)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pmulhrsw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_mulhrs_epi16(a: __m128i, b: __m128i) -> __m128i {
    transmute(pmulhrsw128(a.as_i16x8(), b.as_i16x8()))
}

/// Negates packed 8-bit integers in `a` when the corresponding signed 8-bit
/// integer in `b` is negative, and returns the result.
/// Elements in result are zeroed out when the corresponding element in `b`
/// is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sign_epi8)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(psignb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_sign_epi8(a: __m128i, b: __m128i) -> __m128i {
    transmute(psignb128(a.as_i8x16(), b.as_i8x16()))
}

/// Negates packed 16-bit integers in `a` when the corresponding signed 16-bit
/// integer in `b` is negative, and returns the results.
/// Elements in result are zeroed out when the corresponding element in `b`
/// is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sign_epi16)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(psignw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_sign_epi16(a: __m128i, b: __m128i) -> __m128i {
    transmute(psignw128(a.as_i16x8(), b.as_i16x8()))
}

/// Negates packed 32-bit integers in `a` when the corresponding signed 32-bit
/// integer in `b` is negative, and returns the results.
/// Element in result are zeroed out when the corresponding element in `b`
/// is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sign_epi32)
#[inline]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(psignd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_sign_epi32(a: __m128i, b: __m128i) -> __m128i {
    transmute(psignd128(a.as_i32x4(), b.as_i32x4()))
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
}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_abs_epi8() {
        let r = _mm_abs_epi8(_mm_set1_epi8(-5));
        assert_eq_m128i(r, _mm_set1_epi8(5));
    }

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_abs_epi16() {
        let r = _mm_abs_epi16(_mm_set1_epi16(-5));
        assert_eq_m128i(r, _mm_set1_epi16(5));
    }

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_abs_epi32() {
        let r = _mm_abs_epi32(_mm_set1_epi32(-5));
        assert_eq_m128i(r, _mm_set1_epi32(5));
    }

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_shuffle_epi8() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        );
        #[rustfmt::skip]
        let b = _mm_setr_epi8(
            4, 128_u8 as i8, 4, 3,
            24, 12, 6, 19,
            12, 5, 5, 10,
            4, 1, 8, 0,
        );
        let expected = _mm_setr_epi8(5, 0, 5, 4, 9, 13, 7, 4, 13, 6, 6, 11, 5, 2, 9, 1);
        let r = _mm_shuffle_epi8(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_alignr_epi8() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        );
        #[rustfmt::skip]
        let b = _mm_setr_epi8(
            4, 63, 4, 3,
            24, 12, 6, 19,
            12, 5, 5, 10,
            4, 1, 8, 0,
        );
        let r = _mm_alignr_epi8::<33>(a, b);
        assert_eq_m128i(r, _mm_set1_epi8(0));

        let r = _mm_alignr_epi8::<17>(a, b);
        #[rustfmt::skip]
        let expected = _mm_setr_epi8(
            2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 0,
        );
        assert_eq_m128i(r, expected);

        let r = _mm_alignr_epi8::<16>(a, b);
        assert_eq_m128i(r, a);

        let r = _mm_alignr_epi8::<15>(a, b);
        #[rustfmt::skip]
        let expected = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        assert_eq_m128i(r, expected);

        let r = _mm_alignr_epi8::<0>(a, b);
        assert_eq_m128i(r, b);
    }

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_hadd_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 24, 12, 6, 19);
        let expected = _mm_setr_epi16(3, 7, 11, 15, 132, 7, 36, 25);
        let r = _mm_hadd_epi16(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_hadds_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 32767, 1, -32768, -1);
        let expected = _mm_setr_epi16(3, 7, 11, 15, 132, 7, 32767, -32768);
        let r = _mm_hadds_epi16(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_hadd_epi32() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let b = _mm_setr_epi32(4, 128, 4, 3);
        let expected = _mm_setr_epi32(3, 7, 132, 7);
        let r = _mm_hadd_epi32(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_hsub_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 24, 12, 6, 19);
        let expected = _mm_setr_epi16(-1, -1, -1, -1, -124, 1, 12, -13);
        let r = _mm_hsub_epi16(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_hsubs_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 32767, -1, -32768, 1);
        let expected = _mm_setr_epi16(-1, -1, -1, -1, -124, 1, 32767, -32768);
        let r = _mm_hsubs_epi16(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_hsub_epi32() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let b = _mm_setr_epi32(4, 128, 4, 3);
        let expected = _mm_setr_epi32(-1, -1, -124, 1);
        let r = _mm_hsub_epi32(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_maddubs_epi16() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        );
        #[rustfmt::skip]
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

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_mulhrs_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 32767, -1, -32768, 1);
        let expected = _mm_setr_epi16(0, 0, 0, 0, 5, 0, -7, 0);
        let r = _mm_mulhrs_epi16(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_sign_epi8() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, -14, -15, 16,
        );
        #[rustfmt::skip]
        let b = _mm_setr_epi8(
            4, 63, -4, 3, 24, 12, -6, -19,
            12, 5, -5, 10, 4, 1, -8, 0,
        );
        #[rustfmt::skip]
        let expected = _mm_setr_epi8(
            1, 2, -3, 4, 5, 6, -7, -8,
            9, 10, -11, 12, 13, -14, 15, 0,
        );
        let r = _mm_sign_epi8(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_sign_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, -5, -6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 0, 3, 1, -1, -2, 1);
        let expected = _mm_setr_epi16(1, 2, 0, 4, -5, 6, -7, 8);
        let r = _mm_sign_epi16(a, b);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "ssse3")]
    unsafe fn test_mm_sign_epi32() {
        let a = _mm_setr_epi32(-1, 2, 3, 4);
        let b = _mm_setr_epi32(1, -1, 1, 0);
        let expected = _mm_setr_epi32(-1, -2, 3, 0);
        let r = _mm_sign_epi32(a, b);
        assert_eq_m128i(r, expected);
    }
}
