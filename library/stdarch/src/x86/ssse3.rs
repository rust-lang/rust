#[cfg(test)]
use stdsimd_test::assert_instr;

use v128::*;

/// Compute the absolute value of packed 8-bit signed integers in `a` and
/// return the unsigned results.
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(pabsb))]
pub unsafe fn _mm_abs_epi8(a: i8x16) -> u8x16 {
    pabsb128(a)
}

/// Compute the absolute value of each of the packed 16-bit signed integers in `a` and
/// return the 16-bit unsigned integer
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(pabsw))]
pub unsafe fn _mm_abs_epi16(a: i16x8) -> u16x8 {
    pabsw128(a)
}

/// Compute the absolute value of each of the packed 32-bit signed integers in `a` and
/// return the 32-bit unsigned integer
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(pabsd))]
pub unsafe fn _mm_abs_epi32(a: i32x4) -> u32x4 {
    pabsd128(a)
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
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(pshufb))]
pub unsafe fn _mm_shuffle_epi8(a: u8x16, b: u8x16) -> u8x16 {
    pshufb128(a, b)
}

/// Horizontally add the adjacent pairs of values contained in 2 packed
/// 128-bit vectors of [8 x i16].
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(phaddw))]
pub unsafe fn _mm_hadd_epi16(a: i16x8, b: i16x8) -> i16x8 {
    phaddw128(a, b)
}

/// Horizontally add the adjacent pairs of values contained in 2 packed
/// 128-bit vectors of [8 x i16]. Positive sums greater than 7FFFh are
/// saturated to 7FFFh. Negative sums less than 8000h are saturated to 8000h.
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(phaddsw))]
pub unsafe fn _mm_hadds_epi16(a: i16x8, b: i16x8) -> i16x8 {
    phaddsw128(a, b)
}

/// Horizontally add the adjacent pairs of values contained in 2 packed
/// 128-bit vectors of [4 x i32].
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(phaddd))]
pub unsafe fn _mm_hadd_epi32(a: i32x4, b: i32x4) -> i32x4 {
    phaddd128(a, b)
}

/// Horizontally subtract the adjacent pairs of values contained in 2
/// packed 128-bit vectors of [8 x i16].
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(phsubw))]
pub unsafe fn _mm_hsub_epi16(a: i16x8, b: i16x8) -> i16x8 {
    phsubw128(a, b)
}

/// Horizontally subtract the adjacent pairs of values contained in 2
/// packed 128-bit vectors of [8 x i16]. Positive differences greater than
/// 7FFFh are saturated to 7FFFh. Negative differences less than 8000h are
/// saturated to 8000h.
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(phsubsw))]
pub unsafe fn _mm_hsubs_epi16(a: i16x8, b: i16x8) -> i16x8 {
    phsubsw128(a, b)
}

/// Horizontally subtract the adjacent pairs of values contained in 2
/// packed 128-bit vectors of [4 x i32].
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(phsubd))]
pub unsafe fn _mm_hsub_epi32(a: i32x4, b: i32x4) -> i32x4 {
    phsubd128(a, b)
}

/// Multiply corresponding pairs of packed 8-bit unsigned integer
/// values contained in the first source operand and packed 8-bit signed
/// integer values contained in the second source operand, add pairs of
/// contiguous products with signed saturation, and writes the 16-bit sums to
/// the corresponding bits in the destination.
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(pmaddubsw))]
pub unsafe fn _mm_maddubs_epi16(a: u8x16, b: i8x16) -> i16x8 {
    pmaddubsw128(a, b)
}

/// Multiply packed 16-bit signed integer values, truncate the 32-bit
/// product to the 18 most significant bits by right-shifting, round the
/// truncated value by adding 1, and write bits [16:1] to the destination.
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(pmulhrsw))]
pub unsafe fn _mm_mulhrs_epi16(a: i16x8, b: i16x8) -> i16x8 {
    pmulhrsw128(a, b)
}

/// Negate packed 8-bit integers in `a` when the corresponding signed 8-bit
/// integer in `b` is negative, and return the result.
/// Elements in result are zeroed out when the corresponding element in `b`
/// is zero.
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(psignb))]
pub unsafe fn _mm_sign_epi8(a: i8x16, b: i8x16) -> i8x16 {
    psignb128(a, b)
}

/// Negate packed 16-bit integers in `a` when the corresponding signed 16-bit
/// integer in `b` is negative, and return the results.
/// Elements in result are zeroed out when the corresponding element in `b`
/// is zero.
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(psignw))]
pub unsafe fn _mm_sign_epi16(a: i16x8, b: i16x8) -> i16x8 {
    psignw128(a, b)
}

/// Negate packed 32-bit integers in `a` when the corresponding signed 32-bit
/// integer in `b` is negative, and return the results.
/// Element in result are zeroed out when the corresponding element in `b`
/// is zero.
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(psignd))]
pub unsafe fn _mm_sign_epi32(a: i32x4, b: i32x4) -> i32x4 {
    psignd128(a, b)
}

#[allow(improper_ctypes)]
extern {
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
    use stdsimd_test::simd_test;

    use v128::*;
    use x86::ssse3 as ssse3;

    #[simd_test = "ssse3"]
    unsafe fn _mm_abs_epi8() {
        let r = ssse3::_mm_abs_epi8(i8x16::splat(-5));
        assert_eq!(r, u8x16::splat(5));
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_abs_epi16() {
        let r = ssse3::_mm_abs_epi16(i16x8::splat(-5));
        assert_eq!(r, u16x8::splat(5));
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_abs_epi32() {
        let r = ssse3::_mm_abs_epi32(i32x4::splat(-5));
        assert_eq!(r, u32x4::splat(5));
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_shuffle_epi8() {
        let a = u8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
        );
        let b = u8x16::new(
            4, 128, 4, 3,
            24, 12, 6, 19,
            12, 5, 5, 10,
            4, 1, 8, 0,
        );
        let expected = u8x16::new(
            5, 0, 5, 4,
            9, 13, 7, 4,
            13, 6, 6, 11,
            5, 2, 9, 1,
        );
        let r = ssse3::_mm_shuffle_epi8(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_hadd_epi16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(4, 128, 4, 3, 24, 12, 6, 19);
        let expected = i16x8::new(3, 7, 11, 15, 132, 7, 36, 25);
        let r = ssse3::_mm_hadd_epi16(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_hadds_epi16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(4, 128, 4, 3, 32767, 1, -32768, -1);
        let expected = i16x8::new(3, 7, 11, 15, 132, 7, 32767, -32768);
        let r = ssse3::_mm_hadds_epi16(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_hadd_epi32() {
        let a = i32x4::new(1, 2, 3, 4);
        let b = i32x4::new(4, 128, 4, 3);
        let expected = i32x4::new(3, 7, 132, 7);
        let r = ssse3::_mm_hadd_epi32(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_hsub_epi16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(4, 128, 4, 3, 24, 12, 6, 19);
        let expected = i16x8::new(-1, -1, -1, -1, -124, 1, 12, -13);
        let r = ssse3::_mm_hsub_epi16(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_hsubs_epi16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(4, 128, 4, 3, 32767, -1, -32768, 1);
        let expected = i16x8::new(-1, -1, -1, -1, -124, 1, 32767, -32768);
        let r = ssse3::_mm_hsubs_epi16(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_hsub_epi32() {
        let a = i32x4::new(1, 2, 3, 4);
        let b = i32x4::new(4, 128, 4, 3);
        let expected = i32x4::new(-1, -1, -124, 1);
        let r = ssse3::_mm_hsub_epi32(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_maddubs_epi16() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = i8x16::new(4, 63, 4, 3, 24, 12, 6, 19, 12, 5, 5, 10, 4, 1, 8, 0);
        let expected = i16x8::new(130, 24, 192, 194, 158, 175, 66, 120);
        let r = ssse3::_mm_maddubs_epi16(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_mulhrs_epi16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(4, 128, 4, 3, 32767, -1, -32768, 1);
        let expected = i16x8::new(0, 0, 0, 0, 5, 0, -7, 0);
        let r = ssse3::_mm_mulhrs_epi16(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_sign_epi8() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -14, -15, 16);
        let b = i8x16::new(4, 63, -4, 3, 24, 12, -6, -19, 12, 5, -5, 10, 4, 1, -8, 0);
        let expected = i8x16::new(1, 2, -3, 4, 5, 6, -7, -8, 9, 10, -11, 12, 13, -14, 15, 0);
        let r = ssse3::_mm_sign_epi8(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_sign_epi16() {
        let a = i16x8::new(1, 2, 3, 4, -5, -6, 7, 8);
        let b = i16x8::new(4, 128, 0, 3, 1, -1, -2, 1);
        let expected = i16x8::new(1, 2, 0, 4, -5, 6, -7, 8);
        let r = ssse3::_mm_sign_epi16(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_sign_epi32() {
        let a = i32x4::new(-1, 2, 3, 4);
        let b = i32x4::new(1, -1, 1, 0);
        let expected = i32x4::new(-1, -2, 3, 0);
        let r = ssse3::_mm_sign_epi32(a, b);
        assert_eq!(r, expected);
    }
}
