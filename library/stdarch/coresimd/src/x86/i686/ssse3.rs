//! Supplemental Streaming SIMD Extensions 3 (SSSE3)

#[cfg(test)]
use stdsimd_test::assert_instr;

use v64::*;

/// Compute the absolute value of packed 8-bit integers in `a` and
/// return the unsigned results.
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pabsb))]
pub unsafe fn _mm_abs_pi8(a: __m64) -> __m64 {
    pabsb(a)
}

/// Compute the absolute value of packed 8-bit integers in `a`, and return the
/// unsigned results.
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pabsw))]
pub unsafe fn _mm_abs_pi16(a: __m64) -> __m64 {
    pabsw(a)
}

/// Compute the absolute value of packed 32-bit integers in `a`, and return the
/// unsigned results.
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pabsd))]
pub unsafe fn _mm_abs_pi32(a: __m64) -> __m64 {
    pabsd(a)
}

/// Shuffle packed 8-bit integers in `a` according to shuffle control mask in
/// the corresponding 8-bit element of `b`, and return the results
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pshufb))]
pub unsafe fn _mm_shuffle_pi8(a: __m64, b: __m64) -> __m64 {
    pshufb(a, b)
}

/// Concatenates the two 64-bit integer vector operands, and right-shifts
/// the result by the number of bytes specified in the immediate operand.
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(palignr, n = 15))]
pub unsafe fn _mm_alignr_pi8(a: __m64, b: __m64, n: i32) -> __m64 {
    macro_rules! call {
        ($imm8:expr) => {
            palignrb(a, b, $imm8)
        }
    }
    constify_imm8!(n, call)
}

/// Horizontally add the adjacent pairs of values contained in 2 packed
/// 64-bit vectors of [4 x i16].
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phaddw))]
pub unsafe fn _mm_hadd_pi16(a: __m64, b: __m64) -> __m64 {
    phaddw(a, b)
}

/// Horizontally add the adjacent pairs of values contained in 2 packed
/// 64-bit vectors of [2 x i32].
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phaddd))]
pub unsafe fn _mm_hadd_pi32(a: __m64, b: __m64) -> __m64 {
    phaddd(a, b)
}

/// Horizontally add the adjacent pairs of values contained in 2 packed
/// 64-bit vectors of [4 x i16]. Positive sums greater than 7FFFh are
/// saturated to 7FFFh. Negative sums less than 8000h are saturated to 8000h.
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phaddsw))]
pub unsafe fn _mm_hadds_pi16(a: __m64, b: __m64) -> __m64 {
    phaddsw(a, b)
}

/// Horizontally subtracts the adjacent pairs of values contained in 2
/// packed 64-bit vectors of [4 x i16].
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phsubw))]
pub unsafe fn _mm_hsub_pi16(a: __m64, b: __m64) -> __m64 {
    phsubw(a, b)
}

/// Horizontally subtracts the adjacent pairs of values contained in 2
/// packed 64-bit vectors of [2 x i32].
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phsubd))]
pub unsafe fn _mm_hsub_pi32(a: __m64, b: __m64) -> __m64 {
    phsubd(a, b)
}

/// Horizontally subtracts the adjacent pairs of values contained in 2
/// packed 64-bit vectors of [4 x i16]. Positive differences greater than
/// 7FFFh are saturated to 7FFFh. Negative differences less than 8000h are
/// saturated to 8000h.
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(phsubsw))]
pub unsafe fn _mm_hsubs_pi16(a: __m64, b: __m64) -> __m64 {
    phsubsw(a, b)
}

/// Multiplies corresponding pairs of packed 8-bit unsigned integer
/// values contained in the first source operand and packed 8-bit signed
/// integer values contained in the second source operand, adds pairs of
/// contiguous products with signed saturation, and writes the 16-bit sums to
/// the corresponding bits in the destination.
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pmaddubsw))]
pub unsafe fn _mm_maddubs_pi16(a: __m64, b: __m64) -> __m64 {
    pmaddubsw(a, b)
}

/// Multiplies packed 16-bit signed integer values, truncates the 32-bit
/// products to the 18 most significant bits by right-shifting, rounds the
/// truncated value by adding 1, and writes bits [16:1] to the destination.
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(pmulhrsw))]
pub unsafe fn _mm_mulhrs_pi16(a: __m64, b: __m64) -> __m64 {
    pmulhrsw(a, b)
}

/// Negate packed 8-bit integers in `a` when the corresponding signed 8-bit
/// integer in `b` is negative, and return the results.
/// Element in result are zeroed out when the corresponding element in `b` is
/// zero.
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(psignb))]
pub unsafe fn _mm_sign_pi8(a: __m64, b: __m64) -> __m64 {
    psignb(a, b)
}

/// Negate packed 16-bit integers in `a` when the corresponding signed 16-bit
/// integer in `b` is negative, and return the results.
/// Element in result are zeroed out when the corresponding element in `b` is
/// zero.
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(psignw))]
pub unsafe fn _mm_sign_pi16(a: __m64, b: __m64) -> __m64 {
    psignw(a, b)
}

/// Negate packed 32-bit integers in `a` when the corresponding signed 32-bit
/// integer in `b` is negative, and return the results.
/// Element in result are zeroed out when the corresponding element in `b` is
/// zero.
#[inline(always)]
#[target_feature(enable = "ssse3")]
#[cfg_attr(test, assert_instr(psignd))]
pub unsafe fn _mm_sign_pi32(a: __m64, b: __m64) -> __m64 {
    psignd(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
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

    use v64::*;
    use x86::i686::ssse3;

    #[simd_test = "ssse3"]
    unsafe fn _mm_abs_pi8() {
        let r = u8x8::from(ssse3::_mm_abs_pi8(i8x8::splat(-5).into()));
        assert_eq!(r, u8x8::splat(5));
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_abs_pi16() {
        let r = u16x4::from(ssse3::_mm_abs_pi16(i16x4::splat(-5).into()));
        assert_eq!(r, u16x4::splat(5));
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_abs_pi32() {
        let r = u32x2::from(ssse3::_mm_abs_pi32(i32x2::splat(-5).into()));
        assert_eq!(r, u32x2::splat(5));
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_shuffle_pi8() {
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u8x8::new(4, 128, 4, 3, 24, 12, 6, 19);
        let expected = u8x8::new(5, 0, 5, 4, 1, 5, 7, 4);
        let r = u8x8::from(ssse3::_mm_shuffle_pi8(a.into(), b.into()));
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_alignr_pi8() {
        let a = u32x2::new(0x89ABCDEF_u32, 0x01234567_u32);
        let b = u32x2::new(0xBBAA9988_u32, 0xFFDDEECC_u32);
        let r = ssse3::_mm_alignr_pi8(
            u8x8::from(a).into(),
            u8x8::from(b).into(),
            4,
        );
        assert_eq!(r, ::std::mem::transmute(0x89abcdefffddeecc_u64));
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_hadd_pi16() {
        let a = i16x4::new(1, 2, 3, 4);
        let b = i16x4::new(4, 128, 4, 3);
        let expected = i16x4::new(3, 7, 132, 7);
        let r = i16x4::from(ssse3::_mm_hadd_pi16(a.into(), b.into()));
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_hadd_pi32() {
        let a = i32x2::new(1, 2);
        let b = i32x2::new(4, 128);
        let expected = i32x2::new(3, 132);
        let r = i32x2::from(ssse3::_mm_hadd_pi32(a.into(), b.into()));
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_hadds_pi16() {
        let a = i16x4::new(1, 2, 3, 4);
        let b = i16x4::new(32767, 1, -32768, -1);
        let expected = i16x4::new(3, 7, 32767, -32768);
        let r = i16x4::from(ssse3::_mm_hadds_pi16(a.into(), b.into()));
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_hsub_pi16() {
        let a = i16x4::new(1, 2, 3, 4);
        let b = i16x4::new(4, 128, 4, 3);
        let expected = i16x4::new(-1, -1, -124, 1);
        let r = i16x4::from(ssse3::_mm_hsub_pi16(a.into(), b.into()));
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_hsub_pi32() {
        let a = i32x2::new(1, 2);
        let b = i32x2::new(4, 128);
        let expected = i32x2::new(-1, -124);
        let r = i32x2::from(ssse3::_mm_hsub_pi32(a.into(), b.into()));
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_hsubs_pi16() {
        let a = i16x4::new(1, 2, 3, 4);
        let b = i16x4::new(4, 128, 4, 3);
        let expected = i16x4::new(-1, -1, -124, 1);
        let r = i16x4::from(ssse3::_mm_hsubs_pi16(a.into(), b.into()));
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_maddubs_pi16() {
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i8x8::new(4, 63, 4, 3, 24, 12, 6, 19);
        let expected = i16x4::new(130, 24, 192, 194);
        let r = i16x4::from(ssse3::_mm_maddubs_pi16(a.into(), b.into()));
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_mulhrs_pi16() {
        let a = i16x4::new(1, 2, 3, 4);
        let b = i16x4::new(4, 32767, -1, -32768);
        let expected = i16x4::new(0, 2, 0, -4);
        let r = i16x4::from(ssse3::_mm_mulhrs_pi16(a.into(), b.into()));
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_sign_pi8() {
        let a = i8x8::new(1, 2, 3, 4, -5, -6, 7, 8);
        let b = i8x8::new(4, 64, 0, 3, 1, -1, -2, 1);
        let expected = i8x8::new(1, 2, 0, 4, -5, 6, -7, 8);
        let r = i8x8::from(ssse3::_mm_sign_pi8(a.into(), b.into()));
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_sign_pi16() {
        let a = i16x4::new(-1, 2, 3, 4);
        let b = i16x4::new(1, -1, 1, 0);
        let expected = i16x4::new(-1, -2, 3, 0);
        let r = i16x4::from(ssse3::_mm_sign_pi16(a.into(), b.into()));
        assert_eq!(r, expected);
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_sign_pi32() {
        let a = i32x2::new(-1, 2);
        let b = i32x2::new(1, 0);
        let expected = i32x2::new(-1, 0);
        let r = i32x2::from(ssse3::_mm_sign_pi32(a.into(), b.into()));
        assert_eq!(r, expected);
    }
}
