use v256::*;
use x86::__m256i;

/// Computes the absolute values of packed 32-bit integers in `a`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_abs_epi32(a: i32x8) -> i32x8 {
    unsafe { pabsd(a) }
}

/// Computes the absolute values of packed 16-bit integers in `a`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_abs_epi16(a: i16x16) -> i16x16 {
    unsafe { pabsw(a) }
}

/// Computes the absolute values of packed 8-bit integers in `a`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_abs_epi8(a: i8x32) -> i8x32 {
    unsafe { pabsb(a) }
}

/// Add packed 64-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_add_epi64(a: i64x4, b: i64x4) -> i64x4 {
    a + b
}

/// Add packed 32-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_add_epi32(a: i32x8, b: i32x8) -> i32x8 {
    a + b
}

/// Add packed 16-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_add_epi16(a: i16x16, b: i16x16) -> i16x16 {
    a + b
}

/// Add packed 8-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_add_epi8(a: i8x32, b: i8x32) -> i8x32 {
    a + b
}

/// Add packed 8-bit integers in `a` and `b` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_adds_epi8(a: i8x32, b: i8x32) -> i8x32 {
    unsafe { paddsb(a, b) }
}

/// Add packed 16-bit integers in `a` and `b` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_adds_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { paddsw(a, b) }
}

/// Add packed unsigned 8-bit integers in `a` and `b` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_adds_epu8(a: u8x32, b: u8x32) -> u8x32 {
    unsafe { paddusb(a, b) }
}

/// Add packed unsigned 16-bit integers in `a` and `b` using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_adds_epu16(a: u16x16, b: u16x16) -> u16x16 {
    unsafe { paddusw(a, b) }
}

/// Compute the bitwise AND of 256 bits (representing integer data)
/// in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_and_si256(a: __m256i, b: __m256i) -> __m256i {
    a & b
}

/// Compute the bitwise NOT of 256 bits (representing integer data)
/// in `a` and then AND with `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_andnot_si256(a: __m256i, b: __m256i) -> __m256i {
    (!a) & b
}

/// Average packed unsigned 16-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_avg_epu16 (a: u16x16, b: u16x16) -> u16x16 {
    unsafe { pavgw(a, b) }
}

/// Average packed unsigned 8-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_avg_epu8 (a: u8x32, b: u8x32) -> u8x32 {
    unsafe { pavgb(a, b) }
}

// TODO _mm256_alignr_epi8
// TODO _mm256_blend_epi16
// TODO _mm_blend_epi32
// TODO _mm256_blend_epi32

/// Blend packed 8-bit integers from `a` and `b` using `mask`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_blendv_epi8(a:i8x32,b:i8x32,mask:__m256i) -> i8x32 {
    unsafe { pblendvb(a,b,mask) }
}

// TODO _mm_broadcastb_epi8
// TODO _mm256_broadcastb_epi8
// TODO _mm_broadcastd_epi32
// TODO _mm256_broadcastd_epi32
// TODO _mm_broadcastq_epi64
// TODO _mm256_broadcastq_epi64
// TODO _mm_broadcastsd_pd
// TODO _mm256_broadcastsd_pd
// TODO _mm_broadcastsi128_si256
// TODO _mm256_broadcastsi128_si256
// TODO _mm_broadcastss_ps
// TODO _mm256_broadcastss_ps
// TODO _mm_broadcastw_epi16
// TODO _mm256_broadcastw_epi16
// TODO _mm256_bslli_epi128
// TODO _mm256_bsrli_epi128


/// Compare packed 64-bit integers in `a` and `b` for equality.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpeq_epi64(a: i64x4, b: i64x4) -> i64x4 {
    a.eq(b)
}

/// Compare packed 32-bit integers in `a` and `b` for equality.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpeq_epi32(a: i32x8, b: i32x8) -> i32x8 {
    a.eq(b)
}

/// Compare packed 16-bit integers in `a` and `b` for equality.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpeq_epi16(a: i16x16, b: i16x16) -> i16x16 {
    a.eq(b)
}

/// Compare packed 8-bit integers in `a` and `b` for equality.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpeq_epi8(a: i8x32, b: i8x32) -> i8x32 {
    a.eq(b)
}

/// Compare packed 64-bit integers in `a` and `b` for greater-than.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpgt_epi64(a: i64x4, b: i64x4) -> i64x4 {
    a.gt(b)
}

/// Compare packed 32-bit integers in `a` and `b` for greater-than.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpgt_epi32(a: i32x8, b: i32x8) -> i32x8 {
    a.gt(b)
}

/// Compare packed 16-bit integers in `a` and `b` for greater-than.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpgt_epi16(a: i16x16, b: i16x16) -> i16x16 {
    a.gt(b)
}

/// Compare packed 8-bit integers in `a` and `b` for greater-than.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_cmpgt_epi8(a: i8x32, b: i8x32) -> i8x32 {
    a.gt(b)
}

// TODO _mm256_cvtepi16_epi32
// TODO _mm256_cvtepi16_epi64
// TODO _mm256_cvtepi32_epi64
// TODO _mm256_cvtepi8_epi16
// TODO _mm256_cvtepi8_epi32
// TODO _mm256_cvtepi8_epi64
// TODO _mm256_cvtepu16_epi32
// TODO _mm256_cvtepu16_epi64
// TODO _mm256_cvtepu32_epi64
// TODO _mm256_cvtepu8_epi16
// TODO _mm256_cvtepu8_epi32
// TODO _mm256_cvtepu8_epi64
// TODO _m128i _mm256_extracti128_si256

/// Horizontally add adjacent pairs of 16-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_hadd_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { phaddw(a, b) }
}

/// Horizontally add adjacent pairs of 32-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_hadd_epi32(a: i32x8, b: i32x8) -> i32x8 {
    unsafe { phaddd(a, b) }
}

/// Horizontally add adjacent pairs of 16-bit integers in `a` and `b`
/// using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_hadds_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { phaddsw(a, b) }
}

/// Horizontally substract adjacent pairs of 16-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_hsub_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { phsubw(a, b) }
}

/// Horizontally substract adjacent pairs of 32-bit integers in `a` and `b`.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_hsub_epi32(a: i32x8, b: i32x8) -> i32x8 {
    unsafe { phsubd(a, b) }
}

/// Horizontally subtract adjacent pairs of 16-bit integers in `a` and `b`
/// using saturation.
#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_hsubs_epi16(a: i16x16, b: i16x16) -> i16x16 {
    unsafe { phsubsw(a, b) }
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx2.pabs.b"]
    fn pabsb(a: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx2.pabs.w"]
    fn pabsw(a: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.pabs.d"]
    fn pabsd(a: i32x8) -> i32x8;
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
}


#[cfg(test)]
mod tests {
    use v256::*;
    use x86::avx2;
    use x86::__m256i;
    use std;

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_abs_epi32() {
        let a = i32x8::new(
            0, 1, -1, std::i32::MAX,
            std::i32::MIN + 1, 100, -100, -32);
        let r = avx2::_mm256_abs_epi32(a);
        let e = i32x8::new(
            0, 1, 1, std::i32::MAX,
            (std::i32::MIN + 1).abs(), 100, 100, 32);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_abs_epi16() {
        let a = i16x16::new(
            0, 1, -1, 2,
            -2, 3, -3, 4,
            -4, 5, -5, std::i16::MAX,
            std::i16::MIN + 1, 100, -100, -32);
        let r = avx2::_mm256_abs_epi16(a);
        let e = i16x16::new(
            0, 1, 1, 2,
            2, 3, 3, 4,
            4, 5, 5, std::i16::MAX,
            (std::i16::MIN + 1).abs(), 100, 100, 32);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_abs_epi8() {
        let a = i8x32::new(
            0, 1, -1, 2,
            -2, 3, -3, 4,
            -4, 5, -5, std::i8::MAX,
            std::i8::MIN + 1, 100, -100, -32,
            0, 1, -1, 2,
            -2, 3, -3, 4,
            -4, 5, -5, std::i8::MAX,
            std::i8::MIN + 1, 100, -100, -32);
        let r = avx2::_mm256_abs_epi8(a);
        let e = i8x32::new(
            0, 1, 1, 2, 2, 3, 3, 4,
            4, 5, 5, std::i8::MAX, (std::i8::MIN + 1).abs(), 100, 100, 32,
            0, 1, 1, 2, 2, 3, 3, 4,
            4, 5, 5, std::i8::MAX, (std::i8::MIN + 1).abs(), 100, 100, 32);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_add_epi64() {
        let a = i64x4::new(-10, 0, 100, 1_000_000_000);
        let b = i64x4::new(-1, 0, 1, 2);
        let r = avx2::_mm256_add_epi64(a, b);
        let e = i64x4::new(-11, 0, 101, 1_000_000_002);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_add_epi32() {
        let a = i32x8::new(-1, 0, 1, 2, 3, 4, 5, 6);
        let b = i32x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = avx2::_mm256_add_epi32(a, b);
        let e = i32x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_add_epi16() {
        let a = i16x16::new(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15);
        let b = i16x16::new(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15);
        let r = avx2::_mm256_add_epi16(a, b);
        let e = i16x16::new(
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 26, 28, 30);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_add_epi8() {
        let a = i8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31);
        let b = i8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31);
        let r = avx2::_mm256_add_epi8(a, b);
        let e = i8x32::new(
            0, 2, 4, 6, 8, 10, 12, 14, 16,
            18, 20, 22, 24, 26, 28, 30, 32,
            34, 36, 38, 40, 42, 44, 46, 48,
            50, 52, 54, 56, 58, 60, 62);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epi8() {
        let a = i8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let b = i8x32::new(
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let r = avx2::_mm256_adds_epi8(a, b);
        let e = i8x32::new(
            32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
            64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epi8_saturate_positive() {
        let a = i8x32::splat(0x7F);
        let b = i8x32::splat(1);
        let r = avx2::_mm256_adds_epi8(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epi8_saturate_negative() {
        let a = i8x32::splat(-0x80);
        let b = i8x32::splat(-1);
        let r = avx2::_mm256_adds_epi8(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epi16() {
        let a = i16x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = i16x16::new(
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47);
        let r = avx2::_mm256_adds_epi16(a,  b);
        let e = i16x16::new(
            32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62);

        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epi16_saturate_positive() {
        let a = i16x16::splat(0x7FFF);
        let b = i16x16::splat(1);
        let r = avx2::_mm256_adds_epi16(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epi16_saturate_negative() {
        let a = i16x16::splat(-0x8000);
        let b = i16x16::splat(-1);
        let r = avx2::_mm256_adds_epi16(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epu8() {
        let a = u8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let b = u8x32::new(
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let r = avx2::_mm256_adds_epu8(a, b);
        let e = u8x32::new(
            32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
            64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epu8_saturate() {
        let a = u8x32::splat(0xFF);
        let b = u8x32::splat(1);
        let r = avx2::_mm256_adds_epu8(a, b);
        assert_eq!(r, a);
    }


    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epu16() {
        let a = u16x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = u16x16::new(
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47);
        let r = avx2::_mm256_adds_epu16(a, b);
        let e = u16x16::new(
            32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62);

        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_adds_epu16_saturate() {
        let a = u16x16::splat(0xFFFF);
        let b = u16x16::splat(1);
        let r = avx2::_mm256_adds_epu16(a, b);
        assert_eq!(r, a);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_and_si256() {
        assert_eq!(
            avx2::_mm256_and_si256(
                __m256i::splat(5), __m256i::splat(3)),__m256i::splat(1));
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_andnot_si256() {
        assert_eq!(
            avx2::_mm256_andnot_si256(__m256i::splat(5), __m256i::splat(3)),
            __m256i::splat(2));
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_avg_epu8() {
        let (a, b) = (u8x32::splat(3), u8x32::splat(9));
        let r = avx2::_mm256_avg_epu8(a, b);
        assert_eq!(r, u8x32::splat(6));
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_avg_epu16() {
        let (a, b) = (u16x16::splat(3), u16x16::splat(9));
        let r = avx2::_mm256_avg_epu16(a, b);
        assert_eq!(r, u16x16::splat(6));
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_blendv_epi8() {
        let (a,b) = (i8x32::splat(4),i8x32::splat(2));
        let mask = i8x32::splat(0).replace(2,-1);
        let e = i8x32::splat(4).replace(2,2);
        let r= avx2::_mm256_blendv_epi8(a,b,mask);
        assert_eq!(r,e);
    }

    #[test]
    fn _mm256_cmpeq_epi8() {
        let a = i8x32::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let b = i8x32::new(
            31, 30, 2, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = avx2::_mm256_cmpeq_epi8(a, b);
        assert_eq!(r, i8x32::splat(0).replace(2,0xFFu8 as i8));
    }

    #[test]
    fn _mm256_cmpeq_epi16() {
        let a = i16x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = i16x16::new(
            15, 14, 2, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = avx2::_mm256_cmpeq_epi16(a, b);
        assert_eq!(r, i16x16::splat(0).replace(2, 0xFFFFu16 as i16));
    }

    #[test]
    fn _mm256_cmpeq_epi32() {
        let a = i32x8::new(0, 1, 2, 3,4,5,6,7);
        let b = i32x8::new(7,6,2,4,3, 2, 1, 0);
        let r = avx2::_mm256_cmpeq_epi32(a, b);
        assert_eq!(r, i32x8::splat(0).replace(2, 0xFFFFFFFFu32 as i32));
    }

    #[test]
    fn _mm256_cmpeq_epi64() {
        let a = i64x4::new(0, 1, 2, 3);
        let b = i64x4::new(3, 2, 2, 0);
        let r = avx2::_mm256_cmpeq_epi64(a, b);
        assert_eq!(r, i64x4::splat(0).replace(
            2, 0xFFFFFFFFFFFFFFFFu64 as i64));
    }

    #[test]
    fn _mm256_cmpgt_epi8() {
        let a = i8x32::splat(0).replace(0, 5);
        let b = i8x32::splat(0);
        let r = avx2::_mm256_cmpgt_epi8(a, b);
        assert_eq!(r, i8x32::splat(0).replace(0, 0xFFu8 as i8));
    }

    #[test]
    fn _mm256_cmpgt_epi16() {
        let a = i16x16::splat(0).replace(0, 5);
        let b = i16x16::splat(0);
        let r = avx2::_mm256_cmpgt_epi16(a, b);
        assert_eq!(r, i16x16::splat(0).replace(0, 0xFFFFu16 as i16));
    }

    #[test]
    fn _mm256_cmpgt_epi32() {
        let a = i32x8::splat(0).replace(0, 5);
        let b = i32x8::splat(0);
        let r = avx2::_mm256_cmpgt_epi32(a, b);
        assert_eq!(r, i32x8::splat(0).replace(0, 0xFFFFFFFFu32 as i32));
    }

    #[test]
    fn _mm256_cmpgt_epi64() {
        let a = i64x4::splat(0).replace(0, 5);
        let b = i64x4::splat(0);
        let r = avx2::_mm256_cmpgt_epi64(a, b);
        assert_eq!(r, i64x4::splat(0).replace(
            0, 0xFFFFFFFFFFFFFFFFu64 as i64));
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_hadd_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_hadd_epi16(a, b);
        let e = i16x16::new(4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8);
        assert_eq!(r,e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_hadd_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_hadd_epi32(a, b);
        let e = i32x8::new(4, 4, 8, 8, 4, 4, 8, 8);
        assert_eq!(r,e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_hadds_epi16() {
        let a = i16x16::splat(2).replace(0,0x7FFF).replace(1,1);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_hadds_epi16(a, b);
        let e = i16x16::new(
            0x7FFF, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8);
        assert_eq!(r,e);
    }

    #[test]
    #[target_feature ="+avx2"]
    fn _mm256_hsub_epi16() {
        let a = i16x16::splat(2);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_hsub_epi16(a, b);
        let e = i16x16::splat(0);
        assert_eq!(r,e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_hsub_epi32() {
        let a = i32x8::splat(2);
        let b = i32x8::splat(4);
        let r = avx2::_mm256_hsub_epi32(a, b);
        let e = i32x8::splat(0);
        assert_eq!(r,e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm256_hsubs_epi16() {
        let a = i16x16::splat(2).replace(0,0x7FFF).replace(1,-1);
        let b = i16x16::splat(4);
        let r = avx2::_mm256_hsubs_epi16(a, b);
        let e = i16x16::splat(0).replace(0,0x7FFF);
        assert_eq!(r,e);
    }


}
