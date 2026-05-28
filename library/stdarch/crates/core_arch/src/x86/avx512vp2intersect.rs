//! Vector Pair Intersection to a Pair of Mask Registers (VP2INTERSECT)

use crate::core_arch::{simd::*, x86::*};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Compute intersection of packed 32-bit integer vectors a and b,
/// and store indication of match in the corresponding bit of two mask registers
/// specified by k1 and k2. A match in corresponding elements of a and b is
/// indicated by a set bit in the corresponding bit of the mask registers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_2intersect_epi32&expand=0)
#[inline]
#[target_feature(enable = "avx512vp2intersect,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512vp2intersect", issue = "111137")]
#[cfg_attr(test, assert_instr(vp2intersectd))]
pub unsafe fn _mm_2intersect_epi32(a: __m128i, b: __m128i, k1: *mut __mmask8, k2: *mut __mmask8) {
    (*k1, *k2) = vp2intersectd_128(a.as_i32x4(), b.as_i32x4());
}

/// Compute intersection of packed 64-bit integer vectors a and b,
/// and store indication of match in the corresponding bit of two mask registers
/// specified by k1 and k2. A match in corresponding elements of a and b is
/// indicated by a set bit in the corresponding bit of the mask registers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_2intersect_epi64&expand=0)
#[inline]
#[target_feature(enable = "avx512vp2intersect,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512vp2intersect", issue = "111137")]
#[cfg_attr(test, assert_instr(vp2intersectq))]
pub unsafe fn _mm_2intersect_epi64(a: __m128i, b: __m128i, k1: *mut __mmask8, k2: *mut __mmask8) {
    (*k1, *k2) = vp2intersectq_128(a.as_i64x2(), b.as_i64x2());
}

/// Compute intersection of packed 32-bit integer vectors a and b,
/// and store indication of match in the corresponding bit of two mask registers
/// specified by k1 and k2. A match in corresponding elements of a and b is
/// indicated by a set bit in the corresponding bit of the mask registers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_2intersect_epi32&expand=0)
#[inline]
#[target_feature(enable = "avx512vp2intersect,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512vp2intersect", issue = "111137")]
#[cfg_attr(test, assert_instr(vp2intersectd))]
pub unsafe fn _mm256_2intersect_epi32(
    a: __m256i,
    b: __m256i,
    k1: *mut __mmask8,
    k2: *mut __mmask8,
) {
    (*k1, *k2) = vp2intersectd_256(a.as_i32x8(), b.as_i32x8());
}

/// Compute intersection of packed 64-bit integer vectors a and b,
/// and store indication of match in the corresponding bit of two mask registers
/// specified by k1 and k2. A match in corresponding elements of a and b is
/// indicated by a set bit in the corresponding bit of the mask registers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_2intersect_epi64&expand=0)
#[inline]
#[target_feature(enable = "avx512vp2intersect,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512vp2intersect", issue = "111137")]
#[cfg_attr(test, assert_instr(vp2intersectq))]
pub unsafe fn _mm256_2intersect_epi64(
    a: __m256i,
    b: __m256i,
    k1: *mut __mmask8,
    k2: *mut __mmask8,
) {
    (*k1, *k2) = vp2intersectq_256(a.as_i64x4(), b.as_i64x4());
}

/// Compute intersection of packed 32-bit integer vectors a and b,
/// and store indication of match in the corresponding bit of two mask registers
/// specified by k1 and k2. A match in corresponding elements of a and b is
/// indicated by a set bit in the corresponding bit of the mask registers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_2intersect_epi32&expand=0)
#[inline]
#[target_feature(enable = "avx512vp2intersect,avx512f")]
#[unstable(feature = "stdarch_x86_avx512vp2intersect", issue = "111137")]
#[cfg_attr(test, assert_instr(vp2intersectd))]
pub unsafe fn _mm512_2intersect_epi32(
    a: __m512i,
    b: __m512i,
    k1: *mut __mmask16,
    k2: *mut __mmask16,
) {
    (*k1, *k2) = vp2intersectd_512(a.as_i32x16(), b.as_i32x16());
}

/// Compute intersection of packed 64-bit integer vectors a and b,
/// and store indication of match in the corresponding bit of two mask registers
/// specified by k1 and k2. A match in corresponding elements of a and b is
/// indicated by a set bit in the corresponding bit of the mask registers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_2intersect_epi64&expand=0)
#[inline]
#[target_feature(enable = "avx512vp2intersect,avx512f")]
#[unstable(feature = "stdarch_x86_avx512vp2intersect", issue = "111137")]
#[cfg_attr(test, assert_instr(vp2intersectq))]
pub unsafe fn _mm512_2intersect_epi64(
    a: __m512i,
    b: __m512i,
    k1: *mut __mmask8,
    k2: *mut __mmask8,
) {
    (*k1, *k2) = vp2intersectq_512(a.as_i64x8(), b.as_i64x8());
}

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.x86.avx512.vp2intersect.d.128"]
    fn vp2intersectd_128(a: i32x4, b: i32x4) -> (u8, u8);
    #[link_name = "llvm.x86.avx512.vp2intersect.q.128"]
    fn vp2intersectq_128(a: i64x2, b: i64x2) -> (u8, u8);

    #[link_name = "llvm.x86.avx512.vp2intersect.d.256"]
    fn vp2intersectd_256(a: i32x8, b: i32x8) -> (u8, u8);
    #[link_name = "llvm.x86.avx512.vp2intersect.q.256"]
    fn vp2intersectq_256(a: i64x4, b: i64x4) -> (u8, u8);

    #[link_name = "llvm.x86.avx512.vp2intersect.d.512"]
    fn vp2intersectd_512(a: i32x16, b: i32x16) -> (u16, u16);
    #[link_name = "llvm.x86.avx512.vp2intersect.q.512"]
    fn vp2intersectq_512(a: i64x8, b: i64x8) -> (u8, u8);
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;
    use stdarch_test::simd_test;

    #[simd_test(enable = "avx512vp2intersect,avx512vl")]
    unsafe fn test_mm_2intersect_epi32() {
        let mut k1 = 0;
        let mut k2 = 0;

        let a = _mm_set_epi32(1, 2, 3, 4);
        let b = _mm_set_epi32(3, 4, 5, 6);
        _mm_2intersect_epi32(a, b, &mut k1, &mut k2);
        assert_eq!(k1, 0b0011);
        assert_eq!(k2, 0b1100);

        let a = _mm_set_epi32(1, 2, 3, 4);
        let b = _mm_set_epi32(2, 3, 4, 5);
        _mm_2intersect_epi32(a, b, &mut k1, &mut k2);
        assert_eq!(k1, 0b0111);
        assert_eq!(k2, 0b1110);
    }

    #[simd_test(enable = "avx512vp2intersect,avx512vl")]
    unsafe fn test_mm_2intersect_epi64() {
        let mut k1 = 0;
        let mut k2 = 0;

        let a = _mm_set_epi64x(1, 2);
        let b = _mm_set_epi64x(2, 3);
        _mm_2intersect_epi64(a, b, &mut k1, &mut k2);
        assert_eq!(k1, 0b01);
        assert_eq!(k2, 0b10);

        let a = _mm_set_epi64x(1, 2);
        let b = _mm_set_epi64x(2, 2);
        _mm_2intersect_epi64(a, b, &mut k1, &mut k2);
        assert_eq!(k1, 0b01);
        assert_eq!(k2, 0b11);
    }

    #[simd_test(enable = "avx512vp2intersect,avx512vl")]
    unsafe fn test_mm256_2intersect_epi32() {
        let mut k1 = 0;
        let mut k2 = 0;

        let a = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm256_set_epi32(5, 6, 7, 8, 9, 10, 11, 12);
        _mm256_2intersect_epi32(a, b, &mut k1, &mut k2);
        assert_eq!(k1, 0b00001111);
        assert_eq!(k2, 0b11110000);

        let a = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm256_set_epi32(2, 3, 4, 5, 6, 7, 8, 9);
        _mm256_2intersect_epi32(a, b, &mut k1, &mut k2);
        assert_eq!(k1, 0b01111111);
        assert_eq!(k2, 0b11111110);
    }

    #[simd_test(enable = "avx512vp2intersect,avx512vl")]
    unsafe fn test_mm256_2intersect_epi64() {
        let mut k1 = 0;
        let mut k2 = 0;

        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm256_set_epi64x(3, 4, 5, 6);
        _mm256_2intersect_epi64(a, b, &mut k1, &mut k2);
        assert_eq!(k1, 0b0011);
        assert_eq!(k2, 0b1100);

        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm256_set_epi64x(2, 3, 4, 5);
        _mm256_2intersect_epi64(a, b, &mut k1, &mut k2);
        assert_eq!(k1, 0b0111);
        assert_eq!(k2, 0b1110);
    }

    #[simd_test(enable = "avx512vp2intersect,avx512f")]
    unsafe fn test_mm512_2intersect_epi32() {
        let mut k1 = 0;
        let mut k2 = 0;

        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm512_set_epi32(
            9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        );
        _mm512_2intersect_epi32(a, b, &mut k1, &mut k2);
        assert_eq!(k1, 0b0000000011111111);
        assert_eq!(k2, 0b1111111100000000);

        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm512_set_epi32(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);
        _mm512_2intersect_epi32(a, b, &mut k1, &mut k2);
        assert_eq!(k1, 0b0111111111111111);
        assert_eq!(k2, 0b1111111111111110);
    }

    #[simd_test(enable = "avx512vp2intersect,avx512f")]
    unsafe fn test_mm512_2intersect_epi64() {
        let mut k1 = 0;
        let mut k2 = 0;

        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm512_set_epi64(5, 6, 7, 8, 9, 10, 11, 12);
        _mm512_2intersect_epi64(a, b, &mut k1, &mut k2);
        assert_eq!(k1, 0b00001111);
        assert_eq!(k2, 0b11110000);

        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm512_set_epi64(2, 3, 4, 5, 6, 7, 8, 9);
        _mm512_2intersect_epi64(a, b, &mut k1, &mut k2);
        assert_eq!(k1, 0b01111111);
        assert_eq!(k2, 0b11111110);
    }
}
