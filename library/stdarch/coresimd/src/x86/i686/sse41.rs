//! `i686`'s Streaming SIMD Extensions 4.1 (SSE4.1)

use v128::*;
use x86::*;

#[cfg(test)]
use stdsimd_test::assert_instr;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.sse41.ptestz"]
    fn ptestz(a: i64x2, mask: i64x2) -> i32;
    #[link_name = "llvm.x86.sse41.ptestc"]
    fn ptestc(a: i64x2, mask: i64x2) -> i32;
    #[link_name = "llvm.x86.sse41.ptestnzc"]
    fn ptestnzc(a: i64x2, mask: i64x2) -> i32;
}

/// Tests whether the specified bits in a 128-bit integer vector are all
/// zeros.
///
/// Arguments:
///
/// * `a` - A 128-bit integer vector containing the bits to be tested.
/// * `mask` - A 128-bit integer vector selecting which bits to test in
///            operand `a`.
///
/// Returns:
///
/// * `1` - if the specified bits are all zeros,
/// * `0` - otherwise.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(ptest))]
pub unsafe fn _mm_testz_si128(a: __m128i, mask: __m128i) -> i32 {
    ptestz(i64x2::from(a), i64x2::from(mask))
}

/// Tests whether the specified bits in a 128-bit integer vector are all
/// ones.
///
/// Arguments:
///
/// * `a` - A 128-bit integer vector containing the bits to be tested.
/// * `mask` - A 128-bit integer vector selecting which bits to test in
///            operand `a`.
///
/// Returns:
///
/// * `1` - if the specified bits are all ones,
/// * `0` - otherwise.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(ptest))]
pub unsafe fn _mm_testc_si128(a: __m128i, mask: __m128i) -> i32 {
    ptestc(i64x2::from(a), i64x2::from(mask))
}

/// Tests whether the specified bits in a 128-bit integer vector are
/// neither all zeros nor all ones.
///
/// Arguments:
///
/// * `a` - A 128-bit integer vector containing the bits to be tested.
/// * `mask` - A 128-bit integer vector selecting which bits to test in
///            operand `a`.
///
/// Returns:
///
/// * `1` - if the specified bits are neither all zeros nor all ones,
/// * `0` - otherwise.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(ptest))]
pub unsafe fn _mm_testnzc_si128(a: __m128i, mask: __m128i) -> i32 {
    ptestnzc(i64x2::from(a), i64x2::from(mask))
}

/// Tests whether the specified bits in a 128-bit integer vector are all
/// zeros.
///
/// Arguments:
///
/// * `a` - A 128-bit integer vector containing the bits to be tested.
/// * `mask` - A 128-bit integer vector selecting which bits to test in
///            operand `a`.
///
/// Returns:
///
/// * `1` - if the specified bits are all zeros,
/// * `0` - otherwise.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(ptest))]
pub unsafe fn _mm_test_all_zeros(a: __m128i, mask: __m128i) -> i32 {
    _mm_testz_si128(a, mask)
}

/// Tests whether the specified bits in `a` 128-bit integer vector are all
/// ones.
///
/// Argument:
///
/// * `a` - A 128-bit integer vector containing the bits to be tested.
///
/// Returns:
///
/// * `1` - if the bits specified in the operand are all set to 1,
/// * `0` - otherwise.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pcmpeqd))]
#[cfg_attr(test, assert_instr(ptest))]
pub unsafe fn _mm_test_all_ones(a: __m128i) -> i32 {
    _mm_testc_si128(a, _mm_cmpeq_epi32(a, a))
}

/// Tests whether the specified bits in a 128-bit integer vector are
/// neither all zeros nor all ones.
///
/// Arguments:
///
/// * `a` - A 128-bit integer vector containing the bits to be tested.
/// * `mask` - A 128-bit integer vector selecting which bits to test in
///            operand `a`.
///
/// Returns:
///
/// * `1` - if the specified bits are neither all zeros nor all ones,
/// * `0` - otherwise.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(ptest))]
pub unsafe fn _mm_test_mix_ones_zeros(a: __m128i, mask: __m128i) -> i32 {
    _mm_testnzc_si128(a, mask)
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;
    use x86::i686::sse41;
    use v128::*;

    #[simd_test = "sse4.1"]
    unsafe fn _mm_testz_si128() {
        let a = i8x16::splat(1);
        let mask = i8x16::splat(0);
        let r = sse41::_mm_testz_si128(a.into(), mask.into());
        assert_eq!(r, 1);
        let a = i8x16::splat(0b101);
        let mask = i8x16::splat(0b110);
        let r = sse41::_mm_testz_si128(a.into(), mask.into());
        assert_eq!(r, 0);
        let a = i8x16::splat(0b011);
        let mask = i8x16::splat(0b100);
        let r = sse41::_mm_testz_si128(a.into(), mask.into());
        assert_eq!(r, 1);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_testc_si128() {
        let a = i8x16::splat(-1);
        let mask = i8x16::splat(0);
        let r = sse41::_mm_testc_si128(a.into(), mask.into());
        assert_eq!(r, 1);
        let a = i8x16::splat(0b101);
        let mask = i8x16::splat(0b110);
        let r = sse41::_mm_testc_si128(a.into(), mask.into());
        assert_eq!(r, 0);
        let a = i8x16::splat(0b101);
        let mask = i8x16::splat(0b100);
        let r = sse41::_mm_testc_si128(a.into(), mask.into());
        assert_eq!(r, 1);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_testnzc_si128() {
        let a = i8x16::splat(0);
        let mask = i8x16::splat(1);
        let r = sse41::_mm_testnzc_si128(a.into(), mask.into());
        assert_eq!(r, 0);
        let a = i8x16::splat(-1);
        let mask = i8x16::splat(0);
        let r = sse41::_mm_testnzc_si128(a.into(), mask.into());
        assert_eq!(r, 0);
        let a = i8x16::splat(0b101);
        let mask = i8x16::splat(0b110);
        let r = sse41::_mm_testnzc_si128(a.into(), mask.into());
        assert_eq!(r, 1);
        let a = i8x16::splat(0b101);
        let mask = i8x16::splat(0b101);
        let r = sse41::_mm_testnzc_si128(a.into(), mask.into());
        assert_eq!(r, 0);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_test_all_zeros() {
        let a = i8x16::splat(1);
        let mask = i8x16::splat(0);
        let r = sse41::_mm_test_all_zeros(a.into(), mask.into());
        assert_eq!(r, 1);
        let a = i8x16::splat(0b101);
        let mask = i8x16::splat(0b110);
        let r = sse41::_mm_test_all_zeros(a.into(), mask.into());
        assert_eq!(r, 0);
        let a = i8x16::splat(0b011);
        let mask = i8x16::splat(0b100);
        let r = sse41::_mm_test_all_zeros(a.into(), mask.into());
        assert_eq!(r, 1);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_test_all_ones() {
        let a = i8x16::splat(-1);
        let r = sse41::_mm_test_all_ones(a.into());
        assert_eq!(r, 1);
        let a = i8x16::splat(0b101);
        let r = sse41::_mm_test_all_ones(a.into());
        assert_eq!(r, 0);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_test_mix_ones_zeros() {
        let a = i8x16::splat(0);
        let mask = i8x16::splat(1);
        let r = sse41::_mm_test_mix_ones_zeros(a.into(), mask.into());
        assert_eq!(r, 0);
        let a = i8x16::splat(-1);
        let mask = i8x16::splat(0);
        let r = sse41::_mm_test_mix_ones_zeros(a.into(), mask.into());
        assert_eq!(r, 0);
        let a = i8x16::splat(0b101);
        let mask = i8x16::splat(0b110);
        let r = sse41::_mm_test_mix_ones_zeros(a.into(), mask.into());
        assert_eq!(r, 1);
        let a = i8x16::splat(0b101);
        let mask = i8x16::splat(0b101);
        let r = sse41::_mm_test_mix_ones_zeros(a.into(), mask.into());
        assert_eq!(r, 0);
    }
}
