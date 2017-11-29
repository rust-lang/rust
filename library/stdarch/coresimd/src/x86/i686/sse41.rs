//! `i686`'s Streaming SIMD Extensions 4.1 (SSE4.1)

use v128::*;

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

/// Extract an 64-bit integer from `a` selected with `imm8`
#[inline(always)]
#[target_feature = "+sse4.1"]
// TODO: Add test for Windows
#[cfg_attr(all(test, not(windows), target_arch = "x86_64"),
           assert_instr(pextrq, imm8 = 1))]
// On x86 this emits 2 pextrd instructions
#[cfg_attr(all(test, not(windows), target_arch = "x86"),
           assert_instr(pextrd, imm8 = 1))]
pub unsafe fn _mm_extract_epi64(a: i64x2, imm8: u8) -> i64 {
    a.extract((imm8 & 0b1) as u32)
}

/// Return a copy of `a` with the 64-bit integer from `i` inserted at a
/// location specified by `imm8`.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(all(test, target_arch = "x86_64"), assert_instr(pinsrq, imm8 = 0))]
// On x86 this emits 2 pinsrd instructions
#[cfg_attr(all(test, target_arch = "x86"), assert_instr(pinsrd, imm8 = 0))]
pub unsafe fn _mm_insert_epi64(a: i64x2, i: i64, imm8: u8) -> i64x2 {
    a.replace((imm8 & 0b1) as u32, i)
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
    ptestz(a.into(), mask.into())
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
    ptestc(a.into(), mask.into())
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
    ptestnzc(a.into(), mask.into())
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
    _mm_testc_si128(a, ::x86::_mm_cmpeq_epi32(a.into(), a.into()).into())
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
    unsafe fn _mm_extract_epi64() {
        let a = i64x2::new(0, 1);
        let r = sse41::_mm_extract_epi64(a, 1);
        assert_eq!(r, 1);
        let r = sse41::_mm_extract_epi64(a, 3);
        assert_eq!(r, 1);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_insert_epi64() {
        let a = i64x2::splat(0);
        let e = i64x2::splat(0).replace(1, 32);
        let r = sse41::_mm_insert_epi64(a, 32, 1);
        assert_eq!(r, e);
        let r = sse41::_mm_insert_epi64(a, 32, 3);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_testz_si128() {
        let a = __m128i::splat(1);
        let mask = __m128i::splat(0);
        let r = sse41::_mm_testz_si128(a, mask);
        assert_eq!(r, 1);
        let a = __m128i::splat(0b101);
        let mask = __m128i::splat(0b110);
        let r = sse41::_mm_testz_si128(a, mask);
        assert_eq!(r, 0);
        let a = __m128i::splat(0b011);
        let mask = __m128i::splat(0b100);
        let r = sse41::_mm_testz_si128(a, mask);
        assert_eq!(r, 1);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_testc_si128() {
        let a = __m128i::splat(-1);
        let mask = __m128i::splat(0);
        let r = sse41::_mm_testc_si128(a, mask);
        assert_eq!(r, 1);
        let a = __m128i::splat(0b101);
        let mask = __m128i::splat(0b110);
        let r = sse41::_mm_testc_si128(a, mask);
        assert_eq!(r, 0);
        let a = __m128i::splat(0b101);
        let mask = __m128i::splat(0b100);
        let r = sse41::_mm_testc_si128(a, mask);
        assert_eq!(r, 1);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_testnzc_si128() {
        let a = __m128i::splat(0);
        let mask = __m128i::splat(1);
        let r = sse41::_mm_testnzc_si128(a, mask);
        assert_eq!(r, 0);
        let a = __m128i::splat(-1);
        let mask = __m128i::splat(0);
        let r = sse41::_mm_testnzc_si128(a, mask);
        assert_eq!(r, 0);
        let a = __m128i::splat(0b101);
        let mask = __m128i::splat(0b110);
        let r = sse41::_mm_testnzc_si128(a, mask);
        assert_eq!(r, 1);
        let a = __m128i::splat(0b101);
        let mask = __m128i::splat(0b101);
        let r = sse41::_mm_testnzc_si128(a, mask);
        assert_eq!(r, 0);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_test_all_zeros() {
        let a = __m128i::splat(1);
        let mask = __m128i::splat(0);
        let r = sse41::_mm_test_all_zeros(a, mask);
        assert_eq!(r, 1);
        let a = __m128i::splat(0b101);
        let mask = __m128i::splat(0b110);
        let r = sse41::_mm_test_all_zeros(a, mask);
        assert_eq!(r, 0);
        let a = __m128i::splat(0b011);
        let mask = __m128i::splat(0b100);
        let r = sse41::_mm_test_all_zeros(a, mask);
        assert_eq!(r, 1);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_test_all_ones() {
        let a = __m128i::splat(-1);
        let r = sse41::_mm_test_all_ones(a);
        assert_eq!(r, 1);
        let a = __m128i::splat(0b101);
        let r = sse41::_mm_test_all_ones(a);
        assert_eq!(r, 0);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_test_mix_ones_zeros() {
        let a = __m128i::splat(0);
        let mask = __m128i::splat(1);
        let r = sse41::_mm_test_mix_ones_zeros(a, mask);
        assert_eq!(r, 0);
        let a = __m128i::splat(-1);
        let mask = __m128i::splat(0);
        let r = sse41::_mm_test_mix_ones_zeros(a, mask);
        assert_eq!(r, 0);
        let a = __m128i::splat(0b101);
        let mask = __m128i::splat(0b110);
        let r = sse41::_mm_test_mix_ones_zeros(a, mask);
        assert_eq!(r, 1);
        let a = __m128i::splat(0b101);
        let mask = __m128i::splat(0b101);
        let r = sse41::_mm_test_mix_ones_zeros(a, mask);
        assert_eq!(r, 0);
    }
}
