//! `i686` Streaming SIMD Extensions (SSE)

use v64::{i16x4, u8x8};
use core::mem;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// This type is only required for mapping vector types to llvm's `x86_mmx`
/// type.
#[allow(non_camel_case_types)]
#[repr(simd)]
struct __m64(i64);

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.mmx.pmaxs.w"]
    fn pmaxsw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pmaxu.b"]
    fn pmaxub(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pmins.w"]
    fn pminsw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pminu.b"]
    fn pminub(a: __m64, b: __m64) -> __m64;
}

/// Compares the packed 16-bit signed integers of `a` and `b` writing the
/// greatest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pmaxsw))]
pub unsafe fn _mm_max_pi16(a: i16x4, b: i16x4) -> i16x4 {
    mem::transmute(pmaxsw(mem::transmute(a), mem::transmute(b)))
}

/// Compares the packed 16-bit signed integers of `a` and `b` writing the
/// greatest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pmaxsw))]
pub unsafe fn _m_pmaxsw(a: i16x4, b: i16x4) -> i16x4 {
    _mm_max_pi16(a, b)
}

/// Compares the packed 8-bit signed integers of `a` and `b` writing the
/// greatest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pmaxub))]
pub unsafe fn _mm_max_pu8(a: u8x8, b: u8x8) -> u8x8 {
    mem::transmute(pmaxub(mem::transmute(a), mem::transmute(b)))
}

/// Compares the packed 8-bit signed integers of `a` and `b` writing the
/// greatest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pmaxub))]
pub unsafe fn _m_pmaxub(a: u8x8, b: u8x8) -> u8x8 {
    _mm_max_pu8(a, b)
}

/// Compares the packed 16-bit signed integers of `a` and `b` writing the
/// smallest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pminsw))]
pub unsafe fn _mm_min_pi16(a: i16x4, b: i16x4) -> i16x4 {
    mem::transmute(pminsw(mem::transmute(a), mem::transmute(b)))
}

/// Compares the packed 16-bit signed integers of `a` and `b` writing the
/// smallest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pminsw))]
pub unsafe fn _m_pminsw(a: i16x4, b: i16x4) -> i16x4 {
    _mm_min_pi16(a, b)
}

/// Compares the packed 8-bit signed integers of `a` and `b` writing the
/// smallest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pminub))]
pub unsafe fn _mm_min_pu8(a: u8x8, b: u8x8) -> u8x8 {
    mem::transmute(pminub(mem::transmute(a), mem::transmute(b)))
}

/// Compares the packed 8-bit signed integers of `a` and `b` writing the
/// smallest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pminub))]
pub unsafe fn _m_pminub(a: u8x8, b: u8x8) -> u8x8 {
    _mm_min_pu8(a, b)
}

#[cfg(test)]
mod tests {
    use v64::{i16x4, u8x8};
    use x86::i686::sse;
    use stdsimd_test::simd_test;

    #[simd_test = "sse"]
    unsafe fn _mm_max_pi16() {
        let a = i16x4::new(-1, 6, -3, 8);
        let b = i16x4::new(5, -2, 7, -4);
        let r = i16x4::new(5, 6, 7, 8);

        assert_eq!(r, sse::_mm_max_pi16(a, b));
        assert_eq!(r, sse::_m_pmaxsw(a, b));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_max_pu8() {
        let a = u8x8::new(2, 6, 3, 8, 2, 6, 3, 8);
        let b = u8x8::new(5, 2, 7, 4, 5, 2, 7, 4);
        let r = u8x8::new(5, 6, 7, 8, 5, 6, 7, 8);

        assert_eq!(r, sse::_mm_max_pu8(a, b));
        assert_eq!(r, sse::_m_pmaxub(a, b));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_min_pi16() {
        let a = i16x4::new(-1, 6, -3, 8);
        let b = i16x4::new(5, -2, 7, -4);
        let r = i16x4::new(-1, -2, -3, -4);

        assert_eq!(r, sse::_mm_min_pi16(a, b));
        assert_eq!(r, sse::_m_pminsw(a, b));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_min_pu8() {
        let a = u8x8::new(2, 6, 3, 8, 2, 6, 3, 8);
        let b = u8x8::new(5, 2, 7, 4, 5, 2, 7, 4);
        let r = u8x8::new(2, 2, 3, 4, 2, 2, 3, 4);

        assert_eq!(r, sse::_mm_min_pu8(a, b));
        assert_eq!(r, sse::_m_pminub(a, b));
    }
}
