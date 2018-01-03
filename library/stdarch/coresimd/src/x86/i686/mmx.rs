//! `i586` MMX instruction set.
//!
//! The intrinsics here roughly correspond to those in the `mmintrin.h` C
//! header.
//!
//! The reference is [Intel 64 and IA-32 Architectures Software Developer's
//! Manual Volume 2: Instruction Set Reference, A-Z][intel64_ref].
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf

use v64::*;
use core::mem;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Constructs a 64-bit integer vector initialized to zero.
#[inline(always)]
#[target_feature = "+mmx,+sse"]
// FIXME: this produces a movl instead of xorps on x86
// FIXME: this produces a xor intrinsic instead of xorps on x86_64
#[cfg_attr(all(test, target_arch = "x86_64"), assert_instr(xor))]
pub unsafe fn _mm_setzero_si64() -> __m64 {
    mem::transmute(0_i64)
}

/// Convert packed 16-bit integers from `a` and `b` to packed 8-bit integers
/// using signed saturation.
///
/// Positive values greater than 0x7F are saturated to 0x7F. Negative values
/// less than 0x80 are saturated to 0x80.
#[inline(always)]
#[target_feature = "+mmx,+sse"]
#[cfg_attr(test, assert_instr(packsswb))]
pub unsafe fn _mm_packs_pi16(a: i16x4, b: i16x4) -> i8x8 {
    mem::transmute(packsswb(mem::transmute(a), mem::transmute(b)))
}

/// Convert packed 32-bit integers from `a` and `b` to packed 16-bit integers
/// using signed saturation.
///
/// Positive values greater than 0x7F are saturated to 0x7F. Negative values
/// less than 0x80 are saturated to 0x80.
#[inline(always)]
#[target_feature = "+mmx,+sse"]
#[cfg_attr(test, assert_instr(packssdw))]
pub unsafe fn _mm_packs_pi32(a: i32x2, b: i32x2) -> i16x4 {
    mem::transmute(packssdw(mem::transmute(a), mem::transmute(b)))
}

/// Compares the 8-bit integer elements of two 64-bit integer vectors of
/// [8 x i8] to determine if the element of the first vector is greater than
/// the corresponding element of the second vector.
///
/// The comparison yields 0 for false, 0xFF for true.
#[inline(always)]
#[target_feature = "+mmx"]
#[cfg_attr(test, assert_instr(pcmpgtb))]
pub unsafe fn _mm_cmpgt_pi8(a: __m64, b: __m64) -> __m64 {
    pcmpgtb(a, b)
}

/// Compares the 16-bit integer elements of two 64-bit integer vectors of
/// [4 x i16] to determine if the element of the first vector is greater than
/// the corresponding element of the second vector.
///
/// The comparison yields 0 for false, 0xFFFF for true.
#[inline(always)]
#[target_feature = "+mmx"]
#[cfg_attr(test, assert_instr(pcmpgtw))]
pub unsafe fn _mm_cmpgt_pi16(a: i16x4, b: i16x4) -> i16x4 {
    mem::transmute(pcmpgtw(mem::transmute(a), mem::transmute(b)))
}

/// Unpacks the upper 32 bits from two 64-bit integer vectors of
/// [4 x i16] and interleaves them into a 64-bit integer vector of [4 x i16].
#[inline(always)]
#[target_feature = "+mmx"]
#[cfg_attr(test, assert_instr(punpckhwd))] // FIXME punpcklbw expected
pub unsafe fn _mm_unpackhi_pi16(a: i16x4, b: i16x4) -> i16x4 {
    mem::transmute(punpckhwd(mem::transmute(a), mem::transmute(b)))
}

/// Unpacks the lower 32 bits from two 64-bit integer vectors of [8 x i8]
/// and interleaves them into a 64-bit integer vector of [8 x i8].
#[inline(always)]
#[target_feature = "+mmx"]
#[cfg_attr(test, assert_instr(punpcklbw))]
pub unsafe fn _mm_unpacklo_pi8(a: __m64, b: __m64) -> __m64 {
    punpcklbw(a, b)
}

/// Unpacks the lower 32 bits from two 64-bit integer vectors of
/// [4 x i16] and interleaves them into a 64-bit integer vector of [4 x i16].
#[inline(always)]
#[target_feature = "+mmx"]
#[cfg_attr(test, assert_instr(punpcklwd))]
pub unsafe fn _mm_unpacklo_pi16(a: i16x4, b: i16x4) -> i16x4 {
    mem::transmute(punpcklwd(mem::transmute(a), mem::transmute(b)))
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.mmx.packsswb"]
    fn packsswb(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.packssdw"]
    fn packssdw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pcmpgt.b"]
    fn pcmpgtb(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pcmpgt.w"]
    fn pcmpgtw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.punpckhwd"]
    fn punpckhwd(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.punpcklbw"]
    fn punpcklbw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.punpcklwd"]
    fn punpcklwd(a: __m64, b: __m64) -> __m64;
}

#[cfg(test)]
mod tests {
    use v64::{__m64, i16x4, i32x2, i8x8};
    use x86::i686::mmx;
    use stdsimd_test::simd_test;

    #[simd_test = "sse"] // FIXME: should be mmx
    unsafe fn _mm_setzero_si64() {
        let r: __m64 = ::std::mem::transmute(0_i64);
        assert_eq!(r, mmx::_mm_setzero_si64());
    }

    #[simd_test = "sse"] // FIXME: should be mmx
    unsafe fn _mm_packs_pi16() {
        let a = i16x4::new(-1, 2, -3, 4);
        let b = i16x4::new(-5, 6, -7, 8);
        let r = i8x8::new(-1, 2, -3, 4, -5, 6, -7, 8);
        assert_eq!(r, mmx::_mm_packs_pi16(a, b));
    }

    #[simd_test = "sse"] // FIXME: should be mmx
    unsafe fn _mm_packs_pi32() {
        let a = i32x2::new(-1, 2);
        let b = i32x2::new(-5, 6);
        let r = i16x4::new(-1, 2, -5, 6);
        assert_eq!(r, mmx::_mm_packs_pi32(a, b));
    }

    #[simd_test = "mmx"]
    unsafe fn _mm_cmpgt_pi8() {
        let a = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = i8x8::new(8, 7, 6, 5, 4, 3, 2, 1);
        let r = i8x8::new(0, 0, 0, 0, 0, -1, -1, -1);
        assert_eq!(r, i8x8::from(mmx::_mm_cmpgt_pi8(a.into(), b.into())));
    }

    #[simd_test = "mmx"]
    unsafe fn _mm_cmpgt_pi16() {
        let a = i16x4::new(0, 1, 2, 3);
        let b = i16x4::new(4, 3, 2, 1);
        let r = i16x4::new(0, 0, 0, -1);
        assert_eq!(r, mmx::_mm_cmpgt_pi16(a, b));
    }

    #[simd_test = "mmx"]
    unsafe fn _mm_unpackhi_pi16() {
        let a = i16x4::new(0, 1, 2, 3);
        let b = i16x4::new(4, 5, 6, 7);
        let r = i16x4::new(2, 6, 3, 7);
        assert_eq!(r, mmx::_mm_unpackhi_pi16(a, b));
    }

    #[simd_test = "mmx"]
    unsafe fn _mm_unpacklo_pi8() {
        let a = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = i8x8::new(8, 9, 10, 11, 12, 13, 14, 15);
        let r = i8x8::new(0, 8, 1, 9, 2, 10, 3, 11);
        assert_eq!(r, i8x8::from(mmx::_mm_unpacklo_pi8(a.into(), b.into())));
    }

    #[simd_test = "mmx"]
    unsafe fn _mm_unpacklo_pi16() {
        let a = i16x4::new(0, 1, 2, 3);
        let b = i16x4::new(4, 5, 6, 7);
        let r = i16x4::new(0, 4, 1, 5);
        assert_eq!(r, mmx::_mm_unpacklo_pi16(a, b));
    }
}
