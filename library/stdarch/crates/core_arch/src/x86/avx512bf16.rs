//! [AVX512BF16 intrinsics].
//!
//! [AVX512BF16 intrinsics]: https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769&avx512techs=AVX512_BF16

use crate::arch::asm;
use crate::core_arch::{simd::*, x86::*};
use crate::intrinsics::simd::*;

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.x86.avx512bf16.cvtne2ps2bf16.128"]
    fn cvtne2ps2bf16(a: f32x4, b: f32x4) -> i16x8;
    #[link_name = "llvm.x86.avx512bf16.cvtne2ps2bf16.256"]
    fn cvtne2ps2bf16_256(a: f32x8, b: f32x8) -> i16x16;
    #[link_name = "llvm.x86.avx512bf16.cvtne2ps2bf16.512"]
    fn cvtne2ps2bf16_512(a: f32x16, b: f32x16) -> i16x32;
    #[link_name = "llvm.x86.avx512bf16.cvtneps2bf16.256"]
    fn cvtneps2bf16_256(a: f32x8) -> i16x8;
    #[link_name = "llvm.x86.avx512bf16.cvtneps2bf16.512"]
    fn cvtneps2bf16_512(a: f32x16) -> i16x16;
    #[link_name = "llvm.x86.avx512bf16.dpbf16ps.128"]
    fn dpbf16ps(a: f32x4, b: i16x8, c: i16x8) -> f32x4;
    #[link_name = "llvm.x86.avx512bf16.dpbf16ps.256"]
    fn dpbf16ps_256(a: f32x8, b: i16x16, c: i16x16) -> f32x8;
    #[link_name = "llvm.x86.avx512bf16.dpbf16ps.512"]
    fn dpbf16ps_512(a: f32x16, b: i16x32, c: i16x32) -> f32x16;
}

/// Convert packed single-precision (32-bit) floating-point elements in two 128-bit vectors
/// a and b to packed BF16 (16-bit) floating-point elements, and store the results in a
/// 128-bit wide vector.
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651&avx512techs=AVX512_BF16&text=_mm_cvtne2ps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtne2ps2bf16"))]
pub fn _mm_cvtne2ps_pbh(a: __m128, b: __m128) -> __m128bh {
    unsafe { transmute(cvtne2ps2bf16(a.as_f32x4(), b.as_f32x4())) }
}

/// Convert packed single-precision (32-bit) floating-point elements in two vectors
/// a and b to packed BF16 (16-bit) floating-point elements, and store the results
/// in single vector dst using writemask k (elements are copied from src when the
/// corresponding mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651&avx512techs=AVX512_BF16&text=_mm_mask_cvtne2ps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtne2ps2bf16"))]
pub fn _mm_mask_cvtne2ps_pbh(src: __m128bh, k: __mmask8, a: __m128, b: __m128) -> __m128bh {
    unsafe {
        let cvt = _mm_cvtne2ps_pbh(a, b).as_u16x8();
        transmute(simd_select_bitmask(k, cvt, src.as_u16x8()))
    }
}

/// Convert packed single-precision (32-bit) floating-point elements in two vectors
/// a and b to packed BF16 (16-bit) floating-point elements, and store the results
/// in single vector dst using zeromask k (elements are zeroed out when the corresponding
/// mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651&avx512techs=AVX512_BF16&text=_mm_maskz_cvtne2ps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtne2ps2bf16"))]
pub fn _mm_maskz_cvtne2ps_pbh(k: __mmask8, a: __m128, b: __m128) -> __m128bh {
    unsafe {
        let cvt = _mm_cvtne2ps_pbh(a, b).as_u16x8();
        transmute(simd_select_bitmask(k, cvt, u16x8::ZERO))
    }
}

/// Convert packed single-precision (32-bit) floating-point elements in two 256-bit vectors
/// a and b to packed BF16 (16-bit) floating-point elements, and store the results in a
/// 256-bit wide vector.
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654&avx512techs=AVX512_BF16&text=_mm256_cvtne2ps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtne2ps2bf16"))]
pub fn _mm256_cvtne2ps_pbh(a: __m256, b: __m256) -> __m256bh {
    unsafe { transmute(cvtne2ps2bf16_256(a.as_f32x8(), b.as_f32x8())) }
}

/// Convert packed single-precision (32-bit) floating-point elements in two vectors a and b
/// to packed BF16 (16-bit) floating-point elements and store the results in single vector
/// dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654&avx512techs=AVX512_BF16&text=_mm256_mask_cvtne2ps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtne2ps2bf16"))]
pub fn _mm256_mask_cvtne2ps_pbh(src: __m256bh, k: __mmask16, a: __m256, b: __m256) -> __m256bh {
    unsafe {
        let cvt = _mm256_cvtne2ps_pbh(a, b).as_u16x16();
        transmute(simd_select_bitmask(k, cvt, src.as_u16x16()))
    }
}

/// Convert packed single-precision (32-bit) floating-point elements in two vectors a and b
/// to packed BF16 (16-bit) floating-point elements, and store the results in single vector
/// dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654&avx512techs=AVX512_BF16&text=_mm256_maskz_cvtne2ps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtne2ps2bf16"))]
pub fn _mm256_maskz_cvtne2ps_pbh(k: __mmask16, a: __m256, b: __m256) -> __m256bh {
    unsafe {
        let cvt = _mm256_cvtne2ps_pbh(a, b).as_u16x16();
        transmute(simd_select_bitmask(k, cvt, u16x16::ZERO))
    }
}

/// Convert packed single-precision (32-bit) floating-point elements in two 512-bit vectors
/// a and b to packed BF16 (16-bit) floating-point elements, and store the results in a
/// 512-bit wide vector.
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657&avx512techs=AVX512_BF16&text=_mm512_cvtne2ps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtne2ps2bf16"))]
pub fn _mm512_cvtne2ps_pbh(a: __m512, b: __m512) -> __m512bh {
    unsafe { transmute(cvtne2ps2bf16_512(a.as_f32x16(), b.as_f32x16())) }
}

/// Convert packed single-precision (32-bit) floating-point elements in two vectors
/// a and b to packed BF16 (16-bit) floating-point elements, and store the results
/// in single vector dst using writemask k (elements are copied from src when the
/// corresponding mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657&avx512techs=AVX512_BF16&text=_mm512_mask_cvtne2ps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtne2ps2bf16"))]
pub fn _mm512_mask_cvtne2ps_pbh(src: __m512bh, k: __mmask32, a: __m512, b: __m512) -> __m512bh {
    unsafe {
        let cvt = _mm512_cvtne2ps_pbh(a, b).as_u16x32();
        transmute(simd_select_bitmask(k, cvt, src.as_u16x32()))
    }
}

/// Convert packed single-precision (32-bit) floating-point elements in two vectors
/// a and b to packed BF16 (16-bit) floating-point elements, and store the results
/// in single vector dst using zeromask k (elements are zeroed out when the corresponding
/// mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657&avx512techs=AVX512_BF16&text=_mm512_maskz_cvtne2ps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtne2ps2bf16"))]
pub fn _mm512_maskz_cvtne2ps_pbh(k: __mmask32, a: __m512, b: __m512) -> __m512bh {
    unsafe {
        let cvt = _mm512_cvtne2ps_pbh(a, b).as_u16x32();
        transmute(simd_select_bitmask(k, cvt, u16x32::ZERO))
    }
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed BF16 (16-bit)
/// floating-point elements, and store the results in dst.
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm256_cvtneps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtneps2bf16"))]
pub fn _mm256_cvtneps_pbh(a: __m256) -> __m128bh {
    unsafe { transmute(cvtneps2bf16_256(a.as_f32x8())) }
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed BF16 (16-bit)
/// floating-point elements, and store the results in dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm256_mask_cvtneps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtneps2bf16"))]
pub fn _mm256_mask_cvtneps_pbh(src: __m128bh, k: __mmask8, a: __m256) -> __m128bh {
    unsafe {
        let cvt = _mm256_cvtneps_pbh(a).as_u16x8();
        transmute(simd_select_bitmask(k, cvt, src.as_u16x8()))
    }
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed BF16 (16-bit)
/// floating-point elements, and store the results in dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm256_maskz_cvtneps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtneps2bf16"))]
pub fn _mm256_maskz_cvtneps_pbh(k: __mmask8, a: __m256) -> __m128bh {
    unsafe {
        let cvt = _mm256_cvtneps_pbh(a).as_u16x8();
        transmute(simd_select_bitmask(k, cvt, u16x8::ZERO))
    }
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed BF16 (16-bit)
/// floating-point elements, and store the results in dst.
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm512_cvtneps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtneps2bf16"))]
pub fn _mm512_cvtneps_pbh(a: __m512) -> __m256bh {
    unsafe { transmute(cvtneps2bf16_512(a.as_f32x16())) }
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed BF16 (16-bit)
/// floating-point elements, and store the results in dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm512_mask_cvtneps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtneps2bf16"))]
pub fn _mm512_mask_cvtneps_pbh(src: __m256bh, k: __mmask16, a: __m512) -> __m256bh {
    unsafe {
        let cvt = _mm512_cvtneps_pbh(a).as_u16x16();
        transmute(simd_select_bitmask(k, cvt, src.as_u16x16()))
    }
}

/// Convert packed single-precision (32-bit) floating-point elements in a to packed BF16 (16-bit)
/// floating-point elements, and store the results in dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm512_maskz_cvtneps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vcvtneps2bf16"))]
pub fn _mm512_maskz_cvtneps_pbh(k: __mmask16, a: __m512) -> __m256bh {
    unsafe {
        let cvt = _mm512_cvtneps_pbh(a).as_u16x16();
        transmute(simd_select_bitmask(k, cvt, u16x16::ZERO))
    }
}

/// Compute dot-product of BF16 (16-bit) floating-point pairs in a and b,
/// accumulating the intermediate single-precision (32-bit) floating-point elements
/// with elements in src, and store the results in dst.
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm_dpbf16_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vdpbf16ps"))]
pub fn _mm_dpbf16_ps(src: __m128, a: __m128bh, b: __m128bh) -> __m128 {
    unsafe { transmute(dpbf16ps(src.as_f32x4(), a.as_i16x8(), b.as_i16x8())) }
}

/// Compute dot-product of BF16 (16-bit) floating-point pairs in a and b,
/// accumulating the intermediate single-precision (32-bit) floating-point elements
/// with elements in src, and store the results in dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm_mask_dpbf16_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vdpbf16ps"))]
pub fn _mm_mask_dpbf16_ps(src: __m128, k: __mmask8, a: __m128bh, b: __m128bh) -> __m128 {
    unsafe {
        let rst = _mm_dpbf16_ps(src, a, b).as_f32x4();
        transmute(simd_select_bitmask(k, rst, src.as_f32x4()))
    }
}

/// Compute dot-product of BF16 (16-bit) floating-point pairs in a and b,
/// accumulating the intermediate single-precision (32-bit) floating-point elements
/// with elements in src, and store the results in dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm_maskz_dpbf16_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vdpbf16ps"))]
pub fn _mm_maskz_dpbf16_ps(k: __mmask8, src: __m128, a: __m128bh, b: __m128bh) -> __m128 {
    unsafe {
        let rst = _mm_dpbf16_ps(src, a, b).as_f32x4();
        let zero = _mm_set1_ps(0.0_f32).as_f32x4();
        transmute(simd_select_bitmask(k, rst, zero))
    }
}

/// Compute dot-product of BF16 (16-bit) floating-point pairs in a and b,
/// accumulating the intermediate single-precision (32-bit) floating-point elements
/// with elements in src, and store the results in dst.
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm256_dpbf16_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vdpbf16ps"))]
pub fn _mm256_dpbf16_ps(src: __m256, a: __m256bh, b: __m256bh) -> __m256 {
    unsafe { transmute(dpbf16ps_256(src.as_f32x8(), a.as_i16x16(), b.as_i16x16())) }
}

/// Compute dot-product of BF16 (16-bit) floating-point pairs in a and b,
/// accumulating the intermediate single-precision (32-bit) floating-point elements
/// with elements in src, and store the results in dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm256_mask_dpbf16_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vdpbf16ps"))]
pub fn _mm256_mask_dpbf16_ps(src: __m256, k: __mmask8, a: __m256bh, b: __m256bh) -> __m256 {
    unsafe {
        let rst = _mm256_dpbf16_ps(src, a, b).as_f32x8();
        transmute(simd_select_bitmask(k, rst, src.as_f32x8()))
    }
}

/// Compute dot-product of BF16 (16-bit) floating-point pairs in a and b,
/// accumulating the intermediate single-precision (32-bit) floating-point elements
/// with elements in src, and store the results in dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm256_maskz_dpbf16_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vdpbf16ps"))]
pub fn _mm256_maskz_dpbf16_ps(k: __mmask8, src: __m256, a: __m256bh, b: __m256bh) -> __m256 {
    unsafe {
        let rst = _mm256_dpbf16_ps(src, a, b).as_f32x8();
        transmute(simd_select_bitmask(k, rst, f32x8::ZERO))
    }
}

/// Compute dot-product of BF16 (16-bit) floating-point pairs in a and b,
/// accumulating the intermediate single-precision (32-bit) floating-point elements
/// with elements in src, and store the results in dst.Compute dot-product of BF16 (16-bit)
/// floating-point pairs in a and b, accumulating the intermediate single-precision (32-bit)
/// floating-point elements with elements in src, and store the results in dst.
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm512_dpbf16_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vdpbf16ps"))]
pub fn _mm512_dpbf16_ps(src: __m512, a: __m512bh, b: __m512bh) -> __m512 {
    unsafe { transmute(dpbf16ps_512(src.as_f32x16(), a.as_i16x32(), b.as_i16x32())) }
}

/// Compute dot-product of BF16 (16-bit) floating-point pairs in a and b,
/// accumulating the intermediate single-precision (32-bit) floating-point elements
/// with elements in src, and store the results in dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm512_mask_dpbf16_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vdpbf16ps"))]
pub fn _mm512_mask_dpbf16_ps(src: __m512, k: __mmask16, a: __m512bh, b: __m512bh) -> __m512 {
    unsafe {
        let rst = _mm512_dpbf16_ps(src, a, b).as_f32x16();
        transmute(simd_select_bitmask(k, rst, src.as_f32x16()))
    }
}

/// Compute dot-product of BF16 (16-bit) floating-point pairs in a and b,
/// accumulating the intermediate single-precision (32-bit) floating-point elements
/// with elements in src, and store the results in dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=1769,1651,1654,1657,1660&avx512techs=AVX512_BF16&text=_mm512_maskz_dpbf16_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr("vdpbf16ps"))]
pub fn _mm512_maskz_dpbf16_ps(k: __mmask16, src: __m512, a: __m512bh, b: __m512bh) -> __m512 {
    unsafe {
        let rst = _mm512_dpbf16_ps(src, a, b).as_f32x16();
        transmute(simd_select_bitmask(k, rst, f32x16::ZERO))
    }
}

/// Converts packed BF16 (16-bit) floating-point elements in a to packed single-precision (32-bit)
/// floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cvtpbh_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _mm512_cvtpbh_ps(a: __m256bh) -> __m512 {
    unsafe { _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepi16_epi32(transmute(a)))) }
}

/// Converts packed BF16 (16-bit) floating-point elements in a to packed single-precision (32-bit)
/// floating-point elements, and store the results in dst using writemask k (elements are copied
/// from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cvtpbh_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _mm512_mask_cvtpbh_ps(src: __m512, k: __mmask16, a: __m256bh) -> __m512 {
    unsafe {
        let cvt = _mm512_cvtpbh_ps(a);
        transmute(simd_select_bitmask(k, cvt.as_f32x16(), src.as_f32x16()))
    }
}

/// Converts packed BF16 (16-bit) floating-point elements in a to packed single-precision (32-bit)
/// floating-point elements, and store the results in dst using zeromask k (elements are zeroed out
/// when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_cvtpbh_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _mm512_maskz_cvtpbh_ps(k: __mmask16, a: __m256bh) -> __m512 {
    unsafe {
        let cvt = _mm512_cvtpbh_ps(a);
        transmute(simd_select_bitmask(k, cvt.as_f32x16(), f32x16::ZERO))
    }
}

/// Converts packed BF16 (16-bit) floating-point elements in a to packed single-precision (32-bit)
/// floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtpbh_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _mm256_cvtpbh_ps(a: __m128bh) -> __m256 {
    unsafe { _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepi16_epi32(transmute(a)))) }
}

/// Converts packed BF16 (16-bit) floating-point elements in a to packed single-precision (32-bit)
/// floating-point elements, and store the results in dst using writemask k (elements are copied
/// from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_cvtpbh_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _mm256_mask_cvtpbh_ps(src: __m256, k: __mmask8, a: __m128bh) -> __m256 {
    unsafe {
        let cvt = _mm256_cvtpbh_ps(a);
        transmute(simd_select_bitmask(k, cvt.as_f32x8(), src.as_f32x8()))
    }
}

/// Converts packed BF16 (16-bit) floating-point elements in a to packed single-precision (32-bit)
/// floating-point elements, and store the results in dst using zeromask k (elements are zeroed out
/// when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_cvtpbh_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _mm256_maskz_cvtpbh_ps(k: __mmask8, a: __m128bh) -> __m256 {
    unsafe {
        let cvt = _mm256_cvtpbh_ps(a);
        transmute(simd_select_bitmask(k, cvt.as_f32x8(), f32x8::ZERO))
    }
}

/// Converts packed BF16 (16-bit) floating-point elements in a to single-precision (32-bit) floating-point
/// elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtpbh_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _mm_cvtpbh_ps(a: __m128bh) -> __m128 {
    unsafe { _mm_castsi128_ps(_mm_slli_epi32::<16>(_mm_cvtepi16_epi32(transmute(a)))) }
}

/// Converts packed BF16 (16-bit) floating-point elements in a to single-precision (32-bit) floating-point
/// elements, and store the results in dst using writemask k (elements are copied from src when the corresponding
/// mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cvtpbh_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _mm_mask_cvtpbh_ps(src: __m128, k: __mmask8, a: __m128bh) -> __m128 {
    unsafe {
        let cvt = _mm_cvtpbh_ps(a);
        transmute(simd_select_bitmask(k, cvt.as_f32x4(), src.as_f32x4()))
    }
}

/// Converts packed BF16 (16-bit) floating-point elements in a to single-precision (32-bit) floating-point
/// elements, and store the results in dst using zeromask k (elements are zeroed out when the corresponding
/// mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_cvtpbh_ps)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _mm_maskz_cvtpbh_ps(k: __mmask8, a: __m128bh) -> __m128 {
    unsafe {
        let cvt = _mm_cvtpbh_ps(a);
        transmute(simd_select_bitmask(k, cvt.as_f32x4(), f32x4::ZERO))
    }
}

/// Converts a single BF16 (16-bit) floating-point element in a to a single-precision (32-bit) floating-point
/// element, and store the result in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtsbh_ss)
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
#[unstable(feature = "stdarch_x86_avx512_bf16", issue = "127356")]
pub fn _mm_cvtsbh_ss(a: bf16) -> f32 {
    f32::from_bits((a.to_bits() as u32) << 16)
}

/// Converts packed single-precision (32-bit) floating-point elements in a to packed BF16 (16-bit)
/// floating-point elements, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtneps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[cfg_attr(test, assert_instr("vcvtneps2bf16"))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _mm_cvtneps_pbh(a: __m128) -> __m128bh {
    unsafe {
        let mut dst: __m128bh;
        asm!(
            "vcvtneps2bf16 {dst}, {src}",
            dst = lateout(xmm_reg) dst,
            src = in(xmm_reg) a,
            options(pure, nomem, nostack, preserves_flags)
        );
        dst
    }
}

/// Converts packed single-precision (32-bit) floating-point elements in a to packed BF16 (16-bit)
/// floating-point elements, and store the results in dst using writemask k (elements are copied
/// from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_cvtneps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[cfg_attr(test, assert_instr("vcvtneps2bf16"))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _mm_mask_cvtneps_pbh(src: __m128bh, k: __mmask8, a: __m128) -> __m128bh {
    unsafe {
        let mut dst = src;
        asm!(
            "vcvtneps2bf16 {dst}{{{k}}},{src}",
            dst = inlateout(xmm_reg) dst,
            src = in(xmm_reg) a,
            k = in(kreg) k,
            options(pure, nomem, nostack, preserves_flags)
        );
        dst
    }
}

/// Converts packed single-precision (32-bit) floating-point elements in a to packed BF16 (16-bit)
/// floating-point elements, and store the results in dst using zeromask k (elements are zeroed out
/// when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_cvtneps_pbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[cfg_attr(test, assert_instr("vcvtneps2bf16"))]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub fn _mm_maskz_cvtneps_pbh(k: __mmask8, a: __m128) -> __m128bh {
    unsafe {
        let mut dst: __m128bh;
        asm!(
            "vcvtneps2bf16 {dst}{{{k}}}{{z}},{src}",
            dst = lateout(xmm_reg) dst,
            src = in(xmm_reg) a,
            k = in(kreg) k,
            options(pure, nomem, nostack, preserves_flags)
        );
        dst
    }
}

/// Converts a single-precision (32-bit) floating-point element in a to a BF16 (16-bit) floating-point
/// element, and store the result in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtness_sbh)
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_bf16", issue = "127356")]
pub fn _mm_cvtness_sbh(a: f32) -> bf16 {
    unsafe {
        let value: u16 = simd_extract!(_mm_cvtneps_pbh(_mm_set_ss(a)), 0);
        bf16::from_bits(value)
    }
}

#[cfg(test)]
mod tests {
    use crate::core_arch::simd::u16x4;
    use crate::{
        core_arch::x86::*,
        mem::{transmute, transmute_copy},
    };
    use stdarch_test::simd_test;

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm_cvtne2ps_pbh() {
        let a_array = [178.125_f32, 10.5_f32, 3.75_f32, 50.25_f32];
        let b_array = [-178.125_f32, -10.5_f32, -3.75_f32, -50.25_f32];
        let a: __m128 = transmute(a_array);
        let b: __m128 = transmute(b_array);
        let c: __m128bh = _mm_cvtne2ps_pbh(a, b);
        let result: [u16; 8] = transmute(c.as_u16x8());
        #[rustfmt::skip]
        let expected_result: [u16; 8] = [
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
        ];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm_mask_cvtne2ps_pbh() {
        let a_array = [178.125_f32, 10.5_f32, 3.75_f32, 50.25_f32];
        let b_array = [-178.125_f32, -10.5_f32, -3.75_f32, -50.25_f32];
        #[rustfmt::skip]
        let src_array: [u16; 8] = [
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
        ];
        let src: __m128bh = transmute(src_array);
        let a: __m128 = transmute(a_array);
        let b: __m128 = transmute(b_array);
        let k: __mmask8 = 0b1111_1111;
        let c: __m128bh = _mm_mask_cvtne2ps_pbh(src, k, a, b);
        let result: [u16; 8] = transmute(c.as_u16x8());
        #[rustfmt::skip]
        let expected_result: [u16; 8] = [
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
        ];
        assert_eq!(result, expected_result);
        let k = 0b0000_0000;
        let c = _mm_mask_cvtne2ps_pbh(src, k, a, b);
        let result: [u16; 8] = transmute(c.as_u16x8());
        let expected_result = src_array;
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm_maskz_cvtne2ps_pbh() {
        let a_array = [178.125_f32, 10.5_f32, 3.75_f32, 50.25_f32];
        let b_array = [-178.125_f32, -10.5_f32, -3.75_f32, -50.25_f32];
        let a: __m128 = transmute(a_array);
        let b: __m128 = transmute(b_array);
        let k: __mmask8 = 0b1111_1111;
        let c: __m128bh = _mm_maskz_cvtne2ps_pbh(k, a, b);
        let result: [u16; 8] = transmute(c.as_u16x8());
        #[rustfmt::skip]
        let expected_result: [u16; 8] = [
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
        ];
        assert_eq!(result, expected_result);
        let k = 0b0011_1100;
        let c = _mm_maskz_cvtne2ps_pbh(k, a, b);
        let result: [u16; 8] = transmute(c.as_u16x8());
        #[rustfmt::skip]
        let expected_result: [u16; 8] = [
            0,
            0,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0,
            0,
        ];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm256_cvtne2ps_pbh() {
        #[rustfmt::skip]
        let a_array = [
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
        ];
        let b_array = [
            -178.125_f32,
            -10.5_f32,
            -3.75_f32,
            -50.25_f32,
            -16.5_f32,
            -255.11_f32,
            -1000.158_f32,
            -575.575_f32,
        ];
        let a: __m256 = transmute(a_array);
        let b: __m256 = transmute(b_array);
        let c: __m256bh = _mm256_cvtne2ps_pbh(a, b);
        let result: [u16; 16] = transmute(c.as_u16x16());
        #[rustfmt::skip]
        let expected_result: [u16; 16] = [
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b1_10000011_0000100,
            0b1_10000110_1111111,
            0b1_10001000_1111010,
            0b1_10001000_0010000,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
        ];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm256_mask_cvtne2ps_pbh() {
        #[rustfmt::skip]
        let a_array = [
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
        ];
        let b_array = [
            -178.125_f32,
            -10.5_f32,
            -3.75_f32,
            -50.25_f32,
            -16.5_f32,
            -255.11_f32,
            -1000.158_f32,
            -575.575_f32,
        ];
        let src_array: [u16; 16] = [
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
        ];
        let src: __m256bh = transmute(src_array);
        let a: __m256 = transmute(a_array);
        let b: __m256 = transmute(b_array);
        let k: __mmask16 = 0xffff;
        let c: __m256bh = _mm256_mask_cvtne2ps_pbh(src, k, a, b);
        let result: [u16; 16] = transmute(c.as_u16x16());
        #[rustfmt::skip]
        let expected_result: [u16; 16] = [
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b1_10000011_0000100,
            0b1_10000110_1111111,
            0b1_10001000_1111010,
            0b1_10001000_0010000,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask16 = 0;
        let c: __m256bh = _mm256_mask_cvtne2ps_pbh(src, k, a, b);
        let result: [u16; 16] = transmute(c.as_u16x16());
        let expected_result = src_array;
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm256_maskz_cvtne2ps_pbh() {
        #[rustfmt::skip]
        let a_array = [
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
        ];
        let b_array = [
            -178.125_f32,
            -10.5_f32,
            -3.75_f32,
            -50.25_f32,
            -16.5_f32,
            -255.11_f32,
            -1000.158_f32,
            -575.575_f32,
        ];
        let a: __m256 = transmute(a_array);
        let b: __m256 = transmute(b_array);
        let k: __mmask16 = 0xffff;
        let c: __m256bh = _mm256_maskz_cvtne2ps_pbh(k, a, b);
        let result: [u16; 16] = transmute(c.as_u16x16());
        #[rustfmt::skip]
        let expected_result: [u16; 16] = [
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b1_10000011_0000100,
            0b1_10000110_1111111,
            0b1_10001000_1111010,
            0b1_10001000_0010000,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask16 = 0b0110_1100_0011_0110;
        let c: __m256bh = _mm256_maskz_cvtne2ps_pbh(k, a, b);
        let result: [u16; 16] = transmute(c.as_u16x16());
        #[rustfmt::skip]
        let expected_result: [u16; 16] = [
            0,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0,
            0b1_10000011_0000100,
            0b1_10000110_1111111,
            0,
            0,
            0,
            0,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0,
        ];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512f")]
    unsafe fn test_mm512_cvtne2ps_pbh() {
        #[rustfmt::skip]
        let a_array = [
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
        ];
        let b_array = [
            -178.125_f32,
            -10.5_f32,
            -3.75_f32,
            -50.25_f32,
            -16.5_f32,
            -255.11_f32,
            -1000.158_f32,
            -575.575_f32,
            -178.125_f32,
            -10.5_f32,
            -3.75_f32,
            -50.25_f32,
            -16.5_f32,
            -255.11_f32,
            -1000.158_f32,
            -575.575_f32,
        ];
        let a: __m512 = transmute(a_array);
        let b: __m512 = transmute(b_array);
        let c: __m512bh = _mm512_cvtne2ps_pbh(a, b);
        let result: [u16; 32] = transmute(c.as_u16x32());
        #[rustfmt::skip]
        let expected_result: [u16; 32] = [
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b1_10000011_0000100,
            0b1_10000110_1111111,
            0b1_10001000_1111010,
            0b1_10001000_0010000,
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b1_10000011_0000100,
            0b1_10000110_1111111,
            0b1_10001000_1111010,
            0b1_10001000_0010000,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
        ];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512f")]
    unsafe fn test_mm512_mask_cvtne2ps_pbh() {
        #[rustfmt::skip]
        let a_array = [
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
        ];
        let b_array = [
            -178.125_f32,
            -10.5_f32,
            -3.75_f32,
            -50.25_f32,
            -16.5_f32,
            -255.11_f32,
            -1000.158_f32,
            -575.575_f32,
            -178.125_f32,
            -10.5_f32,
            -3.75_f32,
            -50.25_f32,
            -16.5_f32,
            -255.11_f32,
            -1000.158_f32,
            -575.575_f32,
        ];
        let src_array: [u16; 32] = [
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
        ];
        let src: __m512bh = transmute(src_array);
        let a: __m512 = transmute(a_array);
        let b: __m512 = transmute(b_array);
        let k: __mmask32 = 0xffffffff;
        let c: __m512bh = _mm512_mask_cvtne2ps_pbh(src, k, a, b);
        let result: [u16; 32] = transmute(c.as_u16x32());
        #[rustfmt::skip]
        let expected_result: [u16; 32] = [
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b1_10000011_0000100,
            0b1_10000110_1111111,
            0b1_10001000_1111010,
            0b1_10001000_0010000,
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b1_10000011_0000100,
            0b1_10000110_1111111,
            0b1_10001000_1111010,
            0b1_10001000_0010000,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask32 = 0;
        let c: __m512bh = _mm512_mask_cvtne2ps_pbh(src, k, a, b);
        let result: [u16; 32] = transmute(c.as_u16x32());
        let expected_result = src_array;
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512f")]
    unsafe fn test_mm512_maskz_cvtne2ps_pbh() {
        #[rustfmt::skip]
        let a_array = [
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
        ];
        let b_array = [
            -178.125_f32,
            -10.5_f32,
            -3.75_f32,
            -50.25_f32,
            -16.5_f32,
            -255.11_f32,
            -1000.158_f32,
            -575.575_f32,
            -178.125_f32,
            -10.5_f32,
            -3.75_f32,
            -50.25_f32,
            -16.5_f32,
            -255.11_f32,
            -1000.158_f32,
            -575.575_f32,
        ];
        let a: __m512 = transmute(a_array);
        let b: __m512 = transmute(b_array);
        let k: __mmask32 = 0xffffffff;
        let c: __m512bh = _mm512_maskz_cvtne2ps_pbh(k, a, b);
        let result: [u16; 32] = transmute(c.as_u16x32());
        #[rustfmt::skip]
        let expected_result: [u16; 32] = [
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b1_10000011_0000100,
            0b1_10000110_1111111,
            0b1_10001000_1111010,
            0b1_10001000_0010000,
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b1_10000011_0000100,
            0b1_10000110_1111111,
            0b1_10001000_1111010,
            0b1_10001000_0010000,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask32 = 0b1100_1010_1001_0110_1010_0011_0101_0110;
        let c: __m512bh = _mm512_maskz_cvtne2ps_pbh(k, a, b);
        let result: [u16; 32] = transmute(c.as_u16x32());
        #[rustfmt::skip]
        let expected_result: [u16; 32] = [
            0,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0,
            0b1_10000011_0000100,
            0,
            0b1_10001000_1111010,
            0,
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0,
            0,
            0,
            0b1_10000110_1111111,
            0,
            0b1_10001000_0010000,
            0,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0,
            0b0_10000011_0000100,
            0,
            0,
            0b0_10001000_0010000,
            0,
            0b0_10000010_0101000,
            0,
            0b0_10000100_1001001,
            0,
            0,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
        ];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm256_cvtneps_pbh() {
        #[rustfmt::skip]
        let a_array = [
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
        ];
        let a: __m256 = transmute(a_array);
        let c: __m128bh = _mm256_cvtneps_pbh(a);
        let result: [u16; 8] = transmute(c.as_u16x8());
        #[rustfmt::skip]
        let expected_result: [u16; 8] = [
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
        ];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm256_mask_cvtneps_pbh() {
        #[rustfmt::skip]
        let a_array = [
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
        ];
        let src_array: [u16; 8] = [
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b1_10000011_0000100,
            0b1_10000110_1111111,
            0b1_10001000_1111010,
            0b1_10001000_0010000,
        ];
        let src: __m128bh = transmute(src_array);
        let a: __m256 = transmute(a_array);
        let k: __mmask8 = 0xff;
        let b = _mm256_mask_cvtneps_pbh(src, k, a);
        let result: [u16; 8] = transmute(b.as_u16x8());
        #[rustfmt::skip]
        let expected_result: [u16; 8] = [
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask8 = 0x0;
        let b: __m128bh = _mm256_mask_cvtneps_pbh(src, k, a);
        let result: [u16; 8] = transmute(b.as_u16x8());
        let expected_result: [u16; 8] = src_array;
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm256_maskz_cvtneps_pbh() {
        #[rustfmt::skip]
        let a_array = [
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
        ];
        let a: __m256 = transmute(a_array);
        let k: __mmask8 = 0xff;
        let b = _mm256_maskz_cvtneps_pbh(k, a);
        let result: [u16; 8] = transmute(b.as_u16x8());
        #[rustfmt::skip]
        let expected_result: [u16; 8] = [
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask8 = 0x6;
        let b: __m128bh = _mm256_maskz_cvtneps_pbh(k, a);
        let result: [u16; 8] = transmute(b.as_u16x8());
        let expected_result: [u16; 8] =
            [0, 0b0_10000010_0101000, 0b0_10000000_1110000, 0, 0, 0, 0, 0];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512f")]
    unsafe fn test_mm512_cvtneps_pbh() {
        #[rustfmt::skip]
        let a_array = [
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
        ];
        let a: __m512 = transmute(a_array);
        let c: __m256bh = _mm512_cvtneps_pbh(a);
        let result: [u16; 16] = transmute(c.as_u16x16());
        #[rustfmt::skip]
        let expected_result: [u16; 16] = [
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
        ];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512f")]
    unsafe fn test_mm512_mask_cvtneps_pbh() {
        #[rustfmt::skip]
        let a_array = [
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
        ];
        let src_array: [u16; 16] = [
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b1_10000011_0000100,
            0b1_10000110_1111111,
            0b1_10001000_1111010,
            0b1_10001000_0010000,
            0b1_10000110_0110010,
            0b1_10000010_0101000,
            0b1_10000000_1110000,
            0b1_10000100_1001001,
            0b1_10000011_0000100,
            0b1_10000110_1111111,
            0b1_10001000_1111010,
            0b1_10001000_0010000,
        ];
        let src: __m256bh = transmute(src_array);
        let a: __m512 = transmute(a_array);
        let k: __mmask16 = 0xffff;
        let c: __m256bh = _mm512_mask_cvtneps_pbh(src, k, a);
        let result: [u16; 16] = transmute(c.as_u16x16());
        #[rustfmt::skip]
        let expected_result: [u16; 16] = [
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask16 = 0;
        let c: __m256bh = _mm512_mask_cvtneps_pbh(src, k, a);
        let result: [u16; 16] = transmute(c.as_u16x16());
        let expected_result = src_array;
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512f")]
    unsafe fn test_mm512_maskz_cvtneps_pbh() {
        #[rustfmt::skip]
        let a_array = [
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
            178.125_f32,
            10.5_f32,
            3.75_f32,
            50.25_f32,
            16.5_f32,
            255.11_f32,
            1000.158_f32,
            575.575_f32,
        ];
        let a: __m512 = transmute(a_array);
        let k: __mmask16 = 0xffff;
        let c: __m256bh = _mm512_maskz_cvtneps_pbh(k, a);
        let result: [u16; 16] = transmute(c.as_u16x16());
        #[rustfmt::skip]
        let expected_result: [u16; 16] = [
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
            0b0_10000110_0110010,
            0b0_10000010_0101000,
            0b0_10000000_1110000,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0b0_10001000_0010000,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask16 = 0x653a;
        let c: __m256bh = _mm512_maskz_cvtneps_pbh(k, a);
        let result: [u16; 16] = transmute(c.as_u16x16());
        #[rustfmt::skip]
        let expected_result: [u16; 16] = [
            0,
            0b0_10000010_0101000,
            0,
            0b0_10000100_1001001,
            0b0_10000011_0000100,
            0b0_10000110_1111111,
            0,
            0,
            0b0_10000110_0110010,
            0,
            0b0_10000000_1110000,
            0,
            0,
            0b0_10000110_1111111,
            0b0_10001000_1111010,
            0,
        ];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm_dpbf16_ps() {
        let a_array = [8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32];
        let b_array = [-1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32];
        let a1: __m128 = transmute(a_array);
        let b1: __m128 = transmute(b_array);
        let src: __m128 = transmute([1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]);
        let a: __m128bh = _mm_cvtne2ps_pbh(a1, a1);
        let b: __m128bh = _mm_cvtne2ps_pbh(b1, b1);
        let c: __m128 = _mm_dpbf16_ps(src, a, b);
        let result: [f32; 4] = transmute(c.as_f32x4());
        let expected_result: [f32; 4] = [-18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm_mask_dpbf16_ps() {
        let a_array = [8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32];
        let b_array = [-1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32];
        let a1: __m128 = transmute(a_array);
        let b1: __m128 = transmute(b_array);
        let k: __mmask8 = 0xf3;
        let src: __m128 = transmute([1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]);
        let a: __m128bh = _mm_cvtne2ps_pbh(a1, a1);
        let b: __m128bh = _mm_cvtne2ps_pbh(b1, b1);
        let c: __m128 = _mm_mask_dpbf16_ps(src, k, a, b);
        let result: [f32; 4] = transmute(c.as_f32x4());
        let expected_result: [f32; 4] = [-18.0_f32, -52.0_f32, 3.0_f32, 4.0_f32];
        assert_eq!(result, expected_result);
        let k: __mmask8 = 0xff;
        let c: __m128 = _mm_mask_dpbf16_ps(src, k, a, b);
        let result: [f32; 4] = transmute(c.as_f32x4());
        let expected_result: [f32; 4] = [-18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32];
        assert_eq!(result, expected_result);
        let k: __mmask8 = 0;
        let c: __m128 = _mm_mask_dpbf16_ps(src, k, a, b);
        let result: [f32; 4] = transmute(c.as_f32x4());
        let expected_result: [f32; 4] = [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm_maskz_dpbf16_ps() {
        let a_array = [8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32];
        let b_array = [-1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32];
        let a1: __m128 = transmute(a_array);
        let b1: __m128 = transmute(b_array);
        let k: __mmask8 = 0xf3;
        let src: __m128 = transmute([1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]);
        let a: __m128bh = _mm_cvtne2ps_pbh(a1, a1);
        let b: __m128bh = _mm_cvtne2ps_pbh(b1, b1);
        let c: __m128 = _mm_maskz_dpbf16_ps(k, src, a, b);
        let result: [f32; 4] = transmute(c.as_f32x4());
        let expected_result: [f32; 4] = [-18.0_f32, -52.0_f32, 0.0, 0.0];
        assert_eq!(result, expected_result);
        let k: __mmask8 = 0xff;
        let c: __m128 = _mm_maskz_dpbf16_ps(k, src, a, b);
        let result: [f32; 4] = transmute(c.as_f32x4());
        let expected_result: [f32; 4] = [-18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32];
        assert_eq!(result, expected_result);
        let k: __mmask8 = 0;
        let c: __m128 = _mm_maskz_dpbf16_ps(k, src, a, b);
        let result: [f32; 4] = transmute(c.as_f32x4());
        let expected_result: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm256_dpbf16_ps() {
        #[rustfmt::skip]
        let a_array = [
            8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32, 8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32,
        ];
        let b_array = [
            -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32,
        ];
        let a1: __m256 = transmute(a_array);
        let b1: __m256 = transmute(b_array);
        #[rustfmt::skip]
        let src: __m256 = transmute([
            1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32,
        ]);
        let a: __m256bh = _mm256_cvtne2ps_pbh(a1, a1);
        let b: __m256bh = _mm256_cvtne2ps_pbh(b1, b1);
        let c: __m256 = _mm256_dpbf16_ps(src, a, b);
        let result: [f32; 8] = transmute(c.as_f32x8());
        #[rustfmt::skip]
        let expected_result: [f32; 8] = [
            -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32, -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32,
        ];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm256_mask_dpbf16_ps() {
        #[rustfmt::skip]
        let a_array = [
            8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32, 8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32,
        ];
        let b_array = [
            -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32,
        ];
        let a1: __m256 = transmute(a_array);
        let b1: __m256 = transmute(b_array);
        let k: __mmask8 = 0x33;
        #[rustfmt::skip]
        let src: __m256 = transmute([
            1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32,
        ]);
        let a: __m256bh = _mm256_cvtne2ps_pbh(a1, a1);
        let b: __m256bh = _mm256_cvtne2ps_pbh(b1, b1);
        let c: __m256 = _mm256_mask_dpbf16_ps(src, k, a, b);
        let result: [f32; 8] = transmute(c.as_f32x8());
        #[rustfmt::skip]
        let expected_result: [f32; 8] = [
            -18.0_f32, -52.0_f32, 3.0_f32, 4.0_f32, -18.0_f32, -52.0_f32, 3.0_f32, 4.0_f32,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask8 = 0xff;
        let c: __m256 = _mm256_mask_dpbf16_ps(src, k, a, b);
        let result: [f32; 8] = transmute(c.as_f32x8());
        #[rustfmt::skip]
        let expected_result: [f32; 8] = [
            -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32, -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask8 = 0;
        let c: __m256 = _mm256_mask_dpbf16_ps(src, k, a, b);
        let result: [f32; 8] = transmute(c.as_f32x8());
        #[rustfmt::skip]
        let expected_result: [f32; 8] = [
            1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32,
        ];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm256_maskz_dpbf16_ps() {
        #[rustfmt::skip]
        let a_array = [
            8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32, 8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32,
        ];
        let b_array = [
            -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32,
        ];
        let a1: __m256 = transmute(a_array);
        let b1: __m256 = transmute(b_array);
        let k: __mmask8 = 0x33;
        #[rustfmt::skip]
        let src: __m256 = transmute([
            1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32,
        ]);
        let a: __m256bh = _mm256_cvtne2ps_pbh(a1, a1);
        let b: __m256bh = _mm256_cvtne2ps_pbh(b1, b1);
        let c: __m256 = _mm256_maskz_dpbf16_ps(k, src, a, b);
        let result: [f32; 8] = transmute(c.as_f32x8());
        #[rustfmt::skip]
        let expected_result: [f32; 8] = [
            -18.0_f32, -52.0_f32, 0.0, 0.0, -18.0_f32, -52.0_f32, 0.0, 0.0,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask8 = 0xff;
        let c: __m256 = _mm256_maskz_dpbf16_ps(k, src, a, b);
        let result: [f32; 8] = transmute(c.as_f32x8());
        #[rustfmt::skip]
        let expected_result: [f32; 8] = [
            -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32, -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask8 = 0;
        let c: __m256 = _mm256_maskz_dpbf16_ps(k, src, a, b);
        let result: [f32; 8] = transmute(c.as_f32x8());
        let expected_result: [f32; 8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512f")]
    unsafe fn test_mm512_dpbf16_ps() {
        #[rustfmt::skip]
        let a_array = [
            8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32, 8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32,
            8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32, 8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32,
        ];
        let b_array = [
            -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32,
            -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32,
        ];
        let a1: __m512 = transmute(a_array);
        let b1: __m512 = transmute(b_array);
        let src: __m512 = transmute([
            1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32,
            2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32,
        ]);
        let a: __m512bh = _mm512_cvtne2ps_pbh(a1, a1);
        let b: __m512bh = _mm512_cvtne2ps_pbh(b1, b1);
        let c: __m512 = _mm512_dpbf16_ps(src, a, b);
        let result: [f32; 16] = transmute(c.as_f32x16());
        #[rustfmt::skip]
        let expected_result: [f32; 16] = [
            -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32, -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32,
            -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32, -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32,
        ];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512f")]
    unsafe fn test_mm512_mask_dpbf16_ps() {
        #[rustfmt::skip]
        let a_array = [
            8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32, 8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32,
            8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32, 8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32,
        ];
        let b_array = [
            -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32,
            -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32,
        ];
        let a1: __m512 = transmute(a_array);
        let b1: __m512 = transmute(b_array);
        let k: __mmask16 = 0x3333;
        #[rustfmt::skip]
        let src: __m512 = transmute([
            1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32,
            2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32,
        ]);
        let a: __m512bh = _mm512_cvtne2ps_pbh(a1, a1);
        let b: __m512bh = _mm512_cvtne2ps_pbh(b1, b1);
        let c: __m512 = _mm512_mask_dpbf16_ps(src, k, a, b);
        let result: [f32; 16] = transmute(c.as_f32x16());
        #[rustfmt::skip]
        let expected_result: [f32; 16] = [
            -18.0_f32, -52.0_f32, 3.0_f32, 4.0_f32, -18.0_f32, -52.0_f32, 3.0_f32, 4.0_f32,
            -18.0_f32, -52.0_f32, 3.0_f32, 4.0_f32, -18.0_f32, -52.0_f32, 3.0_f32, 4.0_f32,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask16 = 0xffff;
        let c: __m512 = _mm512_mask_dpbf16_ps(src, k, a, b);
        let result: [f32; 16] = transmute(c.as_f32x16());
        #[rustfmt::skip]
        let expected_result: [f32; 16] = [
            -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32, -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32,
            -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32, -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask16 = 0;
        let c: __m512 = _mm512_mask_dpbf16_ps(src, k, a, b);
        let result: [f32; 16] = transmute(c.as_f32x16());
        #[rustfmt::skip]
        let expected_result: [f32; 16] = [
            1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32,
            2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32,
        ];
        assert_eq!(result, expected_result);
    }

    #[simd_test(enable = "avx512bf16,avx512f")]
    unsafe fn test_mm512_maskz_dpbf16_ps() {
        #[rustfmt::skip]
        let a_array = [
            8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32, 8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32,
            8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32, 8.5_f32, 10.5_f32, 3.75_f32, 50.25_f32,
        ];
        let b_array = [
            -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32,
            -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32, -1.0_f32,
        ];
        let a1: __m512 = transmute(a_array);
        let b1: __m512 = transmute(b_array);
        let k: __mmask16 = 0x3333;
        #[rustfmt::skip]
        let src: __m512 = transmute([
            1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32,
            2.0_f32, 3.0_f32, 4.0_f32, 1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32,
        ]);
        let a: __m512bh = _mm512_cvtne2ps_pbh(a1, a1);
        let b: __m512bh = _mm512_cvtne2ps_pbh(b1, b1);
        let c: __m512 = _mm512_maskz_dpbf16_ps(k, src, a, b);
        let result: [f32; 16] = transmute(c.as_f32x16());
        #[rustfmt::skip]
        let expected_result: [f32; 16] = [
            -18.0_f32, -52.0_f32, 0.0, 0.0, -18.0_f32, -52.0_f32, 0.0, 0.0, -18.0_f32, -52.0_f32,
            0.0, 0.0, -18.0_f32, -52.0_f32, 0.0, 0.0,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask16 = 0xffff;
        let c: __m512 = _mm512_maskz_dpbf16_ps(k, src, a, b);
        let result: [f32; 16] = transmute(c.as_f32x16());
        #[rustfmt::skip]
        let expected_result: [f32; 16] = [
            -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32, -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32,
            -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32, -18.0_f32, -52.0_f32, -16.0_f32, -50.0_f32,
        ];
        assert_eq!(result, expected_result);
        let k: __mmask16 = 0;
        let c: __m512 = _mm512_maskz_dpbf16_ps(k, src, a, b);
        let result: [f32; 16] = transmute(c.as_f32x16());
        #[rustfmt::skip]
        let expected_result: [f32; 16] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(result, expected_result);
    }

    const BF16_ONE: u16 = 0b0_01111111_0000000;
    const BF16_TWO: u16 = 0b0_10000000_0000000;
    const BF16_THREE: u16 = 0b0_10000000_1000000;
    const BF16_FOUR: u16 = 0b0_10000001_0000000;
    const BF16_FIVE: u16 = 0b0_10000001_0100000;
    const BF16_SIX: u16 = 0b0_10000001_1000000;
    const BF16_SEVEN: u16 = 0b0_10000001_1100000;
    const BF16_EIGHT: u16 = 0b0_10000010_0000000;

    #[simd_test(enable = "avx512bf16")]
    unsafe fn test_mm512_cvtpbh_ps() {
        let a = __m256bh([
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        ]);
        let r = _mm512_cvtpbh_ps(a);
        let e = _mm512_setr_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512bf16")]
    unsafe fn test_mm512_mask_cvtpbh_ps() {
        let a = __m256bh([
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        ]);
        let src = _mm512_setr_ps(
            9., 10., 11., 12., 13., 14., 15., 16., 9., 10., 11., 12., 13., 14., 15., 16.,
        );
        let k = 0b1010_1010_1010_1010;
        let r = _mm512_mask_cvtpbh_ps(src, k, a);
        let e = _mm512_setr_ps(
            9., 2., 11., 4., 13., 6., 15., 8., 9., 2., 11., 4., 13., 6., 15., 8.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512bf16")]
    unsafe fn test_mm512_maskz_cvtpbh_ps() {
        let a = __m256bh([
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        ]);
        let k = 0b1010_1010_1010_1010;
        let r = _mm512_maskz_cvtpbh_ps(k, a);
        let e = _mm512_setr_ps(
            0., 2., 0., 4., 0., 6., 0., 8., 0., 2., 0., 4., 0., 6., 0., 8.,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm256_cvtpbh_ps() {
        let a = __m128bh([
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        ]);
        let r = _mm256_cvtpbh_ps(a);
        let e = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm256_mask_cvtpbh_ps() {
        let a = __m128bh([
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        ]);
        let src = _mm256_setr_ps(9., 10., 11., 12., 13., 14., 15., 16.);
        let k = 0b1010_1010;
        let r = _mm256_mask_cvtpbh_ps(src, k, a);
        let e = _mm256_setr_ps(9., 2., 11., 4., 13., 6., 15., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm256_maskz_cvtpbh_ps() {
        let a = __m128bh([
            BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT,
        ]);
        let k = 0b1010_1010;
        let r = _mm256_maskz_cvtpbh_ps(k, a);
        let e = _mm256_setr_ps(0., 2., 0., 4., 0., 6., 0., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm_cvtpbh_ps() {
        let a = __m128bh([BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, 0, 0, 0, 0]);
        let r = _mm_cvtpbh_ps(a);
        let e = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm_mask_cvtpbh_ps() {
        let a = __m128bh([BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, 0, 0, 0, 0]);
        let src = _mm_setr_ps(9., 10., 11., 12.);
        let k = 0b1010;
        let r = _mm_mask_cvtpbh_ps(src, k, a);
        let e = _mm_setr_ps(9., 2., 11., 4.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm_maskz_cvtpbh_ps() {
        let a = __m128bh([BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, 0, 0, 0, 0]);
        let k = 0b1010;
        let r = _mm_maskz_cvtpbh_ps(k, a);
        let e = _mm_setr_ps(0., 2., 0., 4.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx512bf16")]
    unsafe fn test_mm_cvtsbh_ss() {
        let r = _mm_cvtsbh_ss(bf16::from_bits(BF16_ONE));
        assert_eq!(r, 1.);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm_cvtneps_pbh() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let r: u16x4 = transmute_copy(&_mm_cvtneps_pbh(a));
        let e = u16x4::new(BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm_mask_cvtneps_pbh() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let src = __m128bh([5, 6, 7, 8, !0, !0, !0, !0]);
        let k = 0b1010;
        let r: u16x4 = transmute_copy(&_mm_mask_cvtneps_pbh(src, k, a));
        let e = u16x4::new(5, BF16_TWO, 7, BF16_FOUR);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm_maskz_cvtneps_pbh() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let k = 0b1010;
        let r: u16x4 = transmute_copy(&_mm_maskz_cvtneps_pbh(k, a));
        let e = u16x4::new(0, BF16_TWO, 0, BF16_FOUR);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512bf16,avx512vl")]
    unsafe fn test_mm_cvtness_sbh() {
        let r = _mm_cvtness_sbh(1.);
        assert_eq!(r.to_bits(), BF16_ONE);
    }
}
