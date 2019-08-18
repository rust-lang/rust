//! Fused Multiply-Add instruction set (FMA)
//!
//! The FMA instruction set is an extension to the 128 and 256-bit SSE
//! instructions in the x86 microprocessor instruction set to perform fused
//! multiply–add (FMA) operations.
//!
//! The references are:
//!
//! - [Intel 64 and IA-32 Architectures Software Developer's Manual Volume 2:
//!   Instruction Set Reference, A-Z][intel64_ref].
//! - [AMD64 Architecture Programmer's Manual, Volume 3: General-Purpose and
//!   System Instructions][amd64_ref].
//!
//! Wikipedia's [FMA][wiki_fma] page provides a quick overview of the
//! instructions available.
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
//! [amd64_ref]: http://support.amd.com/TechDocs/24594.pdf
//! [wiki_fma]: https://en.wikipedia.org/wiki/Fused_multiply-accumulate

use crate::core_arch::simd_llvm::simd_fma;
use crate::core_arch::x86::*;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and add the intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fmadd_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fmadd_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    simd_fma(a, b, c)
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and add the intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_fmadd_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_fmadd_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    simd_fma(a, b, c)
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and add the intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fmadd_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fmadd_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    simd_fma(a, b, c)
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and add the intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_fmadd_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_fmadd_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    simd_fma(a, b, c)
}

/// Multiplies the lower double-precision (64-bit) floating-point elements in
/// `a` and `b`, and add the intermediate result to the lower element in `c`.
/// Stores the result in the lower element of the returned value, and copy the
/// upper element from `a` to the upper elements of the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fmadd_sd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fmadd_sd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    vfmaddsd(a, b, c)
}

/// Multiplies the lower single-precision (32-bit) floating-point elements in
/// `a` and `b`, and add the intermediate result to the lower element in `c`.
/// Stores the result in the lower element of the returned value, and copy the
/// 3 upper elements from `a` to the upper elements of the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fmadd_ss)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fmadd_ss(a: __m128, b: __m128, c: __m128) -> __m128 {
    vfmaddss(a, b, c)
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and alternatively add and subtract packed elements in `c` to/from
/// the intermediate result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fmaddsub_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmaddsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fmaddsub_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    vfmaddsubpd(a, b, c)
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and alternatively add and subtract packed elements in `c` to/from
/// the intermediate result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_fmaddsub_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmaddsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_fmaddsub_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    vfmaddsubpd256(a, b, c)
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and alternatively add and subtract packed elements in `c` to/from
/// the intermediate result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fmaddsub_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmaddsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fmaddsub_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    vfmaddsubps(a, b, c)
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and alternatively add and subtract packed elements in `c` to/from
/// the intermediate result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_fmaddsub_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmaddsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_fmaddsub_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    vfmaddsubps256(a, b, c)
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the intermediate result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fmsub_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fmsub_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    vfmsubpd(a, b, c)
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the intermediate result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_fmsub_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_fmsub_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    vfmsubpd256(a, b, c)
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the intermediate result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fmsub_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsub213ps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fmsub_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    vfmsubps(a, b, c)
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the intermediate result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_fmsub_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsub213ps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_fmsub_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    vfmsubps256(a, b, c)
}

/// Multiplies the lower double-precision (64-bit) floating-point elements in
/// `a` and `b`, and subtract the lower element in `c` from the intermediate
/// result. Store the result in the lower element of the returned value, and
/// copy the upper element from `a` to the upper elements of the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fmsub_sd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fmsub_sd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    vfmsubsd(a, b, c)
}

/// Multiplies the lower single-precision (32-bit) floating-point elements in
/// `a` and `b`,  and subtract the lower element in `c` from the intermediate
/// result. Store the result in the lower element of the returned value, and
/// copy the 3 upper elements from `a` to the upper elements of the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fmsub_ss)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fmsub_ss(a: __m128, b: __m128, c: __m128) -> __m128 {
    vfmsubss(a, b, c)
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and alternatively subtract and add packed elements in `c` from/to
/// the intermediate result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fmsubadd_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsubadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fmsubadd_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    vfmsubaddpd(a, b, c)
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and alternatively subtract and add packed elements in `c` from/to
/// the intermediate result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_fmsubadd_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsubadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_fmsubadd_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    vfmsubaddpd256(a, b, c)
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and alternatively subtract and add packed elements in `c` from/to
/// the intermediate result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fmsubadd_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsubadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fmsubadd_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    vfmsubaddps(a, b, c)
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and alternatively subtract and add packed elements in `c` from/to
/// the intermediate result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_fmsubadd_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfmsubadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_fmsubadd_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    vfmsubaddps256(a, b, c)
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and add the negated intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fnmadd_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fnmadd_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    vfnmaddpd(a, b, c)
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and add the negated intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_fnmadd_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_fnmadd_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    vfnmaddpd256(a, b, c)
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and add the negated intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fnmadd_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fnmadd_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    vfnmaddps(a, b, c)
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and add the negated intermediate result to packed elements in `c`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_fnmadd_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_fnmadd_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    vfnmaddps256(a, b, c)
}

/// Multiplies the lower double-precision (64-bit) floating-point elements in
/// `a` and `b`, and add the negated intermediate result to the lower element
/// in `c`. Store the result in the lower element of the returned value, and
/// copy the upper element from `a` to the upper elements of the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fnmadd_sd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fnmadd_sd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    vfnmaddsd(a, b, c)
}

/// Multiplies the lower single-precision (32-bit) floating-point elements in
/// `a` and `b`, and add the negated intermediate result to the lower element
/// in `c`. Store the result in the lower element of the returned value, and
/// copy the 3 upper elements from `a` to the upper elements of the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fnmadd_ss)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmadd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fnmadd_ss(a: __m128, b: __m128, c: __m128) -> __m128 {
    vfnmaddss(a, b, c)
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the negated intermediate
/// result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fnmsub_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fnmsub_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    vfnmsubpd(a, b, c)
}

/// Multiplies packed double-precision (64-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the negated intermediate
/// result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_fnmsub_pd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_fnmsub_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    vfnmsubpd256(a, b, c)
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the negated intermediate
/// result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fnmsub_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fnmsub_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    vfnmsubps(a, b, c)
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and subtract packed elements in `c` from the negated intermediate
/// result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_fnmsub_ps)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_fnmsub_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    vfnmsubps256(a, b, c)
}

/// Multiplies the lower double-precision (64-bit) floating-point elements in
/// `a` and `b`, and subtract packed elements in `c` from the negated
/// intermediate result. Store the result in the lower element of the returned
/// value, and copy the upper element from `a` to the upper elements of the
/// result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fnmsub_sd)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fnmsub_sd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    vfnmsubsd(a, b, c)
}

/// Multiplies the lower single-precision (32-bit) floating-point elements in
/// `a` and `b`, and subtract packed elements in `c` from the negated
/// intermediate result. Store the result in the lower element of the
/// returned value, and copy the 3 upper elements from `a` to the upper
/// elements of the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_fnmsub_ss)
#[inline]
#[target_feature(enable = "fma")]
#[cfg_attr(test, assert_instr(vfnmsub))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_fnmsub_ss(a: __m128, b: __m128, c: __m128) -> __m128 {
    vfnmsubss(a, b, c)
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.fma.vfmadd.sd"]
    fn vfmaddsd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
    #[link_name = "llvm.x86.fma.vfmadd.ss"]
    fn vfmaddss(a: __m128, b: __m128, c: __m128) -> __m128;
    #[link_name = "llvm.x86.fma.vfmaddsub.pd"]
    fn vfmaddsubpd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
    #[link_name = "llvm.x86.fma.vfmaddsub.pd.256"]
    fn vfmaddsubpd256(a: __m256d, b: __m256d, c: __m256d) -> __m256d;
    #[link_name = "llvm.x86.fma.vfmaddsub.ps"]
    fn vfmaddsubps(a: __m128, b: __m128, c: __m128) -> __m128;
    #[link_name = "llvm.x86.fma.vfmaddsub.ps.256"]
    fn vfmaddsubps256(a: __m256, b: __m256, c: __m256) -> __m256;
    #[link_name = "llvm.x86.fma.vfmsub.pd"]
    fn vfmsubpd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
    #[link_name = "llvm.x86.fma.vfmsub.pd.256"]
    fn vfmsubpd256(a: __m256d, b: __m256d, c: __m256d) -> __m256d;
    #[link_name = "llvm.x86.fma.vfmsub.ps"]
    fn vfmsubps(a: __m128, b: __m128, c: __m128) -> __m128;
    #[link_name = "llvm.x86.fma.vfmsub.ps.256"]
    fn vfmsubps256(a: __m256, b: __m256, c: __m256) -> __m256;
    #[link_name = "llvm.x86.fma.vfmsub.sd"]
    fn vfmsubsd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
    #[link_name = "llvm.x86.fma.vfmsub.ss"]
    fn vfmsubss(a: __m128, b: __m128, c: __m128) -> __m128;
    #[link_name = "llvm.x86.fma.vfmsubadd.pd"]
    fn vfmsubaddpd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
    #[link_name = "llvm.x86.fma.vfmsubadd.pd.256"]
    fn vfmsubaddpd256(a: __m256d, b: __m256d, c: __m256d) -> __m256d;
    #[link_name = "llvm.x86.fma.vfmsubadd.ps"]
    fn vfmsubaddps(a: __m128, b: __m128, c: __m128) -> __m128;
    #[link_name = "llvm.x86.fma.vfmsubadd.ps.256"]
    fn vfmsubaddps256(a: __m256, b: __m256, c: __m256) -> __m256;
    #[link_name = "llvm.x86.fma.vfnmadd.pd"]
    fn vfnmaddpd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
    #[link_name = "llvm.x86.fma.vfnmadd.pd.256"]
    fn vfnmaddpd256(a: __m256d, b: __m256d, c: __m256d) -> __m256d;
    #[link_name = "llvm.x86.fma.vfnmadd.ps"]
    fn vfnmaddps(a: __m128, b: __m128, c: __m128) -> __m128;
    #[link_name = "llvm.x86.fma.vfnmadd.ps.256"]
    fn vfnmaddps256(a: __m256, b: __m256, c: __m256) -> __m256;
    #[link_name = "llvm.x86.fma.vfnmadd.sd"]
    fn vfnmaddsd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
    #[link_name = "llvm.x86.fma.vfnmadd.ss"]
    fn vfnmaddss(a: __m128, b: __m128, c: __m128) -> __m128;
    #[link_name = "llvm.x86.fma.vfnmsub.pd"]
    fn vfnmsubpd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
    #[link_name = "llvm.x86.fma.vfnmsub.pd.256"]
    fn vfnmsubpd256(a: __m256d, b: __m256d, c: __m256d) -> __m256d;
    #[link_name = "llvm.x86.fma.vfnmsub.ps"]
    fn vfnmsubps(a: __m128, b: __m128, c: __m128) -> __m128;
    #[link_name = "llvm.x86.fma.vfnmsub.ps.256"]
    fn vfnmsubps256(a: __m256, b: __m256, c: __m256) -> __m256;
    #[link_name = "llvm.x86.fma.vfnmsub.sd"]
    fn vfnmsubsd(a: __m128d, b: __m128d, c: __m128d) -> __m128d;
    #[link_name = "llvm.x86.fma.vfnmsub.ss"]
    fn vfnmsubss(a: __m128, b: __m128, c: __m128) -> __m128;
}

#[cfg(test)]
mod tests {
    use std;
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmadd_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(9., 15.);
        assert_eq_m128d(_mm_fmadd_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmadd_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 3., 7., 2.);
        let c = _mm256_setr_pd(4., 9., 1., 7.);
        let r = _mm256_setr_pd(9., 15., 22., 15.);
        assert_eq_m256d(_mm256_fmadd_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmadd_ps() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(9., 15., 22., 15.);
        assert_eq_m128(_mm_fmadd_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmadd_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 0., 10., -1., -2.);
        let b = _mm256_setr_ps(5., 3., 7., 2., 4., -6., 0., 14.);
        let c = _mm256_setr_ps(4., 9., 1., 7., -5., 11., -2., -3.);
        let r = _mm256_setr_ps(9., 15., 22., 15., -5., -49., -2., -31.);
        assert_eq_m256(_mm256_fmadd_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmadd_sd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(9., 2.);
        assert_eq_m128d(_mm_fmadd_sd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmadd_ss() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(9., 2., 3., 4.);
        assert_eq_m128(_mm_fmadd_ss(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmaddsub_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(1., 15.);
        assert_eq_m128d(_mm_fmaddsub_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmaddsub_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 3., 7., 2.);
        let c = _mm256_setr_pd(4., 9., 1., 7.);
        let r = _mm256_setr_pd(1., 15., 20., 15.);
        assert_eq_m256d(_mm256_fmaddsub_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmaddsub_ps() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(1., 15., 20., 15.);
        assert_eq_m128(_mm_fmaddsub_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmaddsub_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 0., 10., -1., -2.);
        let b = _mm256_setr_ps(5., 3., 7., 2., 4., -6., 0., 14.);
        let c = _mm256_setr_ps(4., 9., 1., 7., -5., 11., -2., -3.);
        let r = _mm256_setr_ps(1., 15., 20., 15., 5., -49., 2., -31.);
        assert_eq_m256(_mm256_fmaddsub_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmsub_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(1., -3.);
        assert_eq_m128d(_mm_fmsub_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmsub_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 3., 7., 2.);
        let c = _mm256_setr_pd(4., 9., 1., 7.);
        let r = _mm256_setr_pd(1., -3., 20., 1.);
        assert_eq_m256d(_mm256_fmsub_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmsub_ps() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(1., -3., 20., 1.);
        assert_eq_m128(_mm_fmsub_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmsub_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 0., 10., -1., -2.);
        let b = _mm256_setr_ps(5., 3., 7., 2., 4., -6., 0., 14.);
        let c = _mm256_setr_ps(4., 9., 1., 7., -5., 11., -2., -3.);
        let r = _mm256_setr_ps(1., -3., 20., 1., 5., -71., 2., -25.);
        assert_eq_m256(_mm256_fmsub_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmsub_sd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(1., 2.);
        assert_eq_m128d(_mm_fmsub_sd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmsub_ss() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(1., 2., 3., 4.);
        assert_eq_m128(_mm_fmsub_ss(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmsubadd_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(9., -3.);
        assert_eq_m128d(_mm_fmsubadd_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmsubadd_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 3., 7., 2.);
        let c = _mm256_setr_pd(4., 9., 1., 7.);
        let r = _mm256_setr_pd(9., -3., 22., 1.);
        assert_eq_m256d(_mm256_fmsubadd_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fmsubadd_ps() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(9., -3., 22., 1.);
        assert_eq_m128(_mm_fmsubadd_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fmsubadd_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 0., 10., -1., -2.);
        let b = _mm256_setr_ps(5., 3., 7., 2., 4., -6., 0., 14.);
        let c = _mm256_setr_ps(4., 9., 1., 7., -5., 11., -2., -3.);
        let r = _mm256_setr_ps(9., -3., 22., 1., -5., -71., -2., -25.);
        assert_eq_m256(_mm256_fmsubadd_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmadd_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(-1., 3.);
        assert_eq_m128d(_mm_fnmadd_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fnmadd_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 3., 7., 2.);
        let c = _mm256_setr_pd(4., 9., 1., 7.);
        let r = _mm256_setr_pd(-1., 3., -20., -1.);
        assert_eq_m256d(_mm256_fnmadd_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmadd_ps() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(-1., 3., -20., -1.);
        assert_eq_m128(_mm_fnmadd_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fnmadd_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 0., 10., -1., -2.);
        let b = _mm256_setr_ps(5., 3., 7., 2., 4., -6., 0., 14.);
        let c = _mm256_setr_ps(4., 9., 1., 7., -5., 11., -2., -3.);
        let r = _mm256_setr_ps(-1., 3., -20., -1., -5., 71., -2., 25.);
        assert_eq_m256(_mm256_fnmadd_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmadd_sd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(-1., 2.);
        assert_eq_m128d(_mm_fnmadd_sd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmadd_ss() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(-1., 2., 3., 4.);
        assert_eq_m128(_mm_fnmadd_ss(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmsub_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(-9., -15.);
        assert_eq_m128d(_mm_fnmsub_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fnmsub_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 3., 7., 2.);
        let c = _mm256_setr_pd(4., 9., 1., 7.);
        let r = _mm256_setr_pd(-9., -15., -22., -15.);
        assert_eq_m256d(_mm256_fnmsub_pd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmsub_ps() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(-9., -15., -22., -15.);
        assert_eq_m128(_mm_fnmsub_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm256_fnmsub_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 0., 10., -1., -2.);
        let b = _mm256_setr_ps(5., 3., 7., 2., 4., -6., 0., 14.);
        let c = _mm256_setr_ps(4., 9., 1., 7., -5., 11., -2., -3.);
        let r = _mm256_setr_ps(-9., -15., -22., -15., 5., 49., 2., 31.);
        assert_eq_m256(_mm256_fnmsub_ps(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmsub_sd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 3.);
        let c = _mm_setr_pd(4., 9.);
        let r = _mm_setr_pd(-9., 2.);
        assert_eq_m128d(_mm_fnmsub_sd(a, b, c), r);
    }

    #[simd_test(enable = "fma")]
    unsafe fn test_mm_fnmsub_ss() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let b = _mm_setr_ps(5., 3., 7., 2.);
        let c = _mm_setr_ps(4., 9., 1., 7.);
        let r = _mm_setr_ps(-9., 2., 3., 4.);
        assert_eq_m128(_mm_fnmsub_ss(a, b, c), r);
    }
}
