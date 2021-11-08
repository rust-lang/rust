//! Advanced Vector Extensions (AVX)
//!
//! The references are:
//!
//! - [Intel 64 and IA-32 Architectures Software Developer's Manual Volume 2:
//! Instruction Set Reference, A-Z][intel64_ref]. - [AMD64 Architecture
//! Programmer's Manual, Volume 3: General-Purpose and System
//! Instructions][amd64_ref].
//!
//! [Wikipedia][wiki] provides a quick overview of the instructions available.
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
//! [amd64_ref]: http://support.amd.com/TechDocs/24594.pdf
//! [wiki]: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions

use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    intrinsics,
    mem::{self, transmute},
    ptr,
};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Adds packed double-precision (64-bit) floating-point elements
/// in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_add_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vaddpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_add_pd(a: __m256d, b: __m256d) -> __m256d {
    simd_add(a, b)
}

/// Adds packed single-precision (32-bit) floating-point elements in `a` and
/// `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_add_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vaddps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_add_ps(a: __m256, b: __m256) -> __m256 {
    simd_add(a, b)
}

/// Computes the bitwise AND of a packed double-precision (64-bit)
/// floating-point elements in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_and_pd)
#[inline]
#[target_feature(enable = "avx")]
// FIXME: Should be 'vandpd' instuction.
// See https://github.com/rust-lang/stdarch/issues/71
#[cfg_attr(test, assert_instr(vandps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_and_pd(a: __m256d, b: __m256d) -> __m256d {
    let a: u64x4 = transmute(a);
    let b: u64x4 = transmute(b);
    transmute(simd_and(a, b))
}

/// Computes the bitwise AND of packed single-precision (32-bit) floating-point
/// elements in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_and_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vandps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_and_ps(a: __m256, b: __m256) -> __m256 {
    let a: u32x8 = transmute(a);
    let b: u32x8 = transmute(b);
    transmute(simd_and(a, b))
}

/// Computes the bitwise OR packed double-precision (64-bit) floating-point
/// elements in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_or_pd)
#[inline]
#[target_feature(enable = "avx")]
// FIXME: should be `vorpd` instuction.
// See <https://github.com/rust-lang/stdarch/issues/71>.
#[cfg_attr(test, assert_instr(vorps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_or_pd(a: __m256d, b: __m256d) -> __m256d {
    let a: u64x4 = transmute(a);
    let b: u64x4 = transmute(b);
    transmute(simd_or(a, b))
}

/// Computes the bitwise OR packed single-precision (32-bit) floating-point
/// elements in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_or_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vorps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_or_ps(a: __m256, b: __m256) -> __m256 {
    let a: u32x8 = transmute(a);
    let b: u32x8 = transmute(b);
    transmute(simd_or(a, b))
}

/// Shuffles double-precision (64-bit) floating-point elements within 128-bit
/// lanes using the control in `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_shuffle_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vshufpd, MASK = 3))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_shuffle_pd<const MASK: i32>(a: __m256d, b: __m256d) -> __m256d {
    static_assert_imm8!(MASK);
    simd_shuffle4!(
        a,
        b,
        <const MASK: i32> [
            MASK as u32 & 0b1,
            ((MASK as u32 >> 1) & 0b1) + 4,
            ((MASK as u32 >> 2) & 0b1) + 2,
            ((MASK as u32 >> 3) & 0b1) + 6,
        ],
    )
}

/// Shuffles single-precision (32-bit) floating-point elements in `a` within
/// 128-bit lanes using the control in `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_shuffle_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vshufps, MASK = 3))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_shuffle_ps<const MASK: i32>(a: __m256, b: __m256) -> __m256 {
    static_assert_imm8!(MASK);
    simd_shuffle8!(
        a,
        b,
        <const MASK: i32> [
            MASK as u32 & 0b11,
            (MASK as u32 >> 2) & 0b11,
            ((MASK as u32 >> 4) & 0b11) + 8,
            ((MASK as u32 >> 6) & 0b11) + 8,
            (MASK as u32 & 0b11) + 4,
            ((MASK as u32 >> 2) & 0b11) + 4,
            ((MASK as u32 >> 4) & 0b11) + 12,
            ((MASK as u32 >> 6) & 0b11) + 12,
        ],
    )
}

/// Computes the bitwise NOT of packed double-precision (64-bit) floating-point
/// elements in `a`, and then AND with `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_andnot_pd)
#[inline]
#[target_feature(enable = "avx")]
// FIXME: should be `vandnpd` instruction.
#[cfg_attr(test, assert_instr(vandnps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_andnot_pd(a: __m256d, b: __m256d) -> __m256d {
    let a: u64x4 = transmute(a);
    let b: u64x4 = transmute(b);
    transmute(simd_and(simd_xor(u64x4::splat(!(0_u64)), a), b))
}

/// Computes the bitwise NOT of packed single-precision (32-bit) floating-point
/// elements in `a`
/// and then AND with `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_andnot_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vandnps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_andnot_ps(a: __m256, b: __m256) -> __m256 {
    let a: u32x8 = transmute(a);
    let b: u32x8 = transmute(b);
    transmute(simd_and(simd_xor(u32x8::splat(!(0_u32)), a), b))
}

/// Compares packed double-precision (64-bit) floating-point elements
/// in `a` and `b`, and returns packed maximum values
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_max_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmaxpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_max_pd(a: __m256d, b: __m256d) -> __m256d {
    vmaxpd(a, b)
}

/// Compares packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and returns packed maximum values
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_max_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmaxps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_max_ps(a: __m256, b: __m256) -> __m256 {
    vmaxps(a, b)
}

/// Compares packed double-precision (64-bit) floating-point elements
/// in `a` and `b`, and returns packed minimum values
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_min_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vminpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_min_pd(a: __m256d, b: __m256d) -> __m256d {
    vminpd(a, b)
}

/// Compares packed single-precision (32-bit) floating-point elements in `a`
/// and `b`, and returns packed minimum values
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_min_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vminps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_min_ps(a: __m256, b: __m256) -> __m256 {
    vminps(a, b)
}

/// Multiplies packed double-precision (64-bit) floating-point elements
/// in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mul_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmulpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mul_pd(a: __m256d, b: __m256d) -> __m256d {
    simd_mul(a, b)
}

/// Multiplies packed single-precision (32-bit) floating-point elements in `a` and
/// `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mul_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmulps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_mul_ps(a: __m256, b: __m256) -> __m256 {
    simd_mul(a, b)
}

/// Alternatively adds and subtracts packed double-precision (64-bit)
/// floating-point elements in `a` to/from packed elements in `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_addsub_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vaddsubpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_addsub_pd(a: __m256d, b: __m256d) -> __m256d {
    addsubpd256(a, b)
}

/// Alternatively adds and subtracts packed single-precision (32-bit)
/// floating-point elements in `a` to/from packed elements in `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_addsub_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vaddsubps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_addsub_ps(a: __m256, b: __m256) -> __m256 {
    addsubps256(a, b)
}

/// Subtracts packed double-precision (64-bit) floating-point elements in `b`
/// from packed elements in `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sub_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vsubpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sub_pd(a: __m256d, b: __m256d) -> __m256d {
    simd_sub(a, b)
}

/// Subtracts packed single-precision (32-bit) floating-point elements in `b`
/// from packed elements in `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sub_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vsubps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sub_ps(a: __m256, b: __m256) -> __m256 {
    simd_sub(a, b)
}

/// Computes the division of each of the 8 packed 32-bit floating-point elements
/// in `a` by the corresponding packed elements in `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_div_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vdivps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_div_ps(a: __m256, b: __m256) -> __m256 {
    simd_div(a, b)
}

/// Computes the division of each of the 4 packed 64-bit floating-point elements
/// in `a` by the corresponding packed elements in `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_div_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vdivpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_div_pd(a: __m256d, b: __m256d) -> __m256d {
    simd_div(a, b)
}

/// Rounds packed double-precision (64-bit) floating point elements in `a`
/// according to the flag `ROUNDING`. The value of `ROUNDING` may be as follows:
///
/// - `0x00`: Round to the nearest whole number.
/// - `0x01`: Round down, toward negative infinity.
/// - `0x02`: Round up, toward positive infinity.
/// - `0x03`: Truncate the values.
///
/// For a complete list of options, check [the LLVM docs][llvm_docs].
///
/// [llvm_docs]: https://github.com/llvm-mirror/clang/blob/dcd8d797b20291f1a6b3e0ddda085aa2bbb382a8/lib/Headers/avxintrin.h#L382
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_round_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vroundpd, ROUNDING = 0x3))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_round_pd<const ROUNDING: i32>(a: __m256d) -> __m256d {
    static_assert_imm4!(ROUNDING);
    roundpd256(a, ROUNDING)
}

/// Rounds packed double-precision (64-bit) floating point elements in `a`
/// toward positive infinity.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_ceil_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vroundpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_ceil_pd(a: __m256d) -> __m256d {
    simd_ceil(a)
}

/// Rounds packed double-precision (64-bit) floating point elements in `a`
/// toward negative infinity.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_floor_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vroundpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_floor_pd(a: __m256d) -> __m256d {
    simd_floor(a)
}

/// Rounds packed single-precision (32-bit) floating point elements in `a`
/// according to the flag `ROUNDING`. The value of `ROUNDING` may be as follows:
///
/// - `0x00`: Round to the nearest whole number.
/// - `0x01`: Round down, toward negative infinity.
/// - `0x02`: Round up, toward positive infinity.
/// - `0x03`: Truncate the values.
///
/// For a complete list of options, check [the LLVM docs][llvm_docs].
///
/// [llvm_docs]: https://github.com/llvm-mirror/clang/blob/dcd8d797b20291f1a6b3e0ddda085aa2bbb382a8/lib/Headers/avxintrin.h#L382
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_round_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vroundps, ROUNDING = 0x00))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_round_ps<const ROUNDING: i32>(a: __m256) -> __m256 {
    static_assert_imm4!(ROUNDING);
    roundps256(a, ROUNDING)
}

/// Rounds packed single-precision (32-bit) floating point elements in `a`
/// toward positive infinity.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_ceil_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vroundps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_ceil_ps(a: __m256) -> __m256 {
    simd_ceil(a)
}

/// Rounds packed single-precision (32-bit) floating point elements in `a`
/// toward negative infinity.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_floor_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vroundps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_floor_ps(a: __m256) -> __m256 {
    simd_floor(a)
}

/// Returns the square root of packed single-precision (32-bit) floating point
/// elements in `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sqrt_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vsqrtps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sqrt_ps(a: __m256) -> __m256 {
    sqrtps256(a)
}

/// Returns the square root of packed double-precision (64-bit) floating point
/// elements in `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sqrt_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vsqrtpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_sqrt_pd(a: __m256d) -> __m256d {
    simd_fsqrt(a)
}

/// Blends packed double-precision (64-bit) floating-point elements from
/// `a` and `b` using control mask `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_blend_pd)
#[inline]
#[target_feature(enable = "avx")]
// Note: LLVM7 prefers single-precision blend instructions when
// possible, see: https://bugs.llvm.org/show_bug.cgi?id=38194
// #[cfg_attr(test, assert_instr(vblendpd, imm8 = 9))]
#[cfg_attr(test, assert_instr(vblendps, IMM4 = 9))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_blend_pd<const IMM4: i32>(a: __m256d, b: __m256d) -> __m256d {
    static_assert_imm4!(IMM4);
    simd_shuffle4!(
        a,
        b,
        <const IMM4: i32> [
            ((IMM4 as u32 >> 0) & 1) * 4 + 0,
            ((IMM4 as u32 >> 1) & 1) * 4 + 1,
            ((IMM4 as u32 >> 2) & 1) * 4 + 2,
            ((IMM4 as u32 >> 3) & 1) * 4 + 3,
        ],
    )
}

/// Blends packed single-precision (32-bit) floating-point elements from
/// `a` and `b` using control mask `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_blend_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vblendps, IMM8 = 9))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_blend_ps<const IMM8: i32>(a: __m256, b: __m256) -> __m256 {
    static_assert_imm8!(IMM8);
    simd_shuffle8!(
        a,
        b,
        <const IMM8: i32> [
            ((IMM8 as u32 >> 0) & 1) * 8 + 0,
            ((IMM8 as u32 >> 1) & 1) * 8 + 1,
            ((IMM8 as u32 >> 2) & 1) * 8 + 2,
            ((IMM8 as u32 >> 3) & 1) * 8 + 3,
            ((IMM8 as u32 >> 4) & 1) * 8 + 4,
            ((IMM8 as u32 >> 5) & 1) * 8 + 5,
            ((IMM8 as u32 >> 6) & 1) * 8 + 6,
            ((IMM8 as u32 >> 7) & 1) * 8 + 7,
        ],
    )
}

/// Blends packed double-precision (64-bit) floating-point elements from
/// `a` and `b` using `c` as a mask.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_blendv_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vblendvpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_blendv_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    vblendvpd(a, b, c)
}

/// Blends packed single-precision (32-bit) floating-point elements from
/// `a` and `b` using `c` as a mask.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_blendv_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vblendvps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_blendv_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    vblendvps(a, b, c)
}

/// Conditionally multiplies the packed single-precision (32-bit) floating-point
/// elements in `a` and `b` using the high 4 bits in `imm8`,
/// sum the four products, and conditionally return the sum
///  using the low 4 bits of `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_dp_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vdpps, IMM8 = 0x0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_dp_ps<const IMM8: i32>(a: __m256, b: __m256) -> __m256 {
    static_assert_imm8!(IMM8);
    vdpps(a, b, IMM8)
}

/// Horizontal addition of adjacent pairs in the two packed vectors
/// of 4 64-bit floating points `a` and `b`.
/// In the result, sums of elements from `a` are returned in even locations,
/// while sums of elements from `b` are returned in odd locations.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_hadd_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vhaddpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_hadd_pd(a: __m256d, b: __m256d) -> __m256d {
    vhaddpd(a, b)
}

/// Horizontal addition of adjacent pairs in the two packed vectors
/// of 8 32-bit floating points `a` and `b`.
/// In the result, sums of elements from `a` are returned in locations of
/// indices 0, 1, 4, 5; while sums of elements from `b` are locations
/// 2, 3, 6, 7.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_hadd_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vhaddps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_hadd_ps(a: __m256, b: __m256) -> __m256 {
    vhaddps(a, b)
}

/// Horizontal subtraction of adjacent pairs in the two packed vectors
/// of 4 64-bit floating points `a` and `b`.
/// In the result, sums of elements from `a` are returned in even locations,
/// while sums of elements from `b` are returned in odd locations.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_hsub_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vhsubpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_hsub_pd(a: __m256d, b: __m256d) -> __m256d {
    vhsubpd(a, b)
}

/// Horizontal subtraction of adjacent pairs in the two packed vectors
/// of 8 32-bit floating points `a` and `b`.
/// In the result, sums of elements from `a` are returned in locations of
/// indices 0, 1, 4, 5; while sums of elements from `b` are locations
/// 2, 3, 6, 7.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_hsub_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vhsubps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_hsub_ps(a: __m256, b: __m256) -> __m256 {
    vhsubps(a, b)
}

/// Computes the bitwise XOR of packed double-precision (64-bit) floating-point
/// elements in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_xor_pd)
#[inline]
#[target_feature(enable = "avx")]
// FIXME Should be 'vxorpd' instruction.
#[cfg_attr(test, assert_instr(vxorps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_xor_pd(a: __m256d, b: __m256d) -> __m256d {
    let a: u64x4 = transmute(a);
    let b: u64x4 = transmute(b);
    transmute(simd_xor(a, b))
}

/// Computes the bitwise XOR of packed single-precision (32-bit) floating-point
/// elements in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_xor_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vxorps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_xor_ps(a: __m256, b: __m256) -> __m256 {
    let a: u32x8 = transmute(a);
    let b: u32x8 = transmute(b);
    transmute(simd_xor(a, b))
}

/// Equal (ordered, non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_EQ_OQ: i32 = 0x00;
/// Less-than (ordered, signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_LT_OS: i32 = 0x01;
/// Less-than-or-equal (ordered, signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_LE_OS: i32 = 0x02;
/// Unordered (non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_UNORD_Q: i32 = 0x03;
/// Not-equal (unordered, non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_NEQ_UQ: i32 = 0x04;
/// Not-less-than (unordered, signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_NLT_US: i32 = 0x05;
/// Not-less-than-or-equal (unordered, signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_NLE_US: i32 = 0x06;
/// Ordered (non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_ORD_Q: i32 = 0x07;
/// Equal (unordered, non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_EQ_UQ: i32 = 0x08;
/// Not-greater-than-or-equal (unordered, signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_NGE_US: i32 = 0x09;
/// Not-greater-than (unordered, signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_NGT_US: i32 = 0x0a;
/// False (ordered, non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_FALSE_OQ: i32 = 0x0b;
/// Not-equal (ordered, non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_NEQ_OQ: i32 = 0x0c;
/// Greater-than-or-equal (ordered, signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_GE_OS: i32 = 0x0d;
/// Greater-than (ordered, signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_GT_OS: i32 = 0x0e;
/// True (unordered, non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_TRUE_UQ: i32 = 0x0f;
/// Equal (ordered, signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_EQ_OS: i32 = 0x10;
/// Less-than (ordered, non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_LT_OQ: i32 = 0x11;
/// Less-than-or-equal (ordered, non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_LE_OQ: i32 = 0x12;
/// Unordered (signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_UNORD_S: i32 = 0x13;
/// Not-equal (unordered, signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_NEQ_US: i32 = 0x14;
/// Not-less-than (unordered, non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_NLT_UQ: i32 = 0x15;
/// Not-less-than-or-equal (unordered, non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_NLE_UQ: i32 = 0x16;
/// Ordered (signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_ORD_S: i32 = 0x17;
/// Equal (unordered, signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_EQ_US: i32 = 0x18;
/// Not-greater-than-or-equal (unordered, non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_NGE_UQ: i32 = 0x19;
/// Not-greater-than (unordered, non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_NGT_UQ: i32 = 0x1a;
/// False (ordered, signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_FALSE_OS: i32 = 0x1b;
/// Not-equal (ordered, signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_NEQ_OS: i32 = 0x1c;
/// Greater-than-or-equal (ordered, non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_GE_OQ: i32 = 0x1d;
/// Greater-than (ordered, non-signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_GT_OQ: i32 = 0x1e;
/// True (unordered, signaling)
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _CMP_TRUE_US: i32 = 0x1f;

/// Compares packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the comparison operand
/// specified by `IMM5`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmp_pd)
#[inline]
#[target_feature(enable = "avx,sse2")]
#[cfg_attr(test, assert_instr(vcmpeqpd, IMM5 = 0))] // TODO Validate vcmppd
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cmp_pd<const IMM5: i32>(a: __m128d, b: __m128d) -> __m128d {
    static_assert_imm5!(IMM5);
    vcmppd(a, b, IMM5 as i8)
}

/// Compares packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the comparison operand
/// specified by `IMM5`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmp_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vcmpeqpd, IMM5 = 0))] // TODO Validate vcmppd
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cmp_pd<const IMM5: i32>(a: __m256d, b: __m256d) -> __m256d {
    static_assert_imm5!(IMM5);
    vcmppd256(a, b, IMM5 as u8)
}

/// Compares packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the comparison operand
/// specified by `IMM5`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmp_ps)
#[inline]
#[target_feature(enable = "avx,sse")]
#[cfg_attr(test, assert_instr(vcmpeqps, IMM5 = 0))] // TODO Validate vcmpps
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cmp_ps<const IMM5: i32>(a: __m128, b: __m128) -> __m128 {
    static_assert_imm5!(IMM5);
    vcmpps(a, b, IMM5 as i8)
}

/// Compares packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the comparison operand
/// specified by `IMM5`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmp_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vcmpeqps, IMM5 = 0))] // TODO Validate vcmpps
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cmp_ps<const IMM5: i32>(a: __m256, b: __m256) -> __m256 {
    static_assert_imm5!(IMM5);
    vcmpps256(a, b, IMM5 as u8)
}

/// Compares the lower double-precision (64-bit) floating-point element in
/// `a` and `b` based on the comparison operand specified by `IMM5`,
/// store the result in the lower element of returned vector,
/// and copies the upper element from `a` to the upper element of returned
/// vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmp_sd)
#[inline]
#[target_feature(enable = "avx,sse2")]
#[cfg_attr(test, assert_instr(vcmpeqsd, IMM5 = 0))] // TODO Validate vcmpsd
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cmp_sd<const IMM5: i32>(a: __m128d, b: __m128d) -> __m128d {
    static_assert_imm5!(IMM5);
    vcmpsd(a, b, IMM5 as i8)
}

/// Compares the lower single-precision (32-bit) floating-point element in
/// `a` and `b` based on the comparison operand specified by `IMM5`,
/// store the result in the lower element of returned vector,
/// and copies the upper 3 packed elements from `a` to the upper elements of
/// returned vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmp_ss)
#[inline]
#[target_feature(enable = "avx,sse")]
#[cfg_attr(test, assert_instr(vcmpeqss, IMM5 = 0))] // TODO Validate vcmpss
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cmp_ss<const IMM5: i32>(a: __m128, b: __m128) -> __m128 {
    static_assert_imm5!(IMM5);
    vcmpss(a, b, IMM5 as i8)
}

/// Converts packed 32-bit integers in `a` to packed double-precision (64-bit)
/// floating-point elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepi32_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vcvtdq2pd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtepi32_pd(a: __m128i) -> __m256d {
    simd_cast(a.as_i32x4())
}

/// Converts packed 32-bit integers in `a` to packed single-precision (32-bit)
/// floating-point elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtepi32_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vcvtdq2ps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtepi32_ps(a: __m256i) -> __m256 {
    vcvtdq2ps(a.as_i32x8())
}

/// Converts packed double-precision (64-bit) floating-point elements in `a`
/// to packed single-precision (32-bit) floating-point elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtpd_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vcvtpd2ps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtpd_ps(a: __m256d) -> __m128 {
    vcvtpd2ps(a)
}

/// Converts packed single-precision (32-bit) floating-point elements in `a`
/// to packed 32-bit integers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtps_epi32)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vcvtps2dq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtps_epi32(a: __m256) -> __m256i {
    transmute(vcvtps2dq(a))
}

/// Converts packed single-precision (32-bit) floating-point elements in `a`
/// to packed double-precision (64-bit) floating-point elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtps_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vcvtps2pd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtps_pd(a: __m128) -> __m256d {
    simd_cast(a)
}

/// Converts packed double-precision (64-bit) floating-point elements in `a`
/// to packed 32-bit integers with truncation.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvttpd_epi32)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vcvttpd2dq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvttpd_epi32(a: __m256d) -> __m128i {
    transmute(vcvttpd2dq(a))
}

/// Converts packed double-precision (64-bit) floating-point elements in `a`
/// to packed 32-bit integers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtpd_epi32)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vcvtpd2dq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtpd_epi32(a: __m256d) -> __m128i {
    transmute(vcvtpd2dq(a))
}

/// Converts packed single-precision (32-bit) floating-point elements in `a`
/// to packed 32-bit integers with truncation.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvttps_epi32)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vcvttps2dq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvttps_epi32(a: __m256) -> __m256i {
    transmute(vcvttps2dq(a))
}

/// Extracts 128 bits (composed of 4 packed single-precision (32-bit)
/// floating-point elements) from `a`, selected with `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_extractf128_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(
    all(test, not(target_os = "windows")),
    assert_instr(vextractf128, IMM1 = 1)
)]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_extractf128_ps<const IMM1: i32>(a: __m256) -> __m128 {
    static_assert_imm1!(IMM1);
    simd_shuffle4!(
        a,
        _mm256_undefined_ps(),
        <const IMM1: i32> [[0, 1, 2, 3], [4, 5, 6, 7]][IMM1 as usize],
    )
}

/// Extracts 128 bits (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from `a`, selected with `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_extractf128_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(
    all(test, not(target_os = "windows")),
    assert_instr(vextractf128, IMM1 = 1)
)]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_extractf128_pd<const IMM1: i32>(a: __m256d) -> __m128d {
    static_assert_imm1!(IMM1);
    simd_shuffle2!(a, _mm256_undefined_pd(), <const IMM1: i32> [[0, 1], [2, 3]][IMM1 as usize])
}

/// Extracts 128 bits (composed of integer data) from `a`, selected with `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_extractf128_si256)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(
    all(test, not(target_os = "windows")),
    assert_instr(vextractf128, IMM1 = 1)
)]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_extractf128_si256<const IMM1: i32>(a: __m256i) -> __m128i {
    static_assert_imm1!(IMM1);
    let dst: i64x2 = simd_shuffle2!(
        a.as_i64x4(),
        _mm256_undefined_si256().as_i64x4(),
        <const IMM1: i32> [[0, 1], [2, 3]][IMM1 as usize],
    );
    transmute(dst)
}

/// Zeroes the contents of all XMM or YMM registers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_zeroall)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vzeroall))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_zeroall() {
    vzeroall()
}

/// Zeroes the upper 128 bits of all YMM registers;
/// the lower 128-bits of the registers are unmodified.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_zeroupper)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vzeroupper))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_zeroupper() {
    vzeroupper()
}

/// Shuffles single-precision (32-bit) floating-point elements in `a`
/// within 128-bit lanes using the control in `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permutevar_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vpermilps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_permutevar_ps(a: __m256, b: __m256i) -> __m256 {
    vpermilps256(a, b.as_i32x8())
}

/// Shuffles single-precision (32-bit) floating-point elements in `a`
/// using the control in `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_permutevar_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vpermilps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_permutevar_ps(a: __m128, b: __m128i) -> __m128 {
    vpermilps(a, b.as_i32x4())
}

/// Shuffles single-precision (32-bit) floating-point elements in `a`
/// within 128-bit lanes using the control in `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permute_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vpermilps, IMM8 = 9))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_permute_ps<const IMM8: i32>(a: __m256) -> __m256 {
    static_assert_imm8!(IMM8);
    simd_shuffle8!(
        a,
        _mm256_undefined_ps(),
        <const IMM8: i32> [
            (IMM8 as u32 >> 0) & 0b11,
            (IMM8 as u32 >> 2) & 0b11,
            (IMM8 as u32 >> 4) & 0b11,
            (IMM8 as u32 >> 6) & 0b11,
            ((IMM8 as u32 >> 0) & 0b11) + 4,
            ((IMM8 as u32 >> 2) & 0b11) + 4,
            ((IMM8 as u32 >> 4) & 0b11) + 4,
            ((IMM8 as u32 >> 6) & 0b11) + 4,
        ],
    )
}

/// Shuffles single-precision (32-bit) floating-point elements in `a`
/// using the control in `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_permute_ps)
#[inline]
#[target_feature(enable = "avx,sse")]
#[cfg_attr(test, assert_instr(vpermilps, IMM8 = 9))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_permute_ps<const IMM8: i32>(a: __m128) -> __m128 {
    static_assert_imm8!(IMM8);
    simd_shuffle4!(
        a,
        _mm_undefined_ps(),
        <const IMM8: i32> [
            (IMM8 as u32 >> 0) & 0b11,
            (IMM8 as u32 >> 2) & 0b11,
            (IMM8 as u32 >> 4) & 0b11,
            (IMM8 as u32 >> 6) & 0b11,
        ],
    )
}

/// Shuffles double-precision (64-bit) floating-point elements in `a`
/// within 256-bit lanes using the control in `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permutevar_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vpermilpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_permutevar_pd(a: __m256d, b: __m256i) -> __m256d {
    vpermilpd256(a, b.as_i64x4())
}

/// Shuffles double-precision (64-bit) floating-point elements in `a`
/// using the control in `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_permutevar_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vpermilpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_permutevar_pd(a: __m128d, b: __m128i) -> __m128d {
    vpermilpd(a, b.as_i64x2())
}

/// Shuffles double-precision (64-bit) floating-point elements in `a`
/// within 128-bit lanes using the control in `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permute_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vpermilpd, IMM4 = 0x1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_permute_pd<const IMM4: i32>(a: __m256d) -> __m256d {
    static_assert_imm4!(IMM4);
    simd_shuffle4!(
        a,
        _mm256_undefined_pd(),
        <const IMM4: i32> [
            ((IMM4 as u32 >> 0) & 1),
            ((IMM4 as u32 >> 1) & 1),
            ((IMM4 as u32 >> 2) & 1) + 2,
            ((IMM4 as u32 >> 3) & 1) + 2,
        ],
    )
}

/// Shuffles double-precision (64-bit) floating-point elements in `a`
/// using the control in `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_permute_pd)
#[inline]
#[target_feature(enable = "avx,sse2")]
#[cfg_attr(test, assert_instr(vpermilpd, IMM2 = 0x1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_permute_pd<const IMM2: i32>(a: __m128d) -> __m128d {
    static_assert_imm2!(IMM2);
    simd_shuffle2!(
        a,
        _mm_undefined_pd(),
        <const IMM2: i32> [(IMM2 as u32) & 1, (IMM2 as u32 >> 1) & 1],
    )
}

/// Shuffles 256 bits (composed of 8 packed single-precision (32-bit)
/// floating-point elements) selected by `imm8` from `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permute2f128_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vperm2f128, IMM8 = 0x5))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_permute2f128_ps<const IMM8: i32>(a: __m256, b: __m256) -> __m256 {
    static_assert_imm8!(IMM8);
    vperm2f128ps256(a, b, IMM8 as i8)
}

/// Shuffles 256 bits (composed of 4 packed double-precision (64-bit)
/// floating-point elements) selected by `imm8` from `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permute2f128_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vperm2f128, IMM8 = 0x31))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_permute2f128_pd<const IMM8: i32>(a: __m256d, b: __m256d) -> __m256d {
    static_assert_imm8!(IMM8);
    vperm2f128pd256(a, b, IMM8 as i8)
}

/// Shuffles 128-bits (composed of integer data) selected by `imm8`
/// from `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permute2f128_si256)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vperm2f128, IMM8 = 0x31))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_permute2f128_si256<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    static_assert_imm8!(IMM8);
    transmute(vperm2f128si256(a.as_i32x8(), b.as_i32x8(), IMM8 as i8))
}

/// Broadcasts a single-precision (32-bit) floating-point element from memory
/// to all elements of the returned vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_broadcast_ss)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vbroadcastss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow(clippy::trivially_copy_pass_by_ref)]
pub unsafe fn _mm256_broadcast_ss(f: &f32) -> __m256 {
    _mm256_set1_ps(*f)
}

/// Broadcasts a single-precision (32-bit) floating-point element from memory
/// to all elements of the returned vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_broadcast_ss)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vbroadcastss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow(clippy::trivially_copy_pass_by_ref)]
pub unsafe fn _mm_broadcast_ss(f: &f32) -> __m128 {
    _mm_set1_ps(*f)
}

/// Broadcasts a double-precision (64-bit) floating-point element from memory
/// to all elements of the returned vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_broadcast_sd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vbroadcastsd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow(clippy::trivially_copy_pass_by_ref)]
pub unsafe fn _mm256_broadcast_sd(f: &f64) -> __m256d {
    _mm256_set1_pd(*f)
}

/// Broadcasts 128 bits from memory (composed of 4 packed single-precision
/// (32-bit) floating-point elements) to all elements of the returned vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_broadcast_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vbroadcastf128))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_broadcast_ps(a: &__m128) -> __m256 {
    vbroadcastf128ps256(a)
}

/// Broadcasts 128 bits from memory (composed of 2 packed double-precision
/// (64-bit) floating-point elements) to all elements of the returned vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_broadcast_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vbroadcastf128))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_broadcast_pd(a: &__m128d) -> __m256d {
    vbroadcastf128pd256(a)
}

/// Copies `a` to result, then inserts 128 bits (composed of 4 packed
/// single-precision (32-bit) floating-point elements) from `b` into result
/// at the location specified by `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_insertf128_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(
    all(test, not(target_os = "windows")),
    assert_instr(vinsertf128, IMM1 = 1)
)]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_insertf128_ps<const IMM1: i32>(a: __m256, b: __m128) -> __m256 {
    static_assert_imm1!(IMM1);
    simd_shuffle8!(
        a,
        _mm256_castps128_ps256(b),
        <const IMM1: i32> [[8, 9, 10, 11, 4, 5, 6, 7], [0, 1, 2, 3, 8, 9, 10, 11]][IMM1 as usize],
    )
}

/// Copies `a` to result, then inserts 128 bits (composed of 2 packed
/// double-precision (64-bit) floating-point elements) from `b` into result
/// at the location specified by `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_insertf128_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(
    all(test, not(target_os = "windows")),
    assert_instr(vinsertf128, IMM1 = 1)
)]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_insertf128_pd<const IMM1: i32>(a: __m256d, b: __m128d) -> __m256d {
    static_assert_imm1!(IMM1);
    simd_shuffle4!(
        a,
        _mm256_castpd128_pd256(b),
        <const IMM1: i32> [[4, 5, 2, 3], [0, 1, 4, 5]][IMM1 as usize],
    )
}

/// Copies `a` to result, then inserts 128 bits from `b` into result
/// at the location specified by `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_insertf128_si256)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(
    all(test, not(target_os = "windows")),
    assert_instr(vinsertf128, IMM1 = 1)
)]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_insertf128_si256<const IMM1: i32>(a: __m256i, b: __m128i) -> __m256i {
    static_assert_imm1!(IMM1);
    let dst: i64x4 = simd_shuffle4!(
        a.as_i64x4(),
        _mm256_castsi128_si256(b).as_i64x4(),
        <const IMM1: i32> [[4, 5, 2, 3], [0, 1, 4, 5]][IMM1 as usize],
    );
    transmute(dst)
}

/// Copies `a` to result, and inserts the 8-bit integer `i` into result
/// at the location specified by `index`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_insert_epi8)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_insert_epi8<const INDEX: i32>(a: __m256i, i: i8) -> __m256i {
    static_assert_imm5!(INDEX);
    transmute(simd_insert(a.as_i8x32(), INDEX as u32, i))
}

/// Copies `a` to result, and inserts the 16-bit integer `i` into result
/// at the location specified by `index`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_insert_epi16)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_insert_epi16<const INDEX: i32>(a: __m256i, i: i16) -> __m256i {
    static_assert_imm4!(INDEX);
    transmute(simd_insert(a.as_i16x16(), INDEX as u32, i))
}

/// Copies `a` to result, and inserts the 32-bit integer `i` into result
/// at the location specified by `index`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_insert_epi32)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_insert_epi32<const INDEX: i32>(a: __m256i, i: i32) -> __m256i {
    static_assert_imm3!(INDEX);
    transmute(simd_insert(a.as_i32x8(), INDEX as u32, i))
}

/// Loads 256-bits (composed of 4 packed double-precision (64-bit)
/// floating-point elements) from memory into result.
/// `mem_addr` must be aligned on a 32-byte boundary or a
/// general-protection exception may be generated.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_load_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovaps))] // FIXME vmovapd expected
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow(clippy::cast_ptr_alignment)]
pub unsafe fn _mm256_load_pd(mem_addr: *const f64) -> __m256d {
    *(mem_addr as *const __m256d)
}

/// Stores 256-bits (composed of 4 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` must be aligned on a 32-byte boundary or a
/// general-protection exception may be generated.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_store_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovaps))] // FIXME vmovapd expected
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow(clippy::cast_ptr_alignment)]
pub unsafe fn _mm256_store_pd(mem_addr: *mut f64, a: __m256d) {
    *(mem_addr as *mut __m256d) = a;
}

/// Loads 256-bits (composed of 8 packed single-precision (32-bit)
/// floating-point elements) from memory into result.
/// `mem_addr` must be aligned on a 32-byte boundary or a
/// general-protection exception may be generated.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_load_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovaps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow(clippy::cast_ptr_alignment)]
pub unsafe fn _mm256_load_ps(mem_addr: *const f32) -> __m256 {
    *(mem_addr as *const __m256)
}

/// Stores 256-bits (composed of 8 packed single-precision (32-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` must be aligned on a 32-byte boundary or a
/// general-protection exception may be generated.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_store_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovaps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow(clippy::cast_ptr_alignment)]
pub unsafe fn _mm256_store_ps(mem_addr: *mut f32, a: __m256) {
    *(mem_addr as *mut __m256) = a;
}

/// Loads 256-bits (composed of 4 packed double-precision (64-bit)
/// floating-point elements) from memory into result.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_loadu_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovups))] // FIXME vmovupd expected
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_loadu_pd(mem_addr: *const f64) -> __m256d {
    let mut dst = _mm256_undefined_pd();
    ptr::copy_nonoverlapping(
        mem_addr as *const u8,
        &mut dst as *mut __m256d as *mut u8,
        mem::size_of::<__m256d>(),
    );
    dst
}

/// Stores 256-bits (composed of 4 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_storeu_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovups))] // FIXME vmovupd expected
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_storeu_pd(mem_addr: *mut f64, a: __m256d) {
    storeupd256(mem_addr, a);
}

/// Loads 256-bits (composed of 8 packed single-precision (32-bit)
/// floating-point elements) from memory into result.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_loadu_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovups))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_loadu_ps(mem_addr: *const f32) -> __m256 {
    let mut dst = _mm256_undefined_ps();
    ptr::copy_nonoverlapping(
        mem_addr as *const u8,
        &mut dst as *mut __m256 as *mut u8,
        mem::size_of::<__m256>(),
    );
    dst
}

/// Stores 256-bits (composed of 8 packed single-precision (32-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_storeu_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovups))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_storeu_ps(mem_addr: *mut f32, a: __m256) {
    storeups256(mem_addr, a);
}

/// Loads 256-bits of integer data from memory into result.
/// `mem_addr` must be aligned on a 32-byte boundary or a
/// general-protection exception may be generated.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_load_si256)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovaps))] // FIXME vmovdqa expected
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_load_si256(mem_addr: *const __m256i) -> __m256i {
    *mem_addr
}

/// Stores 256-bits of integer data from `a` into memory.
/// `mem_addr` must be aligned on a 32-byte boundary or a
/// general-protection exception may be generated.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_store_si256)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovaps))] // FIXME vmovdqa expected
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_store_si256(mem_addr: *mut __m256i, a: __m256i) {
    *mem_addr = a;
}

/// Loads 256-bits of integer data from memory into result.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_loadu_si256)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovups))] // FIXME vmovdqu expected
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_loadu_si256(mem_addr: *const __m256i) -> __m256i {
    let mut dst = _mm256_undefined_si256();
    ptr::copy_nonoverlapping(
        mem_addr as *const u8,
        &mut dst as *mut __m256i as *mut u8,
        mem::size_of::<__m256i>(),
    );
    dst
}

/// Stores 256-bits of integer data from `a` into memory.
/// 	`mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_storeu_si256)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovups))] // FIXME vmovdqu expected
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_storeu_si256(mem_addr: *mut __m256i, a: __m256i) {
    storeudq256(mem_addr as *mut i8, a.as_i8x32());
}

/// Loads packed double-precision (64-bit) floating-point elements from memory
/// into result using `mask` (elements are zeroed out when the high bit of the
/// corresponding element is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskload_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmaskmovpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_maskload_pd(mem_addr: *const f64, mask: __m256i) -> __m256d {
    maskloadpd256(mem_addr as *const i8, mask.as_i64x4())
}

/// Stores packed double-precision (64-bit) floating-point elements from `a`
/// into memory using `mask`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskstore_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmaskmovpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_maskstore_pd(mem_addr: *mut f64, mask: __m256i, a: __m256d) {
    maskstorepd256(mem_addr as *mut i8, mask.as_i64x4(), a);
}

/// Loads packed double-precision (64-bit) floating-point elements from memory
/// into result using `mask` (elements are zeroed out when the high bit of the
/// corresponding element is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskload_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmaskmovpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_maskload_pd(mem_addr: *const f64, mask: __m128i) -> __m128d {
    maskloadpd(mem_addr as *const i8, mask.as_i64x2())
}

/// Stores packed double-precision (64-bit) floating-point elements from `a`
/// into memory using `mask`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskstore_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmaskmovpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_maskstore_pd(mem_addr: *mut f64, mask: __m128i, a: __m128d) {
    maskstorepd(mem_addr as *mut i8, mask.as_i64x2(), a);
}

/// Loads packed single-precision (32-bit) floating-point elements from memory
/// into result using `mask` (elements are zeroed out when the high bit of the
/// corresponding element is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskload_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmaskmovps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_maskload_ps(mem_addr: *const f32, mask: __m256i) -> __m256 {
    maskloadps256(mem_addr as *const i8, mask.as_i32x8())
}

/// Stores packed single-precision (32-bit) floating-point elements from `a`
/// into memory using `mask`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskstore_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmaskmovps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_maskstore_ps(mem_addr: *mut f32, mask: __m256i, a: __m256) {
    maskstoreps256(mem_addr as *mut i8, mask.as_i32x8(), a);
}

/// Loads packed single-precision (32-bit) floating-point elements from memory
/// into result using `mask` (elements are zeroed out when the high bit of the
/// corresponding element is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskload_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmaskmovps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_maskload_ps(mem_addr: *const f32, mask: __m128i) -> __m128 {
    maskloadps(mem_addr as *const i8, mask.as_i32x4())
}

/// Stores packed single-precision (32-bit) floating-point elements from `a`
/// into memory using `mask`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskstore_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmaskmovps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_maskstore_ps(mem_addr: *mut f32, mask: __m128i, a: __m128) {
    maskstoreps(mem_addr as *mut i8, mask.as_i32x4(), a);
}

/// Duplicate odd-indexed single-precision (32-bit) floating-point elements
/// from `a`, and returns the results.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_movehdup_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovshdup))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_movehdup_ps(a: __m256) -> __m256 {
    simd_shuffle8!(a, a, [1, 1, 3, 3, 5, 5, 7, 7])
}

/// Duplicate even-indexed single-precision (32-bit) floating-point elements
/// from `a`, and returns the results.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_moveldup_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovsldup))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_moveldup_ps(a: __m256) -> __m256 {
    simd_shuffle8!(a, a, [0, 0, 2, 2, 4, 4, 6, 6])
}

/// Duplicate even-indexed double-precision (64-bit) floating-point elements
/// from `a`, and returns the results.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_movedup_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovddup))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_movedup_pd(a: __m256d) -> __m256d {
    simd_shuffle4!(a, a, [0, 0, 2, 2])
}

/// Loads 256-bits of integer data from unaligned memory into result.
/// This intrinsic may perform better than `_mm256_loadu_si256` when the
/// data crosses a cache line boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_lddqu_si256)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vlddqu))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_lddqu_si256(mem_addr: *const __m256i) -> __m256i {
    transmute(vlddqu(mem_addr as *const i8))
}

/// Moves integer data from a 256-bit integer vector to a 32-byte
/// aligned memory location. To minimize caching, the data is flagged as
/// non-temporal (unlikely to be used again soon)
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_stream_si256)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovntps))] // FIXME vmovntdq
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_stream_si256(mem_addr: *mut __m256i, a: __m256i) {
    intrinsics::nontemporal_store(mem_addr, a);
}

/// Moves double-precision values from a 256-bit vector of `[4 x double]`
/// to a 32-byte aligned memory location. To minimize caching, the data is
/// flagged as non-temporal (unlikely to be used again soon).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_stream_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovntps))] // FIXME vmovntpd
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow(clippy::cast_ptr_alignment)]
pub unsafe fn _mm256_stream_pd(mem_addr: *mut f64, a: __m256d) {
    intrinsics::nontemporal_store(mem_addr as *mut __m256d, a);
}

/// Moves single-precision floating point values from a 256-bit vector
/// of `[8 x float]` to a 32-byte aligned memory location. To minimize
/// caching, the data is flagged as non-temporal (unlikely to be used again
/// soon).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_stream_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovntps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow(clippy::cast_ptr_alignment)]
pub unsafe fn _mm256_stream_ps(mem_addr: *mut f32, a: __m256) {
    intrinsics::nontemporal_store(mem_addr as *mut __m256, a);
}

/// Computes the approximate reciprocal of packed single-precision (32-bit)
/// floating-point elements in `a`, and returns the results. The maximum
/// relative error for this approximation is less than 1.5*2^-12.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_rcp_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vrcpps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_rcp_ps(a: __m256) -> __m256 {
    vrcpps(a)
}

/// Computes the approximate reciprocal square root of packed single-precision
/// (32-bit) floating-point elements in `a`, and returns the results.
/// The maximum relative error for this approximation is less than 1.5*2^-12.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_rsqrt_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vrsqrtps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_rsqrt_ps(a: __m256) -> __m256 {
    vrsqrtps(a)
}

/// Unpacks and interleave double-precision (64-bit) floating-point elements
/// from the high half of each 128-bit lane in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_unpackhi_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vunpckhpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_unpackhi_pd(a: __m256d, b: __m256d) -> __m256d {
    simd_shuffle4!(a, b, [1, 5, 3, 7])
}

/// Unpacks and interleave single-precision (32-bit) floating-point elements
/// from the high half of each 128-bit lane in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_unpackhi_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vunpckhps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_unpackhi_ps(a: __m256, b: __m256) -> __m256 {
    simd_shuffle8!(a, b, [2, 10, 3, 11, 6, 14, 7, 15])
}

/// Unpacks and interleave double-precision (64-bit) floating-point elements
/// from the low half of each 128-bit lane in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_unpacklo_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vunpcklpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_unpacklo_pd(a: __m256d, b: __m256d) -> __m256d {
    simd_shuffle4!(a, b, [0, 4, 2, 6])
}

/// Unpacks and interleave single-precision (32-bit) floating-point elements
/// from the low half of each 128-bit lane in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_unpacklo_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vunpcklps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_unpacklo_ps(a: __m256, b: __m256) -> __m256 {
    simd_shuffle8!(a, b, [0, 8, 1, 9, 4, 12, 5, 13])
}

/// Computes the bitwise AND of 256 bits (representing integer data) in `a` and
/// `b`, and set `ZF` to 1 if the result is zero, otherwise set `ZF` to 0.
/// Computes the bitwise NOT of `a` and then AND with `b`, and set `CF` to 1 if
/// the result is zero, otherwise set `CF` to 0. Return the `ZF` value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_testz_si256)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vptest))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_testz_si256(a: __m256i, b: __m256i) -> i32 {
    ptestz256(a.as_i64x4(), b.as_i64x4())
}

/// Computes the bitwise AND of 256 bits (representing integer data) in `a` and
/// `b`, and set `ZF` to 1 if the result is zero, otherwise set `ZF` to 0.
/// Computes the bitwise NOT of `a` and then AND with `b`, and set `CF` to 1 if
/// the result is zero, otherwise set `CF` to 0. Return the `CF` value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_testc_si256)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vptest))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_testc_si256(a: __m256i, b: __m256i) -> i32 {
    ptestc256(a.as_i64x4(), b.as_i64x4())
}

/// Computes the bitwise AND of 256 bits (representing integer data) in `a` and
/// `b`, and set `ZF` to 1 if the result is zero, otherwise set `ZF` to 0.
/// Computes the bitwise NOT of `a` and then AND with `b`, and set `CF` to 1 if
/// the result is zero, otherwise set `CF` to 0. Return 1 if both the `ZF` and
/// `CF` values are zero, otherwise return 0.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_testnzc_si256)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vptest))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_testnzc_si256(a: __m256i, b: __m256i) -> i32 {
    ptestnzc256(a.as_i64x4(), b.as_i64x4())
}

/// Computes the bitwise AND of 256 bits (representing double-precision (64-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 256-bit
/// value, and set `ZF` to 1 if the sign bit of each 64-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 64-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `ZF` value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_testz_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vtestpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_testz_pd(a: __m256d, b: __m256d) -> i32 {
    vtestzpd256(a, b)
}

/// Computes the bitwise AND of 256 bits (representing double-precision (64-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 256-bit
/// value, and set `ZF` to 1 if the sign bit of each 64-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 64-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `CF` value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_testc_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vtestpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_testc_pd(a: __m256d, b: __m256d) -> i32 {
    vtestcpd256(a, b)
}

/// Computes the bitwise AND of 256 bits (representing double-precision (64-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 256-bit
/// value, and set `ZF` to 1 if the sign bit of each 64-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 64-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return 1 if both the `ZF` and `CF` values
/// are zero, otherwise return 0.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_testnzc_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vtestpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_testnzc_pd(a: __m256d, b: __m256d) -> i32 {
    vtestnzcpd256(a, b)
}

/// Computes the bitwise AND of 128 bits (representing double-precision (64-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 128-bit
/// value, and set `ZF` to 1 if the sign bit of each 64-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 64-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `ZF` value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_testz_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vtestpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_testz_pd(a: __m128d, b: __m128d) -> i32 {
    vtestzpd(a, b)
}

/// Computes the bitwise AND of 128 bits (representing double-precision (64-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 128-bit
/// value, and set `ZF` to 1 if the sign bit of each 64-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 64-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `CF` value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_testc_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vtestpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_testc_pd(a: __m128d, b: __m128d) -> i32 {
    vtestcpd(a, b)
}

/// Computes the bitwise AND of 128 bits (representing double-precision (64-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 128-bit
/// value, and set `ZF` to 1 if the sign bit of each 64-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 64-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return 1 if both the `ZF` and `CF` values
/// are zero, otherwise return 0.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_testnzc_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vtestpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_testnzc_pd(a: __m128d, b: __m128d) -> i32 {
    vtestnzcpd(a, b)
}

/// Computes the bitwise AND of 256 bits (representing single-precision (32-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 256-bit
/// value, and set `ZF` to 1 if the sign bit of each 32-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 32-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `ZF` value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_testz_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vtestps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_testz_ps(a: __m256, b: __m256) -> i32 {
    vtestzps256(a, b)
}

/// Computes the bitwise AND of 256 bits (representing single-precision (32-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 256-bit
/// value, and set `ZF` to 1 if the sign bit of each 32-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 32-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `CF` value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_testc_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vtestps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_testc_ps(a: __m256, b: __m256) -> i32 {
    vtestcps256(a, b)
}

/// Computes the bitwise AND of 256 bits (representing single-precision (32-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 256-bit
/// value, and set `ZF` to 1 if the sign bit of each 32-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 32-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return 1 if both the `ZF` and `CF` values
/// are zero, otherwise return 0.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_testnzc_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vtestps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_testnzc_ps(a: __m256, b: __m256) -> i32 {
    vtestnzcps256(a, b)
}

/// Computes the bitwise AND of 128 bits (representing single-precision (32-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 128-bit
/// value, and set `ZF` to 1 if the sign bit of each 32-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 32-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `ZF` value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_testz_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vtestps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_testz_ps(a: __m128, b: __m128) -> i32 {
    vtestzps(a, b)
}

/// Computes the bitwise AND of 128 bits (representing single-precision (32-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 128-bit
/// value, and set `ZF` to 1 if the sign bit of each 32-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 32-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return the `CF` value.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_testc_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vtestps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_testc_ps(a: __m128, b: __m128) -> i32 {
    vtestcps(a, b)
}

/// Computes the bitwise AND of 128 bits (representing single-precision (32-bit)
/// floating-point elements) in `a` and `b`, producing an intermediate 128-bit
/// value, and set `ZF` to 1 if the sign bit of each 32-bit element in the
/// intermediate value is zero, otherwise set `ZF` to 0. Compute the bitwise
/// NOT of `a` and then AND with `b`, producing an intermediate value, and set
/// `CF` to 1 if the sign bit of each 32-bit element in the intermediate value
/// is zero, otherwise set `CF` to 0. Return 1 if both the `ZF` and `CF` values
/// are zero, otherwise return 0.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_testnzc_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vtestps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_testnzc_ps(a: __m128, b: __m128) -> i32 {
    vtestnzcps(a, b)
}

/// Sets each bit of the returned mask based on the most significant bit of the
/// corresponding packed double-precision (64-bit) floating-point element in
/// `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_movemask_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovmskpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_movemask_pd(a: __m256d) -> i32 {
    movmskpd256(a)
}

/// Sets each bit of the returned mask based on the most significant bit of the
/// corresponding packed single-precision (32-bit) floating-point element in
/// `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_movemask_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vmovmskps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_movemask_ps(a: __m256) -> i32 {
    movmskps256(a)
}

/// Returns vector of type __m256d with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setzero_pd)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vxorps))] // FIXME vxorpd expected
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_setzero_pd() -> __m256d {
    _mm256_set1_pd(0.0)
}

/// Returns vector of type __m256 with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setzero_ps)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vxorps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_setzero_ps() -> __m256 {
    _mm256_set1_ps(0.0)
}

/// Returns vector of type __m256i with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setzero_si256)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vxor))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_setzero_si256() -> __m256i {
    _mm256_set1_epi8(0)
}

/// Sets packed double-precision (64-bit) floating-point elements in returned
/// vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set_pd)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[cfg_attr(test, assert_instr(vinsertf128))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set_pd(a: f64, b: f64, c: f64, d: f64) -> __m256d {
    _mm256_setr_pd(d, c, b, a)
}

/// Sets packed single-precision (32-bit) floating-point elements in returned
/// vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set_ps)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set_ps(
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32,
) -> __m256 {
    _mm256_setr_ps(h, g, f, e, d, c, b, a)
}

/// Sets packed 8-bit integers in returned vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set_epi8)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set_epi8(
    e00: i8,
    e01: i8,
    e02: i8,
    e03: i8,
    e04: i8,
    e05: i8,
    e06: i8,
    e07: i8,
    e08: i8,
    e09: i8,
    e10: i8,
    e11: i8,
    e12: i8,
    e13: i8,
    e14: i8,
    e15: i8,
    e16: i8,
    e17: i8,
    e18: i8,
    e19: i8,
    e20: i8,
    e21: i8,
    e22: i8,
    e23: i8,
    e24: i8,
    e25: i8,
    e26: i8,
    e27: i8,
    e28: i8,
    e29: i8,
    e30: i8,
    e31: i8,
) -> __m256i {
    #[rustfmt::skip]
    _mm256_setr_epi8(
        e31, e30, e29, e28, e27, e26, e25, e24,
        e23, e22, e21, e20, e19, e18, e17, e16,
        e15, e14, e13, e12, e11, e10, e09, e08,
        e07, e06, e05, e04, e03, e02, e01, e00,
    )
}

/// Sets packed 16-bit integers in returned vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set_epi16)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set_epi16(
    e00: i16,
    e01: i16,
    e02: i16,
    e03: i16,
    e04: i16,
    e05: i16,
    e06: i16,
    e07: i16,
    e08: i16,
    e09: i16,
    e10: i16,
    e11: i16,
    e12: i16,
    e13: i16,
    e14: i16,
    e15: i16,
) -> __m256i {
    #[rustfmt::skip]
    _mm256_setr_epi16(
        e15, e14, e13, e12,
        e11, e10, e09, e08,
        e07, e06, e05, e04,
        e03, e02, e01, e00,
    )
}

/// Sets packed 32-bit integers in returned vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set_epi32)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set_epi32(
    e0: i32,
    e1: i32,
    e2: i32,
    e3: i32,
    e4: i32,
    e5: i32,
    e6: i32,
    e7: i32,
) -> __m256i {
    _mm256_setr_epi32(e7, e6, e5, e4, e3, e2, e1, e0)
}

/// Sets packed 64-bit integers in returned vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set_epi64x)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set_epi64x(a: i64, b: i64, c: i64, d: i64) -> __m256i {
    _mm256_setr_epi64x(d, c, b, a)
}

/// Sets packed double-precision (64-bit) floating-point elements in returned
/// vector with the supplied values in reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setr_pd)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_setr_pd(a: f64, b: f64, c: f64, d: f64) -> __m256d {
    __m256d(a, b, c, d)
}

/// Sets packed single-precision (32-bit) floating-point elements in returned
/// vector with the supplied values in reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setr_ps)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_setr_ps(
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32,
) -> __m256 {
    __m256(a, b, c, d, e, f, g, h)
}

/// Sets packed 8-bit integers in returned vector with the supplied values in
/// reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setr_epi8)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_setr_epi8(
    e00: i8,
    e01: i8,
    e02: i8,
    e03: i8,
    e04: i8,
    e05: i8,
    e06: i8,
    e07: i8,
    e08: i8,
    e09: i8,
    e10: i8,
    e11: i8,
    e12: i8,
    e13: i8,
    e14: i8,
    e15: i8,
    e16: i8,
    e17: i8,
    e18: i8,
    e19: i8,
    e20: i8,
    e21: i8,
    e22: i8,
    e23: i8,
    e24: i8,
    e25: i8,
    e26: i8,
    e27: i8,
    e28: i8,
    e29: i8,
    e30: i8,
    e31: i8,
) -> __m256i {
    #[rustfmt::skip]
    transmute(i8x32::new(
        e00, e01, e02, e03, e04, e05, e06, e07,
        e08, e09, e10, e11, e12, e13, e14, e15,
        e16, e17, e18, e19, e20, e21, e22, e23,
        e24, e25, e26, e27, e28, e29, e30, e31,
    ))
}

/// Sets packed 16-bit integers in returned vector with the supplied values in
/// reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setr_epi16)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_setr_epi16(
    e00: i16,
    e01: i16,
    e02: i16,
    e03: i16,
    e04: i16,
    e05: i16,
    e06: i16,
    e07: i16,
    e08: i16,
    e09: i16,
    e10: i16,
    e11: i16,
    e12: i16,
    e13: i16,
    e14: i16,
    e15: i16,
) -> __m256i {
    #[rustfmt::skip]
    transmute(i16x16::new(
        e00, e01, e02, e03,
        e04, e05, e06, e07,
        e08, e09, e10, e11,
        e12, e13, e14, e15,
    ))
}

/// Sets packed 32-bit integers in returned vector with the supplied values in
/// reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setr_epi32)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_setr_epi32(
    e0: i32,
    e1: i32,
    e2: i32,
    e3: i32,
    e4: i32,
    e5: i32,
    e6: i32,
    e7: i32,
) -> __m256i {
    transmute(i32x8::new(e0, e1, e2, e3, e4, e5, e6, e7))
}

/// Sets packed 64-bit integers in returned vector with the supplied values in
/// reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setr_epi64x)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_setr_epi64x(a: i64, b: i64, c: i64, d: i64) -> __m256i {
    transmute(i64x4::new(a, b, c, d))
}

/// Broadcasts double-precision (64-bit) floating-point value `a` to all
/// elements of returned vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set1_pd)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set1_pd(a: f64) -> __m256d {
    _mm256_setr_pd(a, a, a, a)
}

/// Broadcasts single-precision (32-bit) floating-point value `a` to all
/// elements of returned vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set1_ps)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set1_ps(a: f32) -> __m256 {
    _mm256_setr_ps(a, a, a, a, a, a, a, a)
}

/// Broadcasts 8-bit integer `a` to all elements of returned vector.
/// This intrinsic may generate the `vpbroadcastb`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set1_epi8)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vpshufb))]
#[cfg_attr(test, assert_instr(vinsertf128))]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set1_epi8(a: i8) -> __m256i {
    #[rustfmt::skip]
    _mm256_setr_epi8(
        a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a,
    )
}

/// Broadcasts 16-bit integer `a` to all all elements of returned vector.
/// This intrinsic may generate the `vpbroadcastw`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set1_epi16)
#[inline]
#[target_feature(enable = "avx")]
//#[cfg_attr(test, assert_instr(vpshufb))]
#[cfg_attr(test, assert_instr(vinsertf128))]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set1_epi16(a: i16) -> __m256i {
    _mm256_setr_epi16(a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a)
}

/// Broadcasts 32-bit integer `a` to all elements of returned vector.
/// This intrinsic may generate the `vpbroadcastd`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set1_epi32)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set1_epi32(a: i32) -> __m256i {
    _mm256_setr_epi32(a, a, a, a, a, a, a, a)
}

/// Broadcasts 64-bit integer `a` to all elements of returned vector.
/// This intrinsic may generate the `vpbroadcastq`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set1_epi64x)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(all(test, target_arch = "x86_64"), assert_instr(vinsertf128))]
#[cfg_attr(all(test, target_arch = "x86"), assert_instr(vbroadcastsd))]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set1_epi64x(a: i64) -> __m256i {
    _mm256_setr_epi64x(a, a, a, a)
}

/// Cast vector of type __m256d to type __m256.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castpd_ps)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_castpd_ps(a: __m256d) -> __m256 {
    transmute(a)
}

/// Cast vector of type __m256 to type __m256d.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castps_pd)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_castps_pd(a: __m256) -> __m256d {
    transmute(a)
}

/// Casts vector of type __m256 to type __m256i.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castps_si256)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_castps_si256(a: __m256) -> __m256i {
    transmute(a)
}

/// Casts vector of type __m256i to type __m256.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castsi256_ps)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_castsi256_ps(a: __m256i) -> __m256 {
    transmute(a)
}

/// Casts vector of type __m256d to type __m256i.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castpd_si256)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_castpd_si256(a: __m256d) -> __m256i {
    transmute(a)
}

/// Casts vector of type __m256i to type __m256d.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castsi256_pd)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_castsi256_pd(a: __m256i) -> __m256d {
    transmute(a)
}

/// Casts vector of type __m256 to type __m128.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castps256_ps128)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_castps256_ps128(a: __m256) -> __m128 {
    simd_shuffle4!(a, a, [0, 1, 2, 3])
}

/// Casts vector of type __m256d to type __m128d.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castpd256_pd128)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_castpd256_pd128(a: __m256d) -> __m128d {
    simd_shuffle2!(a, a, [0, 1])
}

/// Casts vector of type __m256i to type __m128i.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castsi256_si128)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_castsi256_si128(a: __m256i) -> __m128i {
    let a = a.as_i64x4();
    let dst: i64x2 = simd_shuffle2!(a, a, [0, 1]);
    transmute(dst)
}

/// Casts vector of type __m128 to type __m256;
/// the upper 128 bits of the result are undefined.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castps128_ps256)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_castps128_ps256(a: __m128) -> __m256 {
    // FIXME simd_shuffle8!(a, a, [0, 1, 2, 3, -1, -1, -1, -1])
    simd_shuffle8!(a, a, [0, 1, 2, 3, 0, 0, 0, 0])
}

/// Casts vector of type __m128d to type __m256d;
/// the upper 128 bits of the result are undefined.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castpd128_pd256)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_castpd128_pd256(a: __m128d) -> __m256d {
    // FIXME simd_shuffle4!(a, a, [0, 1, -1, -1])
    simd_shuffle4!(a, a, [0, 1, 0, 0])
}

/// Casts vector of type __m128i to type __m256i;
/// the upper 128 bits of the result are undefined.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_castsi128_si256)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_castsi128_si256(a: __m128i) -> __m256i {
    let a = a.as_i64x2();
    // FIXME simd_shuffle4!(a, a, [0, 1, -1, -1])
    let dst: i64x4 = simd_shuffle4!(a, a, [0, 1, 0, 0]);
    transmute(dst)
}

/// Constructs a 256-bit floating-point vector of `[8 x float]` from a
/// 128-bit floating-point vector of `[4 x float]`. The lower 128 bits contain
/// the value of the source vector. The upper 128 bits are set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_zextps128_ps256)
#[inline]
#[target_feature(enable = "avx,sse")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_zextps128_ps256(a: __m128) -> __m256 {
    simd_shuffle8!(a, _mm_setzero_ps(), [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Constructs a 256-bit integer vector from a 128-bit integer vector.
/// The lower 128 bits contain the value of the source vector. The upper
/// 128 bits are set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_zextsi128_si256)
#[inline]
#[target_feature(enable = "avx,sse2")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_zextsi128_si256(a: __m128i) -> __m256i {
    let b = _mm_setzero_si128().as_i64x2();
    let dst: i64x4 = simd_shuffle4!(a.as_i64x2(), b, [0, 1, 2, 3]);
    transmute(dst)
}

/// Constructs a 256-bit floating-point vector of `[4 x double]` from a
/// 128-bit floating-point vector of `[2 x double]`. The lower 128 bits
/// contain the value of the source vector. The upper 128 bits are set
/// to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_zextpd128_pd256)
#[inline]
#[target_feature(enable = "avx,sse2")]
// This intrinsic is only used for compilation and does not generate any
// instructions, thus it has zero latency.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_zextpd128_pd256(a: __m128d) -> __m256d {
    simd_shuffle4!(a, _mm_setzero_pd(), [0, 1, 2, 3])
}

/// Returns vector of type `__m256` with undefined elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_undefined_ps)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_undefined_ps() -> __m256 {
    _mm256_set1_ps(0.0)
}

/// Returns vector of type `__m256d` with undefined elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_undefined_pd)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_undefined_pd() -> __m256d {
    _mm256_set1_pd(0.0)
}

/// Returns vector of type __m256i with undefined elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_undefined_si256)
#[inline]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_undefined_si256() -> __m256i {
    // FIXME: this function should return MaybeUninit<__m256i>
    mem::MaybeUninit::<__m256i>::uninit().assume_init()
}

/// Sets packed __m256 returned vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set_m128)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vinsertf128))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set_m128(hi: __m128, lo: __m128) -> __m256 {
    simd_shuffle8!(lo, hi, [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Sets packed __m256d returned vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set_m128d)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vinsertf128))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set_m128d(hi: __m128d, lo: __m128d) -> __m256d {
    let hi: __m128 = transmute(hi);
    let lo: __m128 = transmute(lo);
    transmute(_mm256_set_m128(hi, lo))
}

/// Sets packed __m256i returned vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set_m128i)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vinsertf128))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_set_m128i(hi: __m128i, lo: __m128i) -> __m256i {
    let hi: __m128 = transmute(hi);
    let lo: __m128 = transmute(lo);
    transmute(_mm256_set_m128(hi, lo))
}

/// Sets packed __m256 returned vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setr_m128)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vinsertf128))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_setr_m128(lo: __m128, hi: __m128) -> __m256 {
    _mm256_set_m128(hi, lo)
}

/// Sets packed __m256d returned vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setr_m128d)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vinsertf128))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_setr_m128d(lo: __m128d, hi: __m128d) -> __m256d {
    _mm256_set_m128d(hi, lo)
}

/// Sets packed __m256i returned vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setr_m128i)
#[inline]
#[target_feature(enable = "avx")]
#[cfg_attr(test, assert_instr(vinsertf128))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_setr_m128i(lo: __m128i, hi: __m128i) -> __m256i {
    _mm256_set_m128i(hi, lo)
}

/// Loads two 128-bit values (composed of 4 packed single-precision (32-bit)
/// floating-point elements) from memory, and combine them into a 256-bit
/// value.
/// `hiaddr` and `loaddr` do not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_loadu2_m128)
#[inline]
#[target_feature(enable = "avx,sse")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_loadu2_m128(hiaddr: *const f32, loaddr: *const f32) -> __m256 {
    let a = _mm256_castps128_ps256(_mm_loadu_ps(loaddr));
    _mm256_insertf128_ps::<1>(a, _mm_loadu_ps(hiaddr))
}

/// Loads two 128-bit values (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from memory, and combine them into a 256-bit
/// value.
/// `hiaddr` and `loaddr` do not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_loadu2_m128d)
#[inline]
#[target_feature(enable = "avx,sse2")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_loadu2_m128d(hiaddr: *const f64, loaddr: *const f64) -> __m256d {
    let a = _mm256_castpd128_pd256(_mm_loadu_pd(loaddr));
    _mm256_insertf128_pd::<1>(a, _mm_loadu_pd(hiaddr))
}

/// Loads two 128-bit values (composed of integer data) from memory, and combine
/// them into a 256-bit value.
/// `hiaddr` and `loaddr` do not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_loadu2_m128i)
#[inline]
#[target_feature(enable = "avx,sse2")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_loadu2_m128i(hiaddr: *const __m128i, loaddr: *const __m128i) -> __m256i {
    let a = _mm256_castsi128_si256(_mm_loadu_si128(loaddr));
    _mm256_insertf128_si256::<1>(a, _mm_loadu_si128(hiaddr))
}

/// Stores the high and low 128-bit halves (each composed of 4 packed
/// single-precision (32-bit) floating-point elements) from `a` into memory two
/// different 128-bit locations.
/// `hiaddr` and `loaddr` do not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_storeu2_m128)
#[inline]
#[target_feature(enable = "avx,sse")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_storeu2_m128(hiaddr: *mut f32, loaddr: *mut f32, a: __m256) {
    let lo = _mm256_castps256_ps128(a);
    _mm_storeu_ps(loaddr, lo);
    let hi = _mm256_extractf128_ps::<1>(a);
    _mm_storeu_ps(hiaddr, hi);
}

/// Stores the high and low 128-bit halves (each composed of 2 packed
/// double-precision (64-bit) floating-point elements) from `a` into memory two
/// different 128-bit locations.
/// `hiaddr` and `loaddr` do not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_storeu2_m128d)
#[inline]
#[target_feature(enable = "avx,sse2")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_storeu2_m128d(hiaddr: *mut f64, loaddr: *mut f64, a: __m256d) {
    let lo = _mm256_castpd256_pd128(a);
    _mm_storeu_pd(loaddr, lo);
    let hi = _mm256_extractf128_pd::<1>(a);
    _mm_storeu_pd(hiaddr, hi);
}

/// Stores the high and low 128-bit halves (each composed of integer data) from
/// `a` into memory two different 128-bit locations.
/// `hiaddr` and `loaddr` do not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_storeu2_m128i)
#[inline]
#[target_feature(enable = "avx,sse2")]
// This intrinsic has no corresponding instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_storeu2_m128i(hiaddr: *mut __m128i, loaddr: *mut __m128i, a: __m256i) {
    let lo = _mm256_castsi256_si128(a);
    _mm_storeu_si128(loaddr, lo);
    let hi = _mm256_extractf128_si256::<1>(a);
    _mm_storeu_si128(hiaddr, hi);
}

/// Returns the first element of the input vector of `[8 x float]`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cvtss_f32)
#[inline]
#[target_feature(enable = "avx")]
//#[cfg_attr(test, assert_instr(movss))] FIXME
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm256_cvtss_f32(a: __m256) -> f32 {
    simd_extract(a, 0)
}

/// LLVM intrinsics used in the above functions
#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx.addsub.pd.256"]
    fn addsubpd256(a: __m256d, b: __m256d) -> __m256d;
    #[link_name = "llvm.x86.avx.addsub.ps.256"]
    fn addsubps256(a: __m256, b: __m256) -> __m256;
    #[link_name = "llvm.x86.avx.round.pd.256"]
    fn roundpd256(a: __m256d, b: i32) -> __m256d;
    #[link_name = "llvm.x86.avx.round.ps.256"]
    fn roundps256(a: __m256, b: i32) -> __m256;
    #[link_name = "llvm.x86.avx.sqrt.ps.256"]
    fn sqrtps256(a: __m256) -> __m256;
    #[link_name = "llvm.x86.avx.blendv.pd.256"]
    fn vblendvpd(a: __m256d, b: __m256d, c: __m256d) -> __m256d;
    #[link_name = "llvm.x86.avx.blendv.ps.256"]
    fn vblendvps(a: __m256, b: __m256, c: __m256) -> __m256;
    #[link_name = "llvm.x86.avx.dp.ps.256"]
    fn vdpps(a: __m256, b: __m256, imm8: i32) -> __m256;
    #[link_name = "llvm.x86.avx.hadd.pd.256"]
    fn vhaddpd(a: __m256d, b: __m256d) -> __m256d;
    #[link_name = "llvm.x86.avx.hadd.ps.256"]
    fn vhaddps(a: __m256, b: __m256) -> __m256;
    #[link_name = "llvm.x86.avx.hsub.pd.256"]
    fn vhsubpd(a: __m256d, b: __m256d) -> __m256d;
    #[link_name = "llvm.x86.avx.hsub.ps.256"]
    fn vhsubps(a: __m256, b: __m256) -> __m256;
    #[link_name = "llvm.x86.sse2.cmp.pd"]
    fn vcmppd(a: __m128d, b: __m128d, imm8: i8) -> __m128d;
    #[link_name = "llvm.x86.avx.cmp.pd.256"]
    fn vcmppd256(a: __m256d, b: __m256d, imm8: u8) -> __m256d;
    #[link_name = "llvm.x86.sse.cmp.ps"]
    fn vcmpps(a: __m128, b: __m128, imm8: i8) -> __m128;
    #[link_name = "llvm.x86.avx.cmp.ps.256"]
    fn vcmpps256(a: __m256, b: __m256, imm8: u8) -> __m256;
    #[link_name = "llvm.x86.sse2.cmp.sd"]
    fn vcmpsd(a: __m128d, b: __m128d, imm8: i8) -> __m128d;
    #[link_name = "llvm.x86.sse.cmp.ss"]
    fn vcmpss(a: __m128, b: __m128, imm8: i8) -> __m128;
    #[link_name = "llvm.x86.avx.cvtdq2.ps.256"]
    fn vcvtdq2ps(a: i32x8) -> __m256;
    #[link_name = "llvm.x86.avx.cvt.pd2.ps.256"]
    fn vcvtpd2ps(a: __m256d) -> __m128;
    #[link_name = "llvm.x86.avx.cvt.ps2dq.256"]
    fn vcvtps2dq(a: __m256) -> i32x8;
    #[link_name = "llvm.x86.avx.cvtt.pd2dq.256"]
    fn vcvttpd2dq(a: __m256d) -> i32x4;
    #[link_name = "llvm.x86.avx.cvt.pd2dq.256"]
    fn vcvtpd2dq(a: __m256d) -> i32x4;
    #[link_name = "llvm.x86.avx.cvtt.ps2dq.256"]
    fn vcvttps2dq(a: __m256) -> i32x8;
    #[link_name = "llvm.x86.avx.vzeroall"]
    fn vzeroall();
    #[link_name = "llvm.x86.avx.vzeroupper"]
    fn vzeroupper();
    #[link_name = "llvm.x86.avx.vpermilvar.ps.256"]
    fn vpermilps256(a: __m256, b: i32x8) -> __m256;
    #[link_name = "llvm.x86.avx.vpermilvar.ps"]
    fn vpermilps(a: __m128, b: i32x4) -> __m128;
    #[link_name = "llvm.x86.avx.vpermilvar.pd.256"]
    fn vpermilpd256(a: __m256d, b: i64x4) -> __m256d;
    #[link_name = "llvm.x86.avx.vpermilvar.pd"]
    fn vpermilpd(a: __m128d, b: i64x2) -> __m128d;
    #[link_name = "llvm.x86.avx.vperm2f128.ps.256"]
    fn vperm2f128ps256(a: __m256, b: __m256, imm8: i8) -> __m256;
    #[link_name = "llvm.x86.avx.vperm2f128.pd.256"]
    fn vperm2f128pd256(a: __m256d, b: __m256d, imm8: i8) -> __m256d;
    #[link_name = "llvm.x86.avx.vperm2f128.si.256"]
    fn vperm2f128si256(a: i32x8, b: i32x8, imm8: i8) -> i32x8;
    #[link_name = "llvm.x86.avx.vbroadcastf128.ps.256"]
    fn vbroadcastf128ps256(a: &__m128) -> __m256;
    #[link_name = "llvm.x86.avx.vbroadcastf128.pd.256"]
    fn vbroadcastf128pd256(a: &__m128d) -> __m256d;
    #[link_name = "llvm.x86.avx.storeu.pd.256"]
    fn storeupd256(mem_addr: *mut f64, a: __m256d);
    #[link_name = "llvm.x86.avx.storeu.ps.256"]
    fn storeups256(mem_addr: *mut f32, a: __m256);
    #[link_name = "llvm.x86.avx.storeu.dq.256"]
    fn storeudq256(mem_addr: *mut i8, a: i8x32);
    #[link_name = "llvm.x86.avx.maskload.pd.256"]
    fn maskloadpd256(mem_addr: *const i8, mask: i64x4) -> __m256d;
    #[link_name = "llvm.x86.avx.maskstore.pd.256"]
    fn maskstorepd256(mem_addr: *mut i8, mask: i64x4, a: __m256d);
    #[link_name = "llvm.x86.avx.maskload.pd"]
    fn maskloadpd(mem_addr: *const i8, mask: i64x2) -> __m128d;
    #[link_name = "llvm.x86.avx.maskstore.pd"]
    fn maskstorepd(mem_addr: *mut i8, mask: i64x2, a: __m128d);
    #[link_name = "llvm.x86.avx.maskload.ps.256"]
    fn maskloadps256(mem_addr: *const i8, mask: i32x8) -> __m256;
    #[link_name = "llvm.x86.avx.maskstore.ps.256"]
    fn maskstoreps256(mem_addr: *mut i8, mask: i32x8, a: __m256);
    #[link_name = "llvm.x86.avx.maskload.ps"]
    fn maskloadps(mem_addr: *const i8, mask: i32x4) -> __m128;
    #[link_name = "llvm.x86.avx.maskstore.ps"]
    fn maskstoreps(mem_addr: *mut i8, mask: i32x4, a: __m128);
    #[link_name = "llvm.x86.avx.ldu.dq.256"]
    fn vlddqu(mem_addr: *const i8) -> i8x32;
    #[link_name = "llvm.x86.avx.rcp.ps.256"]
    fn vrcpps(a: __m256) -> __m256;
    #[link_name = "llvm.x86.avx.rsqrt.ps.256"]
    fn vrsqrtps(a: __m256) -> __m256;
    #[link_name = "llvm.x86.avx.ptestz.256"]
    fn ptestz256(a: i64x4, b: i64x4) -> i32;
    #[link_name = "llvm.x86.avx.ptestc.256"]
    fn ptestc256(a: i64x4, b: i64x4) -> i32;
    #[link_name = "llvm.x86.avx.ptestnzc.256"]
    fn ptestnzc256(a: i64x4, b: i64x4) -> i32;
    #[link_name = "llvm.x86.avx.vtestz.pd.256"]
    fn vtestzpd256(a: __m256d, b: __m256d) -> i32;
    #[link_name = "llvm.x86.avx.vtestc.pd.256"]
    fn vtestcpd256(a: __m256d, b: __m256d) -> i32;
    #[link_name = "llvm.x86.avx.vtestnzc.pd.256"]
    fn vtestnzcpd256(a: __m256d, b: __m256d) -> i32;
    #[link_name = "llvm.x86.avx.vtestz.pd"]
    fn vtestzpd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.avx.vtestc.pd"]
    fn vtestcpd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.avx.vtestnzc.pd"]
    fn vtestnzcpd(a: __m128d, b: __m128d) -> i32;
    #[link_name = "llvm.x86.avx.vtestz.ps.256"]
    fn vtestzps256(a: __m256, b: __m256) -> i32;
    #[link_name = "llvm.x86.avx.vtestc.ps.256"]
    fn vtestcps256(a: __m256, b: __m256) -> i32;
    #[link_name = "llvm.x86.avx.vtestnzc.ps.256"]
    fn vtestnzcps256(a: __m256, b: __m256) -> i32;
    #[link_name = "llvm.x86.avx.vtestz.ps"]
    fn vtestzps(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.avx.vtestc.ps"]
    fn vtestcps(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.avx.vtestnzc.ps"]
    fn vtestnzcps(a: __m128, b: __m128) -> i32;
    #[link_name = "llvm.x86.avx.movmsk.pd.256"]
    fn movmskpd256(a: __m256d) -> i32;
    #[link_name = "llvm.x86.avx.movmsk.ps.256"]
    fn movmskps256(a: __m256) -> i32;
    #[link_name = "llvm.x86.avx.min.ps.256"]
    fn vminps(a: __m256, b: __m256) -> __m256;
    #[link_name = "llvm.x86.avx.max.ps.256"]
    fn vmaxps(a: __m256, b: __m256) -> __m256;
    #[link_name = "llvm.x86.avx.min.pd.256"]
    fn vminpd(a: __m256d, b: __m256d) -> __m256d;
    #[link_name = "llvm.x86.avx.max.pd.256"]
    fn vmaxpd(a: __m256d, b: __m256d) -> __m256d;
}

#[cfg(test)]
mod tests {
    use crate::hint::black_box;
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_add_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_add_pd(a, b);
        let e = _mm256_setr_pd(6., 8., 10., 12.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_add_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm256_setr_ps(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm256_add_ps(a, b);
        let e = _mm256_setr_ps(10., 12., 14., 16., 18., 20., 22., 24.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_and_pd() {
        let a = _mm256_set1_pd(1.);
        let b = _mm256_set1_pd(0.6);
        let r = _mm256_and_pd(a, b);
        let e = _mm256_set1_pd(0.5);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_and_ps() {
        let a = _mm256_set1_ps(1.);
        let b = _mm256_set1_ps(0.6);
        let r = _mm256_and_ps(a, b);
        let e = _mm256_set1_ps(0.5);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_or_pd() {
        let a = _mm256_set1_pd(1.);
        let b = _mm256_set1_pd(0.6);
        let r = _mm256_or_pd(a, b);
        let e = _mm256_set1_pd(1.2);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_or_ps() {
        let a = _mm256_set1_ps(1.);
        let b = _mm256_set1_ps(0.6);
        let r = _mm256_or_ps(a, b);
        let e = _mm256_set1_ps(1.2);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_shuffle_pd() {
        let a = _mm256_setr_pd(1., 4., 5., 8.);
        let b = _mm256_setr_pd(2., 3., 6., 7.);
        let r = _mm256_shuffle_pd::<0b11_11_11_11>(a, b);
        let e = _mm256_setr_pd(4., 3., 8., 7.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_shuffle_ps() {
        let a = _mm256_setr_ps(1., 4., 5., 8., 9., 12., 13., 16.);
        let b = _mm256_setr_ps(2., 3., 6., 7., 10., 11., 14., 15.);
        let r = _mm256_shuffle_ps::<0b00_00_11_11>(a, b);
        let e = _mm256_setr_ps(8., 8., 2., 2., 16., 16., 10., 10.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_andnot_pd() {
        let a = _mm256_set1_pd(0.);
        let b = _mm256_set1_pd(0.6);
        let r = _mm256_andnot_pd(a, b);
        assert_eq_m256d(r, b);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_andnot_ps() {
        let a = _mm256_set1_ps(0.);
        let b = _mm256_set1_ps(0.6);
        let r = _mm256_andnot_ps(a, b);
        assert_eq_m256(r, b);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_max_pd() {
        let a = _mm256_setr_pd(1., 4., 5., 8.);
        let b = _mm256_setr_pd(2., 3., 6., 7.);
        let r = _mm256_max_pd(a, b);
        let e = _mm256_setr_pd(2., 4., 6., 8.);
        assert_eq_m256d(r, e);
        // > If the values being compared are both 0.0s (of either sign), the
        // > value in the second operand (source operand) is returned.
        let w = _mm256_max_pd(_mm256_set1_pd(0.0), _mm256_set1_pd(-0.0));
        let x = _mm256_max_pd(_mm256_set1_pd(-0.0), _mm256_set1_pd(0.0));
        let wu: [u64; 4] = transmute(w);
        let xu: [u64; 4] = transmute(x);
        assert_eq!(wu, [0x8000_0000_0000_0000u64; 4]);
        assert_eq!(xu, [0u64; 4]);
        // > If only one value is a NaN (SNaN or QNaN) for this instruction, the
        // > second operand (source operand), either a NaN or a valid
        // > floating-point value, is written to the result.
        let y = _mm256_max_pd(_mm256_set1_pd(f64::NAN), _mm256_set1_pd(0.0));
        let z = _mm256_max_pd(_mm256_set1_pd(0.0), _mm256_set1_pd(f64::NAN));
        let yf: [f64; 4] = transmute(y);
        let zf: [f64; 4] = transmute(z);
        assert_eq!(yf, [0.0; 4]);
        assert!(zf.iter().all(|f| f.is_nan()), "{:?}", zf);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_max_ps() {
        let a = _mm256_setr_ps(1., 4., 5., 8., 9., 12., 13., 16.);
        let b = _mm256_setr_ps(2., 3., 6., 7., 10., 11., 14., 15.);
        let r = _mm256_max_ps(a, b);
        let e = _mm256_setr_ps(2., 4., 6., 8., 10., 12., 14., 16.);
        assert_eq_m256(r, e);
        // > If the values being compared are both 0.0s (of either sign), the
        // > value in the second operand (source operand) is returned.
        let w = _mm256_max_ps(_mm256_set1_ps(0.0), _mm256_set1_ps(-0.0));
        let x = _mm256_max_ps(_mm256_set1_ps(-0.0), _mm256_set1_ps(0.0));
        let wu: [u32; 8] = transmute(w);
        let xu: [u32; 8] = transmute(x);
        assert_eq!(wu, [0x8000_0000u32; 8]);
        assert_eq!(xu, [0u32; 8]);
        // > If only one value is a NaN (SNaN or QNaN) for this instruction, the
        // > second operand (source operand), either a NaN or a valid
        // > floating-point value, is written to the result.
        let y = _mm256_max_ps(_mm256_set1_ps(f32::NAN), _mm256_set1_ps(0.0));
        let z = _mm256_max_ps(_mm256_set1_ps(0.0), _mm256_set1_ps(f32::NAN));
        let yf: [f32; 8] = transmute(y);
        let zf: [f32; 8] = transmute(z);
        assert_eq!(yf, [0.0; 8]);
        assert!(zf.iter().all(|f| f.is_nan()), "{:?}", zf);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_min_pd() {
        let a = _mm256_setr_pd(1., 4., 5., 8.);
        let b = _mm256_setr_pd(2., 3., 6., 7.);
        let r = _mm256_min_pd(a, b);
        let e = _mm256_setr_pd(1., 3., 5., 7.);
        assert_eq_m256d(r, e);
        // > If the values being compared are both 0.0s (of either sign), the
        // > value in the second operand (source operand) is returned.
        let w = _mm256_min_pd(_mm256_set1_pd(0.0), _mm256_set1_pd(-0.0));
        let x = _mm256_min_pd(_mm256_set1_pd(-0.0), _mm256_set1_pd(0.0));
        let wu: [u64; 4] = transmute(w);
        let xu: [u64; 4] = transmute(x);
        assert_eq!(wu, [0x8000_0000_0000_0000u64; 4]);
        assert_eq!(xu, [0u64; 4]);
        // > If only one value is a NaN (SNaN or QNaN) for this instruction, the
        // > second operand (source operand), either a NaN or a valid
        // > floating-point value, is written to the result.
        let y = _mm256_min_pd(_mm256_set1_pd(f64::NAN), _mm256_set1_pd(0.0));
        let z = _mm256_min_pd(_mm256_set1_pd(0.0), _mm256_set1_pd(f64::NAN));
        let yf: [f64; 4] = transmute(y);
        let zf: [f64; 4] = transmute(z);
        assert_eq!(yf, [0.0; 4]);
        assert!(zf.iter().all(|f| f.is_nan()), "{:?}", zf);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_min_ps() {
        let a = _mm256_setr_ps(1., 4., 5., 8., 9., 12., 13., 16.);
        let b = _mm256_setr_ps(2., 3., 6., 7., 10., 11., 14., 15.);
        let r = _mm256_min_ps(a, b);
        let e = _mm256_setr_ps(1., 3., 5., 7., 9., 11., 13., 15.);
        assert_eq_m256(r, e);
        // > If the values being compared are both 0.0s (of either sign), the
        // > value in the second operand (source operand) is returned.
        let w = _mm256_min_ps(_mm256_set1_ps(0.0), _mm256_set1_ps(-0.0));
        let x = _mm256_min_ps(_mm256_set1_ps(-0.0), _mm256_set1_ps(0.0));
        let wu: [u32; 8] = transmute(w);
        let xu: [u32; 8] = transmute(x);
        assert_eq!(wu, [0x8000_0000u32; 8]);
        assert_eq!(xu, [0u32; 8]);
        // > If only one value is a NaN (SNaN or QNaN) for this instruction, the
        // > second operand (source operand), either a NaN or a valid
        // > floating-point value, is written to the result.
        let y = _mm256_min_ps(_mm256_set1_ps(f32::NAN), _mm256_set1_ps(0.0));
        let z = _mm256_min_ps(_mm256_set1_ps(0.0), _mm256_set1_ps(f32::NAN));
        let yf: [f32; 8] = transmute(y);
        let zf: [f32; 8] = transmute(z);
        assert_eq!(yf, [0.0; 8]);
        assert!(zf.iter().all(|f| f.is_nan()), "{:?}", zf);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_mul_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_mul_pd(a, b);
        let e = _mm256_setr_pd(5., 12., 21., 32.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_mul_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm256_setr_ps(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm256_mul_ps(a, b);
        let e = _mm256_setr_ps(9., 20., 33., 48., 65., 84., 105., 128.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_addsub_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_addsub_pd(a, b);
        let e = _mm256_setr_pd(-4., 8., -4., 12.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_addsub_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 1., 2., 3., 4.);
        let b = _mm256_setr_ps(5., 6., 7., 8., 5., 6., 7., 8.);
        let r = _mm256_addsub_ps(a, b);
        let e = _mm256_setr_ps(-4., 8., -4., 12., -4., 8., -4., 12.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_sub_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_sub_pd(a, b);
        let e = _mm256_setr_pd(-4., -4., -4., -4.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_sub_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., -1., -2., -3., -4.);
        let b = _mm256_setr_ps(5., 6., 7., 8., 3., 2., 1., 0.);
        let r = _mm256_sub_ps(a, b);
        let e = _mm256_setr_ps(-4., -4., -4., -4., -4., -4., -4., -4.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_round_pd() {
        let a = _mm256_setr_pd(1.55, 2.2, 3.99, -1.2);
        let result_closest = _mm256_round_pd::<0b0000>(a);
        let result_down = _mm256_round_pd::<0b0001>(a);
        let result_up = _mm256_round_pd::<0b0010>(a);
        let expected_closest = _mm256_setr_pd(2., 2., 4., -1.);
        let expected_down = _mm256_setr_pd(1., 2., 3., -2.);
        let expected_up = _mm256_setr_pd(2., 3., 4., -1.);
        assert_eq_m256d(result_closest, expected_closest);
        assert_eq_m256d(result_down, expected_down);
        assert_eq_m256d(result_up, expected_up);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_floor_pd() {
        let a = _mm256_setr_pd(1.55, 2.2, 3.99, -1.2);
        let result_down = _mm256_floor_pd(a);
        let expected_down = _mm256_setr_pd(1., 2., 3., -2.);
        assert_eq_m256d(result_down, expected_down);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_ceil_pd() {
        let a = _mm256_setr_pd(1.55, 2.2, 3.99, -1.2);
        let result_up = _mm256_ceil_pd(a);
        let expected_up = _mm256_setr_pd(2., 3., 4., -1.);
        assert_eq_m256d(result_up, expected_up);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_round_ps() {
        let a = _mm256_setr_ps(1.55, 2.2, 3.99, -1.2, 1.55, 2.2, 3.99, -1.2);
        let result_closest = _mm256_round_ps::<0b0000>(a);
        let result_down = _mm256_round_ps::<0b0001>(a);
        let result_up = _mm256_round_ps::<0b0010>(a);
        let expected_closest = _mm256_setr_ps(2., 2., 4., -1., 2., 2., 4., -1.);
        let expected_down = _mm256_setr_ps(1., 2., 3., -2., 1., 2., 3., -2.);
        let expected_up = _mm256_setr_ps(2., 3., 4., -1., 2., 3., 4., -1.);
        assert_eq_m256(result_closest, expected_closest);
        assert_eq_m256(result_down, expected_down);
        assert_eq_m256(result_up, expected_up);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_floor_ps() {
        let a = _mm256_setr_ps(1.55, 2.2, 3.99, -1.2, 1.55, 2.2, 3.99, -1.2);
        let result_down = _mm256_floor_ps(a);
        let expected_down = _mm256_setr_ps(1., 2., 3., -2., 1., 2., 3., -2.);
        assert_eq_m256(result_down, expected_down);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_ceil_ps() {
        let a = _mm256_setr_ps(1.55, 2.2, 3.99, -1.2, 1.55, 2.2, 3.99, -1.2);
        let result_up = _mm256_ceil_ps(a);
        let expected_up = _mm256_setr_ps(2., 3., 4., -1., 2., 3., 4., -1.);
        assert_eq_m256(result_up, expected_up);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_sqrt_pd() {
        let a = _mm256_setr_pd(4., 9., 16., 25.);
        let r = _mm256_sqrt_pd(a);
        let e = _mm256_setr_pd(2., 3., 4., 5.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_sqrt_ps() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let r = _mm256_sqrt_ps(a);
        let e = _mm256_setr_ps(2., 3., 4., 5., 2., 3., 4., 5.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_div_ps() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = _mm256_div_ps(a, b);
        let e = _mm256_setr_ps(1., 3., 8., 5., 0.5, 1., 0.25, 0.5);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_div_pd() {
        let a = _mm256_setr_pd(4., 9., 16., 25.);
        let b = _mm256_setr_pd(4., 3., 2., 5.);
        let r = _mm256_div_pd(a, b);
        let e = _mm256_setr_pd(1., 3., 8., 5.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_blend_pd() {
        let a = _mm256_setr_pd(4., 9., 16., 25.);
        let b = _mm256_setr_pd(4., 3., 2., 5.);
        let r = _mm256_blend_pd::<0x0>(a, b);
        assert_eq_m256d(r, _mm256_setr_pd(4., 9., 16., 25.));
        let r = _mm256_blend_pd::<0x3>(a, b);
        assert_eq_m256d(r, _mm256_setr_pd(4., 3., 16., 25.));
        let r = _mm256_blend_pd::<0xF>(a, b);
        assert_eq_m256d(r, _mm256_setr_pd(4., 3., 2., 5.));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_blend_ps() {
        let a = _mm256_setr_ps(1., 4., 5., 8., 9., 12., 13., 16.);
        let b = _mm256_setr_ps(2., 3., 6., 7., 10., 11., 14., 15.);
        let r = _mm256_blend_ps::<0x0>(a, b);
        assert_eq_m256(r, _mm256_setr_ps(1., 4., 5., 8., 9., 12., 13., 16.));
        let r = _mm256_blend_ps::<0x3>(a, b);
        assert_eq_m256(r, _mm256_setr_ps(2., 3., 5., 8., 9., 12., 13., 16.));
        let r = _mm256_blend_ps::<0xF>(a, b);
        assert_eq_m256(r, _mm256_setr_ps(2., 3., 6., 7., 9., 12., 13., 16.));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_blendv_pd() {
        let a = _mm256_setr_pd(4., 9., 16., 25.);
        let b = _mm256_setr_pd(4., 3., 2., 5.);
        let c = _mm256_setr_pd(0., 0., !0 as f64, !0 as f64);
        let r = _mm256_blendv_pd(a, b, c);
        let e = _mm256_setr_pd(4., 9., 2., 5.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_blendv_ps() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        #[rustfmt::skip]
        let c = _mm256_setr_ps(
            0., 0., 0., 0., !0 as f32, !0 as f32, !0 as f32, !0 as f32,
        );
        let r = _mm256_blendv_ps(a, b, c);
        let e = _mm256_setr_ps(4., 9., 16., 25., 8., 9., 64., 50.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_dp_ps() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = _mm256_dp_ps::<0xFF>(a, b);
        let e = _mm256_setr_ps(200., 200., 200., 200., 2387., 2387., 2387., 2387.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_hadd_pd() {
        let a = _mm256_setr_pd(4., 9., 16., 25.);
        let b = _mm256_setr_pd(4., 3., 2., 5.);
        let r = _mm256_hadd_pd(a, b);
        let e = _mm256_setr_pd(13., 7., 41., 7.);
        assert_eq_m256d(r, e);

        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_hadd_pd(a, b);
        let e = _mm256_setr_pd(3., 11., 7., 15.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_hadd_ps() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = _mm256_hadd_ps(a, b);
        let e = _mm256_setr_ps(13., 41., 7., 7., 13., 41., 17., 114.);
        assert_eq_m256(r, e);

        let a = _mm256_setr_ps(1., 2., 3., 4., 1., 2., 3., 4.);
        let b = _mm256_setr_ps(5., 6., 7., 8., 5., 6., 7., 8.);
        let r = _mm256_hadd_ps(a, b);
        let e = _mm256_setr_ps(3., 7., 11., 15., 3., 7., 11., 15.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_hsub_pd() {
        let a = _mm256_setr_pd(4., 9., 16., 25.);
        let b = _mm256_setr_pd(4., 3., 2., 5.);
        let r = _mm256_hsub_pd(a, b);
        let e = _mm256_setr_pd(-5., 1., -9., -3.);
        assert_eq_m256d(r, e);

        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_hsub_pd(a, b);
        let e = _mm256_setr_pd(-1., -1., -1., -1.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_hsub_ps() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = _mm256_hsub_ps(a, b);
        let e = _mm256_setr_ps(-5., -9., 1., -3., -5., -9., -1., 14.);
        assert_eq_m256(r, e);

        let a = _mm256_setr_ps(1., 2., 3., 4., 1., 2., 3., 4.);
        let b = _mm256_setr_ps(5., 6., 7., 8., 5., 6., 7., 8.);
        let r = _mm256_hsub_ps(a, b);
        let e = _mm256_setr_ps(-1., -1., -1., -1., -1., -1., -1., -1.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_xor_pd() {
        let a = _mm256_setr_pd(4., 9., 16., 25.);
        let b = _mm256_set1_pd(0.);
        let r = _mm256_xor_pd(a, b);
        assert_eq_m256d(r, a);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_xor_ps() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = _mm256_set1_ps(0.);
        let r = _mm256_xor_ps(a, b);
        assert_eq_m256(r, a);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_cmp_pd() {
        let a = _mm_setr_pd(4., 9.);
        let b = _mm_setr_pd(4., 3.);
        let r = _mm_cmp_pd::<_CMP_GE_OS>(a, b);
        assert!(get_m128d(r, 0).is_nan());
        assert!(get_m128d(r, 1).is_nan());
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_cmp_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_cmp_pd::<_CMP_GE_OS>(a, b);
        let e = _mm256_set1_pd(0.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_cmp_ps() {
        let a = _mm_setr_ps(4., 3., 2., 5.);
        let b = _mm_setr_ps(4., 9., 16., 25.);
        let r = _mm_cmp_ps::<_CMP_GE_OS>(a, b);
        assert!(get_m128(r, 0).is_nan());
        assert_eq!(get_m128(r, 1), 0.);
        assert_eq!(get_m128(r, 2), 0.);
        assert_eq!(get_m128(r, 3), 0.);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_cmp_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 1., 2., 3., 4.);
        let b = _mm256_setr_ps(5., 6., 7., 8., 5., 6., 7., 8.);
        let r = _mm256_cmp_ps::<_CMP_GE_OS>(a, b);
        let e = _mm256_set1_ps(0.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_cmp_sd() {
        let a = _mm_setr_pd(4., 9.);
        let b = _mm_setr_pd(4., 3.);
        let r = _mm_cmp_sd::<_CMP_GE_OS>(a, b);
        assert!(get_m128d(r, 0).is_nan());
        assert_eq!(get_m128d(r, 1), 9.);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_cmp_ss() {
        let a = _mm_setr_ps(4., 3., 2., 5.);
        let b = _mm_setr_ps(4., 9., 16., 25.);
        let r = _mm_cmp_ss::<_CMP_GE_OS>(a, b);
        assert!(get_m128(r, 0).is_nan());
        assert_eq!(get_m128(r, 1), 3.);
        assert_eq!(get_m128(r, 2), 2.);
        assert_eq!(get_m128(r, 3), 5.);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_cvtepi32_pd() {
        let a = _mm_setr_epi32(4, 9, 16, 25);
        let r = _mm256_cvtepi32_pd(a);
        let e = _mm256_setr_pd(4., 9., 16., 25.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_cvtepi32_ps() {
        let a = _mm256_setr_epi32(4, 9, 16, 25, 4, 9, 16, 25);
        let r = _mm256_cvtepi32_ps(a);
        let e = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_cvtpd_ps() {
        let a = _mm256_setr_pd(4., 9., 16., 25.);
        let r = _mm256_cvtpd_ps(a);
        let e = _mm_setr_ps(4., 9., 16., 25.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_cvtps_epi32() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let r = _mm256_cvtps_epi32(a);
        let e = _mm256_setr_epi32(4, 9, 16, 25, 4, 9, 16, 25);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_cvtps_pd() {
        let a = _mm_setr_ps(4., 9., 16., 25.);
        let r = _mm256_cvtps_pd(a);
        let e = _mm256_setr_pd(4., 9., 16., 25.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_cvttpd_epi32() {
        let a = _mm256_setr_pd(4., 9., 16., 25.);
        let r = _mm256_cvttpd_epi32(a);
        let e = _mm_setr_epi32(4, 9, 16, 25);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_cvtpd_epi32() {
        let a = _mm256_setr_pd(4., 9., 16., 25.);
        let r = _mm256_cvtpd_epi32(a);
        let e = _mm_setr_epi32(4, 9, 16, 25);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_cvttps_epi32() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let r = _mm256_cvttps_epi32(a);
        let e = _mm256_setr_epi32(4, 9, 16, 25, 4, 9, 16, 25);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_extractf128_ps() {
        let a = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = _mm256_extractf128_ps::<0>(a);
        let e = _mm_setr_ps(4., 3., 2., 5.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_extractf128_pd() {
        let a = _mm256_setr_pd(4., 3., 2., 5.);
        let r = _mm256_extractf128_pd::<0>(a);
        let e = _mm_setr_pd(4., 3.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_extractf128_si256() {
        let a = _mm256_setr_epi64x(4, 3, 2, 5);
        let r = _mm256_extractf128_si256::<0>(a);
        let e = _mm_setr_epi64x(4, 3);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_zeroall() {
        _mm256_zeroall();
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_zeroupper() {
        _mm256_zeroupper();
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_permutevar_ps() {
        let a = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let b = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm256_permutevar_ps(a, b);
        let e = _mm256_setr_ps(3., 2., 5., 4., 9., 64., 50., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_permutevar_ps() {
        let a = _mm_setr_ps(4., 3., 2., 5.);
        let b = _mm_setr_epi32(1, 2, 3, 4);
        let r = _mm_permutevar_ps(a, b);
        let e = _mm_setr_ps(3., 2., 5., 4.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_permute_ps() {
        let a = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = _mm256_permute_ps::<0x1b>(a);
        let e = _mm256_setr_ps(5., 2., 3., 4., 50., 64., 9., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_permute_ps() {
        let a = _mm_setr_ps(4., 3., 2., 5.);
        let r = _mm_permute_ps::<0x1b>(a);
        let e = _mm_setr_ps(5., 2., 3., 4.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_permutevar_pd() {
        let a = _mm256_setr_pd(4., 3., 2., 5.);
        let b = _mm256_setr_epi64x(1, 2, 3, 4);
        let r = _mm256_permutevar_pd(a, b);
        let e = _mm256_setr_pd(4., 3., 5., 2.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_permutevar_pd() {
        let a = _mm_setr_pd(4., 3.);
        let b = _mm_setr_epi64x(3, 0);
        let r = _mm_permutevar_pd(a, b);
        let e = _mm_setr_pd(3., 4.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_permute_pd() {
        let a = _mm256_setr_pd(4., 3., 2., 5.);
        let r = _mm256_permute_pd::<5>(a);
        let e = _mm256_setr_pd(3., 4., 5., 2.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_permute_pd() {
        let a = _mm_setr_pd(4., 3.);
        let r = _mm_permute_pd::<1>(a);
        let e = _mm_setr_pd(3., 4.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_permute2f128_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 1., 2., 3., 4.);
        let b = _mm256_setr_ps(5., 6., 7., 8., 5., 6., 7., 8.);
        let r = _mm256_permute2f128_ps::<0x13>(a, b);
        let e = _mm256_setr_ps(5., 6., 7., 8., 1., 2., 3., 4.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_permute2f128_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_permute2f128_pd::<0x31>(a, b);
        let e = _mm256_setr_pd(3., 4., 7., 8.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_permute2f128_si256() {
        let a = _mm256_setr_epi32(1, 2, 3, 4, 1, 2, 3, 4);
        let b = _mm256_setr_epi32(5, 6, 7, 8, 5, 6, 7, 8);
        let r = _mm256_permute2f128_si256::<0x20>(a, b);
        let e = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_broadcast_ss() {
        let r = _mm256_broadcast_ss(&3.);
        let e = _mm256_set1_ps(3.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_broadcast_ss() {
        let r = _mm_broadcast_ss(&3.);
        let e = _mm_set1_ps(3.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_broadcast_sd() {
        let r = _mm256_broadcast_sd(&3.);
        let e = _mm256_set1_pd(3.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_broadcast_ps() {
        let a = _mm_setr_ps(4., 3., 2., 5.);
        let r = _mm256_broadcast_ps(&a);
        let e = _mm256_setr_ps(4., 3., 2., 5., 4., 3., 2., 5.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_broadcast_pd() {
        let a = _mm_setr_pd(4., 3.);
        let r = _mm256_broadcast_pd(&a);
        let e = _mm256_setr_pd(4., 3., 4., 3.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_insertf128_ps() {
        let a = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let b = _mm_setr_ps(4., 9., 16., 25.);
        let r = _mm256_insertf128_ps::<0>(a, b);
        let e = _mm256_setr_ps(4., 9., 16., 25., 8., 9., 64., 50.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_insertf128_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm_setr_pd(5., 6.);
        let r = _mm256_insertf128_pd::<0>(a, b);
        let e = _mm256_setr_pd(5., 6., 3., 4.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_insertf128_si256() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let b = _mm_setr_epi64x(5, 6);
        let r = _mm256_insertf128_si256::<0>(a, b);
        let e = _mm256_setr_epi64x(5, 6, 3, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_insert_epi8() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm256_insert_epi8::<31>(a, 0);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 0,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_insert_epi16() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi16(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        let r = _mm256_insert_epi16::<15>(a, 0);
        #[rustfmt::skip]
        let e = _mm256_setr_epi16(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 0,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_insert_epi32() {
        let a = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm256_insert_epi32::<7>(a, 0);
        let e = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_load_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let p = &a as *const _ as *const f64;
        let r = _mm256_load_pd(p);
        let e = _mm256_setr_pd(1., 2., 3., 4.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_store_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let mut r = _mm256_undefined_pd();
        _mm256_store_pd(&mut r as *mut _ as *mut f64, a);
        assert_eq_m256d(r, a);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_load_ps() {
        let a = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let p = &a as *const _ as *const f32;
        let r = _mm256_load_ps(p);
        let e = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_store_ps() {
        let a = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let mut r = _mm256_undefined_ps();
        _mm256_store_ps(&mut r as *mut _ as *mut f32, a);
        assert_eq_m256(r, a);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_loadu_pd() {
        let a = &[1.0f64, 2., 3., 4.];
        let p = a.as_ptr();
        let r = _mm256_loadu_pd(black_box(p));
        let e = _mm256_setr_pd(1., 2., 3., 4.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_storeu_pd() {
        let a = _mm256_set1_pd(9.);
        let mut r = _mm256_undefined_pd();
        _mm256_storeu_pd(&mut r as *mut _ as *mut f64, a);
        assert_eq_m256d(r, a);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_loadu_ps() {
        let a = &[4., 3., 2., 5., 8., 9., 64., 50.];
        let p = a.as_ptr();
        let r = _mm256_loadu_ps(black_box(p));
        let e = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_storeu_ps() {
        let a = _mm256_set1_ps(9.);
        let mut r = _mm256_undefined_ps();
        _mm256_storeu_ps(&mut r as *mut _ as *mut f32, a);
        assert_eq_m256(r, a);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_load_si256() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let p = &a as *const _;
        let r = _mm256_load_si256(p);
        let e = _mm256_setr_epi64x(1, 2, 3, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_store_si256() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let mut r = _mm256_undefined_si256();
        _mm256_store_si256(&mut r as *mut _, a);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_loadu_si256() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let p = &a as *const _;
        let r = _mm256_loadu_si256(black_box(p));
        let e = _mm256_setr_epi64x(1, 2, 3, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_storeu_si256() {
        let a = _mm256_set1_epi8(9);
        let mut r = _mm256_undefined_si256();
        _mm256_storeu_si256(&mut r as *mut _, a);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_maskload_pd() {
        let a = &[1.0f64, 2., 3., 4.];
        let p = a.as_ptr();
        let mask = _mm256_setr_epi64x(0, !0, 0, !0);
        let r = _mm256_maskload_pd(black_box(p), mask);
        let e = _mm256_setr_pd(0., 2., 0., 4.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_maskstore_pd() {
        let mut r = _mm256_set1_pd(0.);
        let mask = _mm256_setr_epi64x(0, !0, 0, !0);
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        _mm256_maskstore_pd(&mut r as *mut _ as *mut f64, mask, a);
        let e = _mm256_setr_pd(0., 2., 0., 4.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_maskload_pd() {
        let a = &[1.0f64, 2.];
        let p = a.as_ptr();
        let mask = _mm_setr_epi64x(0, !0);
        let r = _mm_maskload_pd(black_box(p), mask);
        let e = _mm_setr_pd(0., 2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_maskstore_pd() {
        let mut r = _mm_set1_pd(0.);
        let mask = _mm_setr_epi64x(0, !0);
        let a = _mm_setr_pd(1., 2.);
        _mm_maskstore_pd(&mut r as *mut _ as *mut f64, mask, a);
        let e = _mm_setr_pd(0., 2.);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_maskload_ps() {
        let a = &[1.0f32, 2., 3., 4., 5., 6., 7., 8.];
        let p = a.as_ptr();
        let mask = _mm256_setr_epi32(0, !0, 0, !0, 0, !0, 0, !0);
        let r = _mm256_maskload_ps(black_box(p), mask);
        let e = _mm256_setr_ps(0., 2., 0., 4., 0., 6., 0., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_maskstore_ps() {
        let mut r = _mm256_set1_ps(0.);
        let mask = _mm256_setr_epi32(0, !0, 0, !0, 0, !0, 0, !0);
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        _mm256_maskstore_ps(&mut r as *mut _ as *mut f32, mask, a);
        let e = _mm256_setr_ps(0., 2., 0., 4., 0., 6., 0., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_maskload_ps() {
        let a = &[1.0f32, 2., 3., 4.];
        let p = a.as_ptr();
        let mask = _mm_setr_epi32(0, !0, 0, !0);
        let r = _mm_maskload_ps(black_box(p), mask);
        let e = _mm_setr_ps(0., 2., 0., 4.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_maskstore_ps() {
        let mut r = _mm_set1_ps(0.);
        let mask = _mm_setr_epi32(0, !0, 0, !0);
        let a = _mm_setr_ps(1., 2., 3., 4.);
        _mm_maskstore_ps(&mut r as *mut _ as *mut f32, mask, a);
        let e = _mm_setr_ps(0., 2., 0., 4.);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_movehdup_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_movehdup_ps(a);
        let e = _mm256_setr_ps(2., 2., 4., 4., 6., 6., 8., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_moveldup_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_moveldup_ps(a);
        let e = _mm256_setr_ps(1., 1., 3., 3., 5., 5., 7., 7.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_movedup_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let r = _mm256_movedup_pd(a);
        let e = _mm256_setr_pd(1., 1., 3., 3.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_lddqu_si256() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let p = &a as *const _;
        let r = _mm256_lddqu_si256(black_box(p));
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_stream_si256() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let mut r = _mm256_undefined_si256();
        _mm256_stream_si256(&mut r as *mut _, a);
        assert_eq_m256i(r, a);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_stream_pd() {
        #[repr(align(32))]
        struct Memory {
            pub data: [f64; 4],
        }
        let a = _mm256_set1_pd(7.0);
        let mut mem = Memory { data: [-1.0; 4] };

        _mm256_stream_pd(&mut mem.data[0] as *mut f64, a);
        for i in 0..4 {
            assert_eq!(mem.data[i], get_m256d(a, i));
        }
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_stream_ps() {
        #[repr(align(32))]
        struct Memory {
            pub data: [f32; 8],
        }
        let a = _mm256_set1_ps(7.0);
        let mut mem = Memory { data: [-1.0; 8] };

        _mm256_stream_ps(&mut mem.data[0] as *mut f32, a);
        for i in 0..8 {
            assert_eq!(mem.data[i], get_m256(a, i));
        }
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_rcp_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_rcp_ps(a);
        #[rustfmt::skip]
        let e = _mm256_setr_ps(
            0.99975586, 0.49987793, 0.33325195, 0.24993896,
            0.19995117, 0.16662598, 0.14282227, 0.12496948,
        );
        let rel_err = 0.00048828125;
        for i in 0..8 {
            assert_approx_eq!(get_m256(r, i), get_m256(e, i), 2. * rel_err);
        }
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_rsqrt_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_rsqrt_ps(a);
        #[rustfmt::skip]
        let e = _mm256_setr_ps(
            0.99975586, 0.7069092, 0.5772705, 0.49987793,
            0.44714355, 0.40820313, 0.3779297, 0.3534546,
        );
        let rel_err = 0.00048828125;
        for i in 0..8 {
            assert_approx_eq!(get_m256(r, i), get_m256(e, i), 2. * rel_err);
        }
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_unpackhi_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_unpackhi_pd(a, b);
        let e = _mm256_setr_pd(2., 6., 4., 8.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_unpackhi_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm256_setr_ps(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm256_unpackhi_ps(a, b);
        let e = _mm256_setr_ps(3., 11., 4., 12., 7., 15., 8., 16.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_unpacklo_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_unpacklo_pd(a, b);
        let e = _mm256_setr_pd(1., 5., 3., 7.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_unpacklo_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm256_setr_ps(9., 10., 11., 12., 13., 14., 15., 16.);
        let r = _mm256_unpacklo_ps(a, b);
        let e = _mm256_setr_ps(1., 9., 2., 10., 5., 13., 6., 14.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_testz_si256() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let b = _mm256_setr_epi64x(5, 6, 7, 8);
        let r = _mm256_testz_si256(a, b);
        assert_eq!(r, 0);
        let b = _mm256_set1_epi64x(0);
        let r = _mm256_testz_si256(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_testc_si256() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let b = _mm256_setr_epi64x(5, 6, 7, 8);
        let r = _mm256_testc_si256(a, b);
        assert_eq!(r, 0);
        let b = _mm256_set1_epi64x(0);
        let r = _mm256_testc_si256(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_testnzc_si256() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let b = _mm256_setr_epi64x(5, 6, 7, 8);
        let r = _mm256_testnzc_si256(a, b);
        assert_eq!(r, 1);
        let a = _mm256_setr_epi64x(0, 0, 0, 0);
        let b = _mm256_setr_epi64x(0, 0, 0, 0);
        let r = _mm256_testnzc_si256(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_testz_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_testz_pd(a, b);
        assert_eq!(r, 1);
        let a = _mm256_set1_pd(-1.);
        let r = _mm256_testz_pd(a, a);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_testc_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_testc_pd(a, b);
        assert_eq!(r, 1);
        let a = _mm256_set1_pd(1.);
        let b = _mm256_set1_pd(-1.);
        let r = _mm256_testc_pd(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_testnzc_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_testnzc_pd(a, b);
        assert_eq!(r, 0);
        let a = _mm256_setr_pd(1., -1., -1., -1.);
        let b = _mm256_setr_pd(-1., -1., 1., 1.);
        let r = _mm256_testnzc_pd(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_testz_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 6.);
        let r = _mm_testz_pd(a, b);
        assert_eq!(r, 1);
        let a = _mm_set1_pd(-1.);
        let r = _mm_testz_pd(a, a);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_testc_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 6.);
        let r = _mm_testc_pd(a, b);
        assert_eq!(r, 1);
        let a = _mm_set1_pd(1.);
        let b = _mm_set1_pd(-1.);
        let r = _mm_testc_pd(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_testnzc_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 6.);
        let r = _mm_testnzc_pd(a, b);
        assert_eq!(r, 0);
        let a = _mm_setr_pd(1., -1.);
        let b = _mm_setr_pd(-1., -1.);
        let r = _mm_testnzc_pd(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_testz_ps() {
        let a = _mm256_set1_ps(1.);
        let r = _mm256_testz_ps(a, a);
        assert_eq!(r, 1);
        let a = _mm256_set1_ps(-1.);
        let r = _mm256_testz_ps(a, a);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_testc_ps() {
        let a = _mm256_set1_ps(1.);
        let r = _mm256_testc_ps(a, a);
        assert_eq!(r, 1);
        let b = _mm256_set1_ps(-1.);
        let r = _mm256_testc_ps(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_testnzc_ps() {
        let a = _mm256_set1_ps(1.);
        let r = _mm256_testnzc_ps(a, a);
        assert_eq!(r, 0);
        let a = _mm256_setr_ps(1., -1., -1., -1., -1., -1., -1., -1.);
        let b = _mm256_setr_ps(-1., -1., 1., 1., 1., 1., 1., 1.);
        let r = _mm256_testnzc_ps(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_testz_ps() {
        let a = _mm_set1_ps(1.);
        let r = _mm_testz_ps(a, a);
        assert_eq!(r, 1);
        let a = _mm_set1_ps(-1.);
        let r = _mm_testz_ps(a, a);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_testc_ps() {
        let a = _mm_set1_ps(1.);
        let r = _mm_testc_ps(a, a);
        assert_eq!(r, 1);
        let b = _mm_set1_ps(-1.);
        let r = _mm_testc_ps(a, b);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm_testnzc_ps() {
        let a = _mm_set1_ps(1.);
        let r = _mm_testnzc_ps(a, a);
        assert_eq!(r, 0);
        let a = _mm_setr_ps(1., -1., -1., -1.);
        let b = _mm_setr_ps(-1., -1., 1., 1.);
        let r = _mm_testnzc_ps(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_movemask_pd() {
        let a = _mm256_setr_pd(1., -2., 3., -4.);
        let r = _mm256_movemask_pd(a);
        assert_eq!(r, 0xA);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_movemask_ps() {
        let a = _mm256_setr_ps(1., -2., 3., -4., 1., -2., 3., -4.);
        let r = _mm256_movemask_ps(a);
        assert_eq!(r, 0xAA);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_setzero_pd() {
        let r = _mm256_setzero_pd();
        assert_eq_m256d(r, _mm256_set1_pd(0.));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_setzero_ps() {
        let r = _mm256_setzero_ps();
        assert_eq_m256(r, _mm256_set1_ps(0.));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_setzero_si256() {
        let r = _mm256_setzero_si256();
        assert_eq_m256i(r, _mm256_set1_epi8(0));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set_pd() {
        let r = _mm256_set_pd(1., 2., 3., 4.);
        assert_eq_m256d(r, _mm256_setr_pd(4., 3., 2., 1.));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set_ps() {
        let r = _mm256_set_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m256(r, _mm256_setr_ps(8., 7., 6., 5., 4., 3., 2., 1.));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set_epi8() {
        #[rustfmt::skip]
        let r = _mm256_set_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            32, 31, 30, 29, 28, 27, 26, 25,
            24, 23, 22, 21, 20, 19, 18, 17,
            16, 15, 14, 13, 12, 11, 10, 9,
            8, 7, 6, 5, 4, 3, 2, 1
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set_epi16() {
        #[rustfmt::skip]
        let r = _mm256_set_epi16(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        );
        #[rustfmt::skip]
        let e = _mm256_setr_epi16(
            16, 15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set_epi32() {
        let r = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m256i(r, _mm256_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set_epi64x() {
        let r = _mm256_set_epi64x(1, 2, 3, 4);
        assert_eq_m256i(r, _mm256_setr_epi64x(4, 3, 2, 1));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_setr_pd() {
        let r = _mm256_setr_pd(1., 2., 3., 4.);
        assert_eq_m256d(r, _mm256_setr_pd(1., 2., 3., 4.));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_setr_ps() {
        let r = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m256(r, _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_setr_epi8() {
        #[rustfmt::skip]
        let r = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32
        );

        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_setr_epi16() {
        #[rustfmt::skip]
        let r = _mm256_setr_epi16(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        );
        #[rustfmt::skip]
        let e = _mm256_setr_epi16(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_setr_epi32() {
        let r = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m256i(r, _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_setr_epi64x() {
        let r = _mm256_setr_epi64x(1, 2, 3, 4);
        assert_eq_m256i(r, _mm256_setr_epi64x(1, 2, 3, 4));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set1_pd() {
        let r = _mm256_set1_pd(1.);
        assert_eq_m256d(r, _mm256_set1_pd(1.));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set1_ps() {
        let r = _mm256_set1_ps(1.);
        assert_eq_m256(r, _mm256_set1_ps(1.));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set1_epi8() {
        let r = _mm256_set1_epi8(1);
        assert_eq_m256i(r, _mm256_set1_epi8(1));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set1_epi16() {
        let r = _mm256_set1_epi16(1);
        assert_eq_m256i(r, _mm256_set1_epi16(1));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set1_epi32() {
        let r = _mm256_set1_epi32(1);
        assert_eq_m256i(r, _mm256_set1_epi32(1));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set1_epi64x() {
        let r = _mm256_set1_epi64x(1);
        assert_eq_m256i(r, _mm256_set1_epi64x(1));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_castpd_ps() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let r = _mm256_castpd_ps(a);
        let e = _mm256_setr_ps(0., 1.875, 0., 2., 0., 2.125, 0., 2.25);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_castps_pd() {
        let a = _mm256_setr_ps(0., 1.875, 0., 2., 0., 2.125, 0., 2.25);
        let r = _mm256_castps_pd(a);
        let e = _mm256_setr_pd(1., 2., 3., 4.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_castps_si256() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_castps_si256(a);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            0, 0, -128, 63, 0, 0, 0, 64,
            0, 0, 64, 64, 0, 0, -128, 64,
            0, 0, -96, 64, 0, 0, -64, 64,
            0, 0, -32, 64, 0, 0, 0, 65,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_castsi256_ps() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            0, 0, -128, 63, 0, 0, 0, 64,
            0, 0, 64, 64, 0, 0, -128, 64,
            0, 0, -96, 64, 0, 0, -64, 64,
            0, 0, -32, 64, 0, 0, 0, 65,
        );
        let r = _mm256_castsi256_ps(a);
        let e = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_castpd_si256() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let r = _mm256_castpd_si256(a);
        assert_eq_m256d(transmute(r), a);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_castsi256_pd() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let r = _mm256_castsi256_pd(a);
        assert_eq_m256d(r, transmute(a));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_castps256_ps128() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_castps256_ps128(a);
        assert_eq_m128(r, _mm_setr_ps(1., 2., 3., 4.));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_castpd256_pd128() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let r = _mm256_castpd256_pd128(a);
        assert_eq_m128d(r, _mm_setr_pd(1., 2.));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_castsi256_si128() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let r = _mm256_castsi256_si128(a);
        assert_eq_m128i(r, _mm_setr_epi64x(1, 2));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_zextps128_ps256() {
        let a = _mm_setr_ps(1., 2., 3., 4.);
        let r = _mm256_zextps128_ps256(a);
        let e = _mm256_setr_ps(1., 2., 3., 4., 0., 0., 0., 0.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_zextsi128_si256() {
        let a = _mm_setr_epi64x(1, 2);
        let r = _mm256_zextsi128_si256(a);
        let e = _mm256_setr_epi64x(1, 2, 0, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_zextpd128_pd256() {
        let a = _mm_setr_pd(1., 2.);
        let r = _mm256_zextpd128_pd256(a);
        let e = _mm256_setr_pd(1., 2., 0., 0.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set_m128() {
        let hi = _mm_setr_ps(5., 6., 7., 8.);
        let lo = _mm_setr_ps(1., 2., 3., 4.);
        let r = _mm256_set_m128(hi, lo);
        let e = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set_m128d() {
        let hi = _mm_setr_pd(3., 4.);
        let lo = _mm_setr_pd(1., 2.);
        let r = _mm256_set_m128d(hi, lo);
        let e = _mm256_setr_pd(1., 2., 3., 4.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_set_m128i() {
        #[rustfmt::skip]
        let hi = _mm_setr_epi8(
            17, 18, 19, 20,
            21, 22, 23, 24,
            25, 26, 27, 28,
            29, 30, 31, 32,
        );
        #[rustfmt::skip]
        let lo = _mm_setr_epi8(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
        );
        let r = _mm256_set_m128i(hi, lo);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_setr_m128() {
        let lo = _mm_setr_ps(1., 2., 3., 4.);
        let hi = _mm_setr_ps(5., 6., 7., 8.);
        let r = _mm256_setr_m128(lo, hi);
        let e = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_setr_m128d() {
        let lo = _mm_setr_pd(1., 2.);
        let hi = _mm_setr_pd(3., 4.);
        let r = _mm256_setr_m128d(lo, hi);
        let e = _mm256_setr_pd(1., 2., 3., 4.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_setr_m128i() {
        #[rustfmt::skip]
        let lo = _mm_setr_epi8(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
        );
        #[rustfmt::skip]
        let hi = _mm_setr_epi8(
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm256_setr_m128i(lo, hi);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_loadu2_m128() {
        let hi = &[5., 6., 7., 8.];
        let hiaddr = hi.as_ptr();
        let lo = &[1., 2., 3., 4.];
        let loaddr = lo.as_ptr();
        let r = _mm256_loadu2_m128(hiaddr, loaddr);
        let e = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_loadu2_m128d() {
        let hi = &[3., 4.];
        let hiaddr = hi.as_ptr();
        let lo = &[1., 2.];
        let loaddr = lo.as_ptr();
        let r = _mm256_loadu2_m128d(hiaddr, loaddr);
        let e = _mm256_setr_pd(1., 2., 3., 4.);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_loadu2_m128i() {
        #[rustfmt::skip]
        let hi = _mm_setr_epi8(
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        #[rustfmt::skip]
        let lo = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        );
        let r = _mm256_loadu2_m128i(&hi as *const _ as *const _, &lo as *const _ as *const _);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_storeu2_m128() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let mut hi = _mm_undefined_ps();
        let mut lo = _mm_undefined_ps();
        _mm256_storeu2_m128(
            &mut hi as *mut _ as *mut f32,
            &mut lo as *mut _ as *mut f32,
            a,
        );
        assert_eq_m128(hi, _mm_setr_ps(5., 6., 7., 8.));
        assert_eq_m128(lo, _mm_setr_ps(1., 2., 3., 4.));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_storeu2_m128d() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let mut hi = _mm_undefined_pd();
        let mut lo = _mm_undefined_pd();
        _mm256_storeu2_m128d(
            &mut hi as *mut _ as *mut f64,
            &mut lo as *mut _ as *mut f64,
            a,
        );
        assert_eq_m128d(hi, _mm_setr_pd(3., 4.));
        assert_eq_m128d(lo, _mm_setr_pd(1., 2.));
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_storeu2_m128i() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let mut hi = _mm_undefined_si128();
        let mut lo = _mm_undefined_si128();
        _mm256_storeu2_m128i(&mut hi as *mut _, &mut lo as *mut _, a);
        #[rustfmt::skip]
        let e_hi = _mm_setr_epi8(
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32
        );
        #[rustfmt::skip]
        let e_lo = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16
        );

        assert_eq_m128i(hi, e_hi);
        assert_eq_m128i(lo, e_lo);
    }

    #[simd_test(enable = "avx")]
    unsafe fn test_mm256_cvtss_f32() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_cvtss_f32(a);
        assert_eq!(r, 1.);
    }
}
