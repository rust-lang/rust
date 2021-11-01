//! Streaming SIMD Extensions 4.1 (SSE4.1)

use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem::transmute,
};

#[cfg(test)]
use stdarch_test::assert_instr;

// SSE4 rounding constans
/// round to nearest
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FROUND_TO_NEAREST_INT: i32 = 0x00;
/// round down
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FROUND_TO_NEG_INF: i32 = 0x01;
/// round up
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FROUND_TO_POS_INF: i32 = 0x02;
/// truncate
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FROUND_TO_ZERO: i32 = 0x03;
/// use MXCSR.RC; see `vendor::_MM_SET_ROUNDING_MODE`
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FROUND_CUR_DIRECTION: i32 = 0x04;
/// do not suppress exceptions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FROUND_RAISE_EXC: i32 = 0x00;
/// suppress exceptions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FROUND_NO_EXC: i32 = 0x08;
/// round to nearest and do not suppress exceptions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FROUND_NINT: i32 = 0x00;
/// round down and do not suppress exceptions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FROUND_FLOOR: i32 = _MM_FROUND_RAISE_EXC | _MM_FROUND_TO_NEG_INF;
/// round up and do not suppress exceptions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FROUND_CEIL: i32 = _MM_FROUND_RAISE_EXC | _MM_FROUND_TO_POS_INF;
/// truncate and do not suppress exceptions
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FROUND_TRUNC: i32 = _MM_FROUND_RAISE_EXC | _MM_FROUND_TO_ZERO;
/// use MXCSR.RC and do not suppress exceptions; see
/// `vendor::_MM_SET_ROUNDING_MODE`
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FROUND_RINT: i32 = _MM_FROUND_RAISE_EXC | _MM_FROUND_CUR_DIRECTION;
/// use MXCSR.RC and suppress exceptions; see `vendor::_MM_SET_ROUNDING_MODE`
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _MM_FROUND_NEARBYINT: i32 = _MM_FROUND_NO_EXC | _MM_FROUND_CUR_DIRECTION;

/// Blend packed 8-bit integers from `a` and `b` using `mask`
///
/// The high bit of each corresponding mask byte determines the selection.
/// If the high bit is set the element of `a` is selected. The element
/// of `b` is selected otherwise.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_blendv_epi8)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pblendvb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_blendv_epi8(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
    transmute(pblendvb(a.as_i8x16(), b.as_i8x16(), mask.as_i8x16()))
}

/// Blend packed 16-bit integers from `a` and `b` using the mask `IMM8`.
///
/// The mask bits determine the selection. A clear bit selects the
/// corresponding element of `a`, and a set bit the corresponding
/// element of `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_blend_epi16)
#[inline]
#[target_feature(enable = "sse4.1")]
// Note: LLVM7 prefers the single-precision floating-point domain when possible
// see https://bugs.llvm.org/show_bug.cgi?id=38195
// #[cfg_attr(test, assert_instr(pblendw, IMM8 = 0xF0))]
#[cfg_attr(test, assert_instr(blendps, IMM8 = 0xF0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_blend_epi16<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    static_assert_imm8!(IMM8);
    transmute(pblendw(a.as_i16x8(), b.as_i16x8(), IMM8 as u8))
}

/// Blend packed double-precision (64-bit) floating-point elements from `a`
/// and `b` using `mask`
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_blendv_pd)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(blendvpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_blendv_pd(a: __m128d, b: __m128d, mask: __m128d) -> __m128d {
    blendvpd(a, b, mask)
}

/// Blend packed single-precision (32-bit) floating-point elements from `a`
/// and `b` using `mask`
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_blendv_ps)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(blendvps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_blendv_ps(a: __m128, b: __m128, mask: __m128) -> __m128 {
    blendvps(a, b, mask)
}

/// Blend packed double-precision (64-bit) floating-point elements from `a`
/// and `b` using control mask `IMM2`
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_blend_pd)
#[inline]
#[target_feature(enable = "sse4.1")]
// Note: LLVM7 prefers the single-precision floating-point domain when possible
// see https://bugs.llvm.org/show_bug.cgi?id=38195
// #[cfg_attr(test, assert_instr(blendpd, IMM2 = 0b10))]
#[cfg_attr(test, assert_instr(blendps, IMM2 = 0b10))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_blend_pd<const IMM2: i32>(a: __m128d, b: __m128d) -> __m128d {
    static_assert_imm2!(IMM2);
    blendpd(a, b, IMM2 as u8)
}

/// Blend packed single-precision (32-bit) floating-point elements from `a`
/// and `b` using mask `IMM4`
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_blend_ps)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(blendps, IMM4 = 0b0101))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_blend_ps<const IMM4: i32>(a: __m128, b: __m128) -> __m128 {
    static_assert_imm4!(IMM4);
    blendps(a, b, IMM4 as u8)
}

/// Extracts a single-precision (32-bit) floating-point element from `a`,
/// selected with `IMM8`. The returned `i32` stores the float's bit-pattern,
/// and may be converted back to a floating point number via casting.
///
/// # Example
/// ```rust
/// # #[cfg(target_arch = "x86")]
/// # use std::arch::x86::*;
/// # #[cfg(target_arch = "x86_64")]
/// # use std::arch::x86_64::*;
/// # fn main() {
/// #    if is_x86_feature_detected!("sse4.1") {
/// #       #[target_feature(enable = "sse4.1")]
/// #       unsafe fn worker() {
/// let mut float_store = vec![1.0, 1.0, 2.0, 3.0];
/// unsafe {
///     let simd_floats = _mm_set_ps(2.5, 5.0, 7.5, 10.0);
///     let x: i32 = _mm_extract_ps::<2>(simd_floats);
///     float_store.push(f32::from_bits(x as u32));
/// }
/// assert_eq!(float_store, vec![1.0, 1.0, 2.0, 3.0, 5.0]);
/// #       }
/// #       unsafe { worker() }
/// #   }
/// # }
/// ```
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_extract_ps)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(
    all(test, not(target_os = "windows")),
    assert_instr(extractps, IMM8 = 0)
)]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_extract_ps<const IMM8: i32>(a: __m128) -> i32 {
    static_assert_imm2!(IMM8);
    transmute(simd_extract::<_, f32>(a, IMM8 as u32))
}

/// Extracts an 8-bit integer from `a`, selected with `IMM8`. Returns a 32-bit
/// integer containing the zero-extended integer data.
///
/// See [LLVM commit D20468](https://reviews.llvm.org/D20468).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_extract_epi8)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pextrb, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_extract_epi8<const IMM8: i32>(a: __m128i) -> i32 {
    static_assert_imm4!(IMM8);
    simd_extract::<_, u8>(a.as_u8x16(), IMM8 as u32) as i32
}

/// Extracts an 32-bit integer from `a` selected with `IMM8`
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_extract_epi32)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(
    all(test, not(target_os = "windows")),
    assert_instr(extractps, IMM8 = 1)
)]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_extract_epi32<const IMM8: i32>(a: __m128i) -> i32 {
    static_assert_imm2!(IMM8);
    simd_extract::<_, i32>(a.as_i32x4(), IMM8 as u32)
}

/// Select a single value in `a` to store at some position in `b`,
/// Then zero elements according to `IMM8`.
///
/// `IMM8` specifies which bits from operand `a` will be copied, which bits in
/// the result they will be copied to, and which bits in the result will be
/// cleared. The following assignments are made:
///
/// * Bits `[7:6]` specify the bits to copy from operand `a`:
///     - `00`: Selects bits `[31:0]` from operand `a`.
///     - `01`: Selects bits `[63:32]` from operand `a`.
///     - `10`: Selects bits `[95:64]` from operand `a`.
///     - `11`: Selects bits `[127:96]` from operand `a`.
///
/// * Bits `[5:4]` specify the bits in the result to which the selected bits
/// from operand `a` are copied:
///     - `00`: Copies the selected bits from `a` to result bits `[31:0]`.
///     - `01`: Copies the selected bits from `a` to result bits `[63:32]`.
///     - `10`: Copies the selected bits from `a` to result bits `[95:64]`.
///     - `11`: Copies the selected bits from `a` to result bits `[127:96]`.
///
/// * Bits `[3:0]`: If any of these bits are set, the corresponding result
/// element is cleared.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_insert_ps)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(insertps, IMM8 = 0b1010))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_insert_ps<const IMM8: i32>(a: __m128, b: __m128) -> __m128 {
    static_assert_imm8!(IMM8);
    insertps(a, b, IMM8 as u8)
}

/// Returns a copy of `a` with the 8-bit integer from `i` inserted at a
/// location specified by `IMM8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_insert_epi8)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pinsrb, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_insert_epi8<const IMM8: i32>(a: __m128i, i: i32) -> __m128i {
    static_assert_imm4!(IMM8);
    transmute(simd_insert(a.as_i8x16(), IMM8 as u32, i as i8))
}

/// Returns a copy of `a` with the 32-bit integer from `i` inserted at a
/// location specified by `IMM8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_insert_epi32)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pinsrd, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_insert_epi32<const IMM8: i32>(a: __m128i, i: i32) -> __m128i {
    static_assert_imm2!(IMM8);
    transmute(simd_insert(a.as_i32x4(), IMM8 as u32, i))
}

/// Compares packed 8-bit integers in `a` and `b` and returns packed maximum
/// values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_max_epi8)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmaxsb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_max_epi8(a: __m128i, b: __m128i) -> __m128i {
    transmute(pmaxsb(a.as_i8x16(), b.as_i8x16()))
}

/// Compares packed unsigned 16-bit integers in `a` and `b`, and returns packed
/// maximum.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_max_epu16)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmaxuw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_max_epu16(a: __m128i, b: __m128i) -> __m128i {
    transmute(pmaxuw(a.as_u16x8(), b.as_u16x8()))
}

/// Compares packed 32-bit integers in `a` and `b`, and returns packed maximum
/// values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_max_epi32)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmaxsd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_max_epi32(a: __m128i, b: __m128i) -> __m128i {
    transmute(pmaxsd(a.as_i32x4(), b.as_i32x4()))
}

/// Compares packed unsigned 32-bit integers in `a` and `b`, and returns packed
/// maximum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_max_epu32)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmaxud))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_max_epu32(a: __m128i, b: __m128i) -> __m128i {
    transmute(pmaxud(a.as_u32x4(), b.as_u32x4()))
}

/// Compares packed 8-bit integers in `a` and `b` and returns packed minimum
/// values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_min_epi8)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pminsb))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_min_epi8(a: __m128i, b: __m128i) -> __m128i {
    transmute(pminsb(a.as_i8x16(), b.as_i8x16()))
}

/// Compares packed unsigned 16-bit integers in `a` and `b`, and returns packed
/// minimum.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_min_epu16)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pminuw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_min_epu16(a: __m128i, b: __m128i) -> __m128i {
    transmute(pminuw(a.as_u16x8(), b.as_u16x8()))
}

/// Compares packed 32-bit integers in `a` and `b`, and returns packed minimum
/// values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_min_epi32)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pminsd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_min_epi32(a: __m128i, b: __m128i) -> __m128i {
    transmute(pminsd(a.as_i32x4(), b.as_i32x4()))
}

/// Compares packed unsigned 32-bit integers in `a` and `b`, and returns packed
/// minimum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_min_epu32)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pminud))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_min_epu32(a: __m128i, b: __m128i) -> __m128i {
    transmute(pminud(a.as_u32x4(), b.as_u32x4()))
}

/// Converts packed 32-bit integers from `a` and `b` to packed 16-bit integers
/// using unsigned saturation
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_packus_epi32)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(packusdw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_packus_epi32(a: __m128i, b: __m128i) -> __m128i {
    transmute(packusdw(a.as_i32x4(), b.as_i32x4()))
}

/// Compares packed 64-bit integers in `a` and `b` for equality
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cmpeq_epi64)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pcmpeqq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cmpeq_epi64(a: __m128i, b: __m128i) -> __m128i {
    transmute(simd_eq::<_, i64x2>(a.as_i64x2(), b.as_i64x2()))
}

/// Sign extend packed 8-bit integers in `a` to packed 16-bit integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtepi8_epi16)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovsxbw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cvtepi8_epi16(a: __m128i) -> __m128i {
    let a = a.as_i8x16();
    let a: i8x8 = simd_shuffle8!(a, a, [0, 1, 2, 3, 4, 5, 6, 7]);
    transmute(simd_cast::<_, i16x8>(a))
}

/// Sign extend packed 8-bit integers in `a` to packed 32-bit integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtepi8_epi32)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovsxbd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cvtepi8_epi32(a: __m128i) -> __m128i {
    let a = a.as_i8x16();
    let a: i8x4 = simd_shuffle4!(a, a, [0, 1, 2, 3]);
    transmute(simd_cast::<_, i32x4>(a))
}

/// Sign extend packed 8-bit integers in the low 8 bytes of `a` to packed
/// 64-bit integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtepi8_epi64)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovsxbq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cvtepi8_epi64(a: __m128i) -> __m128i {
    let a = a.as_i8x16();
    let a: i8x2 = simd_shuffle2!(a, a, [0, 1]);
    transmute(simd_cast::<_, i64x2>(a))
}

/// Sign extend packed 16-bit integers in `a` to packed 32-bit integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtepi16_epi32)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovsxwd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cvtepi16_epi32(a: __m128i) -> __m128i {
    let a = a.as_i16x8();
    let a: i16x4 = simd_shuffle4!(a, a, [0, 1, 2, 3]);
    transmute(simd_cast::<_, i32x4>(a))
}

/// Sign extend packed 16-bit integers in `a` to packed 64-bit integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtepi16_epi64)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovsxwq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cvtepi16_epi64(a: __m128i) -> __m128i {
    let a = a.as_i16x8();
    let a: i16x2 = simd_shuffle2!(a, a, [0, 1]);
    transmute(simd_cast::<_, i64x2>(a))
}

/// Sign extend packed 32-bit integers in `a` to packed 64-bit integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtepi32_epi64)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovsxdq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cvtepi32_epi64(a: __m128i) -> __m128i {
    let a = a.as_i32x4();
    let a: i32x2 = simd_shuffle2!(a, a, [0, 1]);
    transmute(simd_cast::<_, i64x2>(a))
}

/// Zeroes extend packed unsigned 8-bit integers in `a` to packed 16-bit integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtepu8_epi16)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovzxbw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cvtepu8_epi16(a: __m128i) -> __m128i {
    let a = a.as_u8x16();
    let a: u8x8 = simd_shuffle8!(a, a, [0, 1, 2, 3, 4, 5, 6, 7]);
    transmute(simd_cast::<_, i16x8>(a))
}

/// Zeroes extend packed unsigned 8-bit integers in `a` to packed 32-bit integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtepu8_epi32)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovzxbd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cvtepu8_epi32(a: __m128i) -> __m128i {
    let a = a.as_u8x16();
    let a: u8x4 = simd_shuffle4!(a, a, [0, 1, 2, 3]);
    transmute(simd_cast::<_, i32x4>(a))
}

/// Zeroes extend packed unsigned 8-bit integers in `a` to packed 64-bit integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtepu8_epi64)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovzxbq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cvtepu8_epi64(a: __m128i) -> __m128i {
    let a = a.as_u8x16();
    let a: u8x2 = simd_shuffle2!(a, a, [0, 1]);
    transmute(simd_cast::<_, i64x2>(a))
}

/// Zeroes extend packed unsigned 16-bit integers in `a`
/// to packed 32-bit integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtepu16_epi32)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovzxwd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cvtepu16_epi32(a: __m128i) -> __m128i {
    let a = a.as_u16x8();
    let a: u16x4 = simd_shuffle4!(a, a, [0, 1, 2, 3]);
    transmute(simd_cast::<_, i32x4>(a))
}

/// Zeroes extend packed unsigned 16-bit integers in `a`
/// to packed 64-bit integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtepu16_epi64)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovzxwq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cvtepu16_epi64(a: __m128i) -> __m128i {
    let a = a.as_u16x8();
    let a: u16x2 = simd_shuffle2!(a, a, [0, 1]);
    transmute(simd_cast::<_, i64x2>(a))
}

/// Zeroes extend packed unsigned 32-bit integers in `a`
/// to packed 64-bit integers
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtepu32_epi64)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovzxdq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_cvtepu32_epi64(a: __m128i) -> __m128i {
    let a = a.as_u32x4();
    let a: u32x2 = simd_shuffle2!(a, a, [0, 1]);
    transmute(simd_cast::<_, i64x2>(a))
}

/// Returns the dot product of two __m128d vectors.
///
/// `IMM8[1:0]` is the broadcast mask, and `IMM8[5:4]` is the condition mask.
/// If a condition mask bit is zero, the corresponding multiplication is
/// replaced by a value of `0.0`. If a broadcast mask bit is one, the result of
/// the dot product will be stored in the return value component. Otherwise if
/// the broadcast mask bit is zero then the return component will be zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_dp_pd)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(dppd, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_dp_pd<const IMM8: i32>(a: __m128d, b: __m128d) -> __m128d {
    static_assert_imm8!(IMM8);
    dppd(a, b, IMM8 as u8)
}

/// Returns the dot product of two __m128 vectors.
///
/// `IMM8[3:0]` is the broadcast mask, and `IMM8[7:4]` is the condition mask.
/// If a condition mask bit is zero, the corresponding multiplication is
/// replaced by a value of `0.0`. If a broadcast mask bit is one, the result of
/// the dot product will be stored in the return value component. Otherwise if
/// the broadcast mask bit is zero then the return component will be zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_dp_ps)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(dpps, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_dp_ps<const IMM8: i32>(a: __m128, b: __m128) -> __m128 {
    static_assert_imm8!(IMM8);
    dpps(a, b, IMM8 as u8)
}

/// Round the packed double-precision (64-bit) floating-point elements in `a`
/// down to an integer value, and stores the results as packed double-precision
/// floating-point elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_floor_pd)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_floor_pd(a: __m128d) -> __m128d {
    simd_floor(a)
}

/// Round the packed single-precision (32-bit) floating-point elements in `a`
/// down to an integer value, and stores the results as packed single-precision
/// floating-point elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_floor_ps)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_floor_ps(a: __m128) -> __m128 {
    simd_floor(a)
}

/// Round the lower double-precision (64-bit) floating-point element in `b`
/// down to an integer value, store the result as a double-precision
/// floating-point element in the lower element of the intrinsic result,
/// and copies the upper element from `a` to the upper element of the intrinsic
/// result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_floor_sd)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundsd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_floor_sd(a: __m128d, b: __m128d) -> __m128d {
    roundsd(a, b, _MM_FROUND_FLOOR)
}

/// Round the lower single-precision (32-bit) floating-point element in `b`
/// down to an integer value, store the result as a single-precision
/// floating-point element in the lower element of the intrinsic result,
/// and copies the upper 3 packed elements from `a` to the upper elements
/// of the intrinsic result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_floor_ss)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_floor_ss(a: __m128, b: __m128) -> __m128 {
    roundss(a, b, _MM_FROUND_FLOOR)
}

/// Round the packed double-precision (64-bit) floating-point elements in `a`
/// up to an integer value, and stores the results as packed double-precision
/// floating-point elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_ceil_pd)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundpd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_ceil_pd(a: __m128d) -> __m128d {
    simd_ceil(a)
}

/// Round the packed single-precision (32-bit) floating-point elements in `a`
/// up to an integer value, and stores the results as packed single-precision
/// floating-point elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_ceil_ps)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundps))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_ceil_ps(a: __m128) -> __m128 {
    simd_ceil(a)
}

/// Round the lower double-precision (64-bit) floating-point element in `b`
/// up to an integer value, store the result as a double-precision
/// floating-point element in the lower element of the intrisic result,
/// and copies the upper element from `a` to the upper element
/// of the intrinsic result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_ceil_sd)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundsd))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_ceil_sd(a: __m128d, b: __m128d) -> __m128d {
    roundsd(a, b, _MM_FROUND_CEIL)
}

/// Round the lower single-precision (32-bit) floating-point element in `b`
/// up to an integer value, store the result as a single-precision
/// floating-point element in the lower element of the intrinsic result,
/// and copies the upper 3 packed elements from `a` to the upper elements
/// of the intrinsic result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_ceil_ss)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundss))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_ceil_ss(a: __m128, b: __m128) -> __m128 {
    roundss(a, b, _MM_FROUND_CEIL)
}

/// Round the packed double-precision (64-bit) floating-point elements in `a`
/// using the `ROUNDING` parameter, and stores the results as packed
/// double-precision floating-point elements.
/// Rounding is done according to the rounding parameter, which can be one of:
///
/// ```
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// // round to nearest, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
/// // round down, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;
/// // round up, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC;
/// // truncate, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;
/// // use MXCSR.RC; see `_MM_SET_ROUNDING_MODE`:
/// # let _x =
/// _MM_FROUND_CUR_DIRECTION;
/// # }
/// ```
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_round_pd)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundpd, ROUNDING = 0))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_round_pd<const ROUNDING: i32>(a: __m128d) -> __m128d {
    static_assert_imm4!(ROUNDING);
    roundpd(a, ROUNDING)
}

/// Round the packed single-precision (32-bit) floating-point elements in `a`
/// using the `ROUNDING` parameter, and stores the results as packed
/// single-precision floating-point elements.
/// Rounding is done according to the rounding parameter, which can be one of:
///
/// ```
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// // round to nearest, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
/// // round down, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;
/// // round up, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC;
/// // truncate, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;
/// // use MXCSR.RC; see `_MM_SET_ROUNDING_MODE`:
/// # let _x =
/// _MM_FROUND_CUR_DIRECTION;
/// # }
/// ```
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_round_ps)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundps, ROUNDING = 0))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_round_ps<const ROUNDING: i32>(a: __m128) -> __m128 {
    static_assert_imm4!(ROUNDING);
    roundps(a, ROUNDING)
}

/// Round the lower double-precision (64-bit) floating-point element in `b`
/// using the `ROUNDING` parameter, store the result as a double-precision
/// floating-point element in the lower element of the intrinsic result,
/// and copies the upper element from `a` to the upper element of the intrinsic
/// result.
/// Rounding is done according to the rounding parameter, which can be one of:
///
/// ```
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// // round to nearest, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
/// // round down, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;
/// // round up, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC;
/// // truncate, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;
/// // use MXCSR.RC; see `_MM_SET_ROUNDING_MODE`:
/// # let _x =
/// _MM_FROUND_CUR_DIRECTION;
/// # }
/// ```
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_round_sd)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundsd, ROUNDING = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_round_sd<const ROUNDING: i32>(a: __m128d, b: __m128d) -> __m128d {
    static_assert_imm4!(ROUNDING);
    roundsd(a, b, ROUNDING)
}

/// Round the lower single-precision (32-bit) floating-point element in `b`
/// using the `ROUNDING` parameter, store the result as a single-precision
/// floating-point element in the lower element of the intrinsic result,
/// and copies the upper 3 packed elements from `a` to the upper elements
/// of the instrinsic result.
/// Rounding is done according to the rounding parameter, which can be one of:
///
/// ```
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// // round to nearest, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
/// // round down, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;
/// // round up, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC;
/// // truncate, and suppress exceptions:
/// # let _x =
/// _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;
/// // use MXCSR.RC; see `_MM_SET_ROUNDING_MODE`:
/// # let _x =
/// _MM_FROUND_CUR_DIRECTION;
/// # }
/// ```
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_round_ss)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundss, ROUNDING = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_round_ss<const ROUNDING: i32>(a: __m128, b: __m128) -> __m128 {
    static_assert_imm4!(ROUNDING);
    roundss(a, b, ROUNDING)
}

/// Finds the minimum unsigned 16-bit element in the 128-bit __m128i vector,
/// returning a vector containing its value in its first position, and its
/// index
/// in its second position; all other elements are set to zero.
///
/// This intrinsic corresponds to the `VPHMINPOSUW` / `PHMINPOSUW`
/// instruction.
///
/// Arguments:
///
/// * `a` - A 128-bit vector of type `__m128i`.
///
/// Returns:
///
/// A 128-bit value where:
///
/// * bits `[15:0]` - contain the minimum value found in parameter `a`,
/// * bits `[18:16]` - contain the index of the minimum value
/// * remaining bits are set to `0`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_minpos_epu16)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(phminposuw))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_minpos_epu16(a: __m128i) -> __m128i {
    transmute(phminposuw(a.as_u16x8()))
}

/// Multiplies the low 32-bit integers from each packed 64-bit
/// element in `a` and `b`, and returns the signed 64-bit result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mul_epi32)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmuldq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_mul_epi32(a: __m128i, b: __m128i) -> __m128i {
    transmute(pmuldq(a.as_i32x4(), b.as_i32x4()))
}

/// Multiplies the packed 32-bit integers in `a` and `b`, producing intermediate
/// 64-bit integers, and returns the lowest 32-bit, whatever they might be,
/// reinterpreted as a signed integer. While `pmulld __m128i::splat(2),
/// __m128i::splat(2)` returns the obvious `__m128i::splat(4)`, due to wrapping
/// arithmetic `pmulld __m128i::splat(i32::MAX), __m128i::splat(2)` would
/// return a negative number.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mullo_epi32)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmulld))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_mullo_epi32(a: __m128i, b: __m128i) -> __m128i {
    transmute(simd_mul(a.as_i32x4(), b.as_i32x4()))
}

/// Subtracts 8-bit unsigned integer values and computes the absolute
/// values of the differences to the corresponding bits in the destination.
/// Then sums of the absolute differences are returned according to the bit
/// fields in the immediate operand.
///
/// The following algorithm is performed:
///
/// ```ignore
/// i = IMM8[2] * 4
/// j = IMM8[1:0] * 4
/// for k := 0 to 7
///     d0 = abs(a[i + k + 0] - b[j + 0])
///     d1 = abs(a[i + k + 1] - b[j + 1])
///     d2 = abs(a[i + k + 2] - b[j + 2])
///     d3 = abs(a[i + k + 3] - b[j + 3])
///     r[k] = d0 + d1 + d2 + d3
/// ```
///
/// Arguments:
///
/// * `a` - A 128-bit vector of type `__m128i`.
/// * `b` - A 128-bit vector of type `__m128i`.
/// * `IMM8` - An 8-bit immediate operand specifying how the absolute
///   differences are to be calculated
///     * Bit `[2]` specify the offset for operand `a`
///     * Bits `[1:0]` specify the offset for operand `b`
///
/// Returns:
///
/// * A `__m128i` vector containing the sums of the sets of   absolute
///   differences between both operands.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mpsadbw_epu8)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(mpsadbw, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_mpsadbw_epu8<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    static_assert_imm3!(IMM8);
    transmute(mpsadbw(a.as_u8x16(), b.as_u8x16(), IMM8 as u8))
}

/// Tests whether the specified bits in a 128-bit integer vector are all
/// zeros.
///
/// Arguments:
///
/// * `a` - A 128-bit integer vector containing the bits to be tested.
/// * `mask` - A 128-bit integer vector selecting which bits to test in
///   operand `a`.
///
/// Returns:
///
/// * `1` - if the specified bits are all zeros,
/// * `0` - otherwise.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_testz_si128)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(ptest))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_testz_si128(a: __m128i, mask: __m128i) -> i32 {
    ptestz(a.as_i64x2(), mask.as_i64x2())
}

/// Tests whether the specified bits in a 128-bit integer vector are all
/// ones.
///
/// Arguments:
///
/// * `a` - A 128-bit integer vector containing the bits to be tested.
/// * `mask` - A 128-bit integer vector selecting which bits to test in
///   operand `a`.
///
/// Returns:
///
/// * `1` - if the specified bits are all ones,
/// * `0` - otherwise.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_testc_si128)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(ptest))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_testc_si128(a: __m128i, mask: __m128i) -> i32 {
    ptestc(a.as_i64x2(), mask.as_i64x2())
}

/// Tests whether the specified bits in a 128-bit integer vector are
/// neither all zeros nor all ones.
///
/// Arguments:
///
/// * `a` - A 128-bit integer vector containing the bits to be tested.
/// * `mask` - A 128-bit integer vector selecting which bits to test in
///   operand `a`.
///
/// Returns:
///
/// * `1` - if the specified bits are neither all zeros nor all ones,
/// * `0` - otherwise.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_testnzc_si128)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(ptest))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_testnzc_si128(a: __m128i, mask: __m128i) -> i32 {
    ptestnzc(a.as_i64x2(), mask.as_i64x2())
}

/// Tests whether the specified bits in a 128-bit integer vector are all
/// zeros.
///
/// Arguments:
///
/// * `a` - A 128-bit integer vector containing the bits to be tested.
/// * `mask` - A 128-bit integer vector selecting which bits to test in
///   operand `a`.
///
/// Returns:
///
/// * `1` - if the specified bits are all zeros,
/// * `0` - otherwise.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_test_all_zeros)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(ptest))]
#[stable(feature = "simd_x86", since = "1.27.0")]
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
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_test_all_ones)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pcmpeqd))]
#[cfg_attr(test, assert_instr(ptest))]
#[stable(feature = "simd_x86", since = "1.27.0")]
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
///   operand `a`.
///
/// Returns:
///
/// * `1` - if the specified bits are neither all zeros nor all ones,
/// * `0` - otherwise.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_test_mix_ones_zeros)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(ptest))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_test_mix_ones_zeros(a: __m128i, mask: __m128i) -> i32 {
    _mm_testnzc_si128(a, mask)
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.sse41.pblendvb"]
    fn pblendvb(a: i8x16, b: i8x16, mask: i8x16) -> i8x16;
    #[link_name = "llvm.x86.sse41.blendvpd"]
    fn blendvpd(a: __m128d, b: __m128d, mask: __m128d) -> __m128d;
    #[link_name = "llvm.x86.sse41.blendvps"]
    fn blendvps(a: __m128, b: __m128, mask: __m128) -> __m128;
    #[link_name = "llvm.x86.sse41.blendpd"]
    fn blendpd(a: __m128d, b: __m128d, imm2: u8) -> __m128d;
    #[link_name = "llvm.x86.sse41.blendps"]
    fn blendps(a: __m128, b: __m128, imm4: u8) -> __m128;
    #[link_name = "llvm.x86.sse41.pblendw"]
    fn pblendw(a: i16x8, b: i16x8, imm8: u8) -> i16x8;
    #[link_name = "llvm.x86.sse41.insertps"]
    fn insertps(a: __m128, b: __m128, imm8: u8) -> __m128;
    #[link_name = "llvm.x86.sse41.pmaxsb"]
    fn pmaxsb(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.x86.sse41.pmaxuw"]
    fn pmaxuw(a: u16x8, b: u16x8) -> u16x8;
    #[link_name = "llvm.x86.sse41.pmaxsd"]
    fn pmaxsd(a: i32x4, b: i32x4) -> i32x4;
    #[link_name = "llvm.x86.sse41.pmaxud"]
    fn pmaxud(a: u32x4, b: u32x4) -> u32x4;
    #[link_name = "llvm.x86.sse41.pminsb"]
    fn pminsb(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.x86.sse41.pminuw"]
    fn pminuw(a: u16x8, b: u16x8) -> u16x8;
    #[link_name = "llvm.x86.sse41.pminsd"]
    fn pminsd(a: i32x4, b: i32x4) -> i32x4;
    #[link_name = "llvm.x86.sse41.pminud"]
    fn pminud(a: u32x4, b: u32x4) -> u32x4;
    #[link_name = "llvm.x86.sse41.packusdw"]
    fn packusdw(a: i32x4, b: i32x4) -> u16x8;
    #[link_name = "llvm.x86.sse41.dppd"]
    fn dppd(a: __m128d, b: __m128d, imm8: u8) -> __m128d;
    #[link_name = "llvm.x86.sse41.dpps"]
    fn dpps(a: __m128, b: __m128, imm8: u8) -> __m128;
    #[link_name = "llvm.x86.sse41.round.pd"]
    fn roundpd(a: __m128d, rounding: i32) -> __m128d;
    #[link_name = "llvm.x86.sse41.round.ps"]
    fn roundps(a: __m128, rounding: i32) -> __m128;
    #[link_name = "llvm.x86.sse41.round.sd"]
    fn roundsd(a: __m128d, b: __m128d, rounding: i32) -> __m128d;
    #[link_name = "llvm.x86.sse41.round.ss"]
    fn roundss(a: __m128, b: __m128, rounding: i32) -> __m128;
    #[link_name = "llvm.x86.sse41.phminposuw"]
    fn phminposuw(a: u16x8) -> u16x8;
    #[link_name = "llvm.x86.sse41.pmuldq"]
    fn pmuldq(a: i32x4, b: i32x4) -> i64x2;
    #[link_name = "llvm.x86.sse41.mpsadbw"]
    fn mpsadbw(a: u8x16, b: u8x16, imm8: u8) -> u16x8;
    #[link_name = "llvm.x86.sse41.ptestz"]
    fn ptestz(a: i64x2, mask: i64x2) -> i32;
    #[link_name = "llvm.x86.sse41.ptestc"]
    fn ptestc(a: i64x2, mask: i64x2) -> i32;
    #[link_name = "llvm.x86.sse41.ptestnzc"]
    fn ptestnzc(a: i64x2, mask: i64x2) -> i32;
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;
    use std::mem;
    use stdarch_test::simd_test;

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_blendv_epi8() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        #[rustfmt::skip]
        let b = _mm_setr_epi8(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        #[rustfmt::skip]
        let mask = _mm_setr_epi8(
            0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0, -1,
        );
        #[rustfmt::skip]
        let e = _mm_setr_epi8(
            0, 17, 2, 19, 4, 21, 6, 23, 8, 25, 10, 27, 12, 29, 14, 31,
        );
        assert_eq_m128i(_mm_blendv_epi8(a, b, mask), e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_blendv_pd() {
        let a = _mm_set1_pd(0.0);
        let b = _mm_set1_pd(1.0);
        let mask = transmute(_mm_setr_epi64x(0, -1));
        let r = _mm_blendv_pd(a, b, mask);
        let e = _mm_setr_pd(0.0, 1.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_blendv_ps() {
        let a = _mm_set1_ps(0.0);
        let b = _mm_set1_ps(1.0);
        let mask = transmute(_mm_setr_epi32(0, -1, 0, -1));
        let r = _mm_blendv_ps(a, b, mask);
        let e = _mm_setr_ps(0.0, 1.0, 0.0, 1.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_blend_pd() {
        let a = _mm_set1_pd(0.0);
        let b = _mm_set1_pd(1.0);
        let r = _mm_blend_pd::<0b10>(a, b);
        let e = _mm_setr_pd(0.0, 1.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_blend_ps() {
        let a = _mm_set1_ps(0.0);
        let b = _mm_set1_ps(1.0);
        let r = _mm_blend_ps::<0b1010>(a, b);
        let e = _mm_setr_ps(0.0, 1.0, 0.0, 1.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_blend_epi16() {
        let a = _mm_set1_epi16(0);
        let b = _mm_set1_epi16(1);
        let r = _mm_blend_epi16::<0b1010_1100>(a, b);
        let e = _mm_setr_epi16(0, 0, 1, 1, 0, 1, 0, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_extract_ps() {
        let a = _mm_setr_ps(0.0, 1.0, 2.0, 3.0);
        let r: f32 = transmute(_mm_extract_ps::<1>(a));
        assert_eq!(r, 1.0);
        let r: f32 = transmute(_mm_extract_ps::<3>(a));
        assert_eq!(r, 3.0);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_extract_epi8() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            -1, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15
        );
        let r1 = _mm_extract_epi8::<0>(a);
        let r2 = _mm_extract_epi8::<3>(a);
        assert_eq!(r1, 0xFF);
        assert_eq!(r2, 3);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_extract_epi32() {
        let a = _mm_setr_epi32(0, 1, 2, 3);
        let r = _mm_extract_epi32::<1>(a);
        assert_eq!(r, 1);
        let r = _mm_extract_epi32::<3>(a);
        assert_eq!(r, 3);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_insert_ps() {
        let a = _mm_set1_ps(1.0);
        let b = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let r = _mm_insert_ps::<0b11_00_1100>(a, b);
        let e = _mm_setr_ps(4.0, 1.0, 0.0, 0.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_insert_epi8() {
        let a = _mm_set1_epi8(0);
        let e = _mm_setr_epi8(0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r = _mm_insert_epi8::<1>(a, 32);
        assert_eq_m128i(r, e);
        let e = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0);
        let r = _mm_insert_epi8::<14>(a, 32);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_insert_epi32() {
        let a = _mm_set1_epi32(0);
        let e = _mm_setr_epi32(0, 32, 0, 0);
        let r = _mm_insert_epi32::<1>(a, 32);
        assert_eq_m128i(r, e);
        let e = _mm_setr_epi32(0, 0, 0, 32);
        let r = _mm_insert_epi32::<3>(a, 32);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_max_epi8() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            1, 4, 5, 8, 9, 12, 13, 16,
            17, 20, 21, 24, 25, 28, 29, 32,
        );
        #[rustfmt::skip]
        let b = _mm_setr_epi8(
            2, 3, 6, 7, 10, 11, 14, 15,
            18, 19, 22, 23, 26, 27, 30, 31,
        );
        let r = _mm_max_epi8(a, b);
        #[rustfmt::skip]
        let e = _mm_setr_epi8(
            2, 4, 6, 8, 10, 12, 14, 16,
            18, 20, 22, 24, 26, 28, 30, 32,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_max_epu16() {
        let a = _mm_setr_epi16(1, 4, 5, 8, 9, 12, 13, 16);
        let b = _mm_setr_epi16(2, 3, 6, 7, 10, 11, 14, 15);
        let r = _mm_max_epu16(a, b);
        let e = _mm_setr_epi16(2, 4, 6, 8, 10, 12, 14, 16);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_max_epi32() {
        let a = _mm_setr_epi32(1, 4, 5, 8);
        let b = _mm_setr_epi32(2, 3, 6, 7);
        let r = _mm_max_epi32(a, b);
        let e = _mm_setr_epi32(2, 4, 6, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_max_epu32() {
        let a = _mm_setr_epi32(1, 4, 5, 8);
        let b = _mm_setr_epi32(2, 3, 6, 7);
        let r = _mm_max_epu32(a, b);
        let e = _mm_setr_epi32(2, 4, 6, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_min_epi8_1() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            1, 4, 5, 8, 9, 12, 13, 16,
            17, 20, 21, 24, 25, 28, 29, 32,
        );
        #[rustfmt::skip]
        let b = _mm_setr_epi8(
            2, 3, 6, 7, 10, 11, 14, 15,
            18, 19, 22, 23, 26, 27, 30, 31,
        );
        let r = _mm_min_epi8(a, b);
        #[rustfmt::skip]
        let e = _mm_setr_epi8(
            1, 3, 5, 7, 9, 11, 13, 15,
            17, 19, 21, 23, 25, 27, 29, 31,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_min_epi8_2() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            1, -4, -5, 8, -9, -12, 13, -16,
            17, 20, 21, 24, 25, 28, 29, 32,
        );
        #[rustfmt::skip]
        let b = _mm_setr_epi8(
            2, -3, -6, 7, -10, -11, 14, -15,
            18, 19, 22, 23, 26, 27, 30, 31,
        );
        let r = _mm_min_epi8(a, b);
        #[rustfmt::skip]
        let e = _mm_setr_epi8(
            1, -4, -6, 7, -10, -12, 13, -16,
            17, 19, 21, 23, 25, 27, 29, 31,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_min_epu16() {
        let a = _mm_setr_epi16(1, 4, 5, 8, 9, 12, 13, 16);
        let b = _mm_setr_epi16(2, 3, 6, 7, 10, 11, 14, 15);
        let r = _mm_min_epu16(a, b);
        let e = _mm_setr_epi16(1, 3, 5, 7, 9, 11, 13, 15);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_min_epi32_1() {
        let a = _mm_setr_epi32(1, 4, 5, 8);
        let b = _mm_setr_epi32(2, 3, 6, 7);
        let r = _mm_min_epi32(a, b);
        let e = _mm_setr_epi32(1, 3, 5, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_min_epi32_2() {
        let a = _mm_setr_epi32(-1, 4, 5, -7);
        let b = _mm_setr_epi32(-2, 3, -6, 8);
        let r = _mm_min_epi32(a, b);
        let e = _mm_setr_epi32(-2, 3, -6, -7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_min_epu32() {
        let a = _mm_setr_epi32(1, 4, 5, 8);
        let b = _mm_setr_epi32(2, 3, 6, 7);
        let r = _mm_min_epu32(a, b);
        let e = _mm_setr_epi32(1, 3, 5, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_packus_epi32() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let b = _mm_setr_epi32(-1, -2, -3, -4);
        let r = _mm_packus_epi32(a, b);
        let e = _mm_setr_epi16(1, 2, 3, 4, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_cmpeq_epi64() {
        let a = _mm_setr_epi64x(0, 1);
        let b = _mm_setr_epi64x(0, 0);
        let r = _mm_cmpeq_epi64(a, b);
        let e = _mm_setr_epi64x(-1, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_cvtepi8_epi16() {
        let a = _mm_set1_epi8(10);
        let r = _mm_cvtepi8_epi16(a);
        let e = _mm_set1_epi16(10);
        assert_eq_m128i(r, e);
        let a = _mm_set1_epi8(-10);
        let r = _mm_cvtepi8_epi16(a);
        let e = _mm_set1_epi16(-10);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_cvtepi8_epi32() {
        let a = _mm_set1_epi8(10);
        let r = _mm_cvtepi8_epi32(a);
        let e = _mm_set1_epi32(10);
        assert_eq_m128i(r, e);
        let a = _mm_set1_epi8(-10);
        let r = _mm_cvtepi8_epi32(a);
        let e = _mm_set1_epi32(-10);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_cvtepi8_epi64() {
        let a = _mm_set1_epi8(10);
        let r = _mm_cvtepi8_epi64(a);
        let e = _mm_set1_epi64x(10);
        assert_eq_m128i(r, e);
        let a = _mm_set1_epi8(-10);
        let r = _mm_cvtepi8_epi64(a);
        let e = _mm_set1_epi64x(-10);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_cvtepi16_epi32() {
        let a = _mm_set1_epi16(10);
        let r = _mm_cvtepi16_epi32(a);
        let e = _mm_set1_epi32(10);
        assert_eq_m128i(r, e);
        let a = _mm_set1_epi16(-10);
        let r = _mm_cvtepi16_epi32(a);
        let e = _mm_set1_epi32(-10);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_cvtepi16_epi64() {
        let a = _mm_set1_epi16(10);
        let r = _mm_cvtepi16_epi64(a);
        let e = _mm_set1_epi64x(10);
        assert_eq_m128i(r, e);
        let a = _mm_set1_epi16(-10);
        let r = _mm_cvtepi16_epi64(a);
        let e = _mm_set1_epi64x(-10);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_cvtepi32_epi64() {
        let a = _mm_set1_epi32(10);
        let r = _mm_cvtepi32_epi64(a);
        let e = _mm_set1_epi64x(10);
        assert_eq_m128i(r, e);
        let a = _mm_set1_epi32(-10);
        let r = _mm_cvtepi32_epi64(a);
        let e = _mm_set1_epi64x(-10);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_cvtepu8_epi16() {
        let a = _mm_set1_epi8(10);
        let r = _mm_cvtepu8_epi16(a);
        let e = _mm_set1_epi16(10);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_cvtepu8_epi32() {
        let a = _mm_set1_epi8(10);
        let r = _mm_cvtepu8_epi32(a);
        let e = _mm_set1_epi32(10);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_cvtepu8_epi64() {
        let a = _mm_set1_epi8(10);
        let r = _mm_cvtepu8_epi64(a);
        let e = _mm_set1_epi64x(10);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_cvtepu16_epi32() {
        let a = _mm_set1_epi16(10);
        let r = _mm_cvtepu16_epi32(a);
        let e = _mm_set1_epi32(10);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_cvtepu16_epi64() {
        let a = _mm_set1_epi16(10);
        let r = _mm_cvtepu16_epi64(a);
        let e = _mm_set1_epi64x(10);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_cvtepu32_epi64() {
        let a = _mm_set1_epi32(10);
        let r = _mm_cvtepu32_epi64(a);
        let e = _mm_set1_epi64x(10);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_dp_pd() {
        let a = _mm_setr_pd(2.0, 3.0);
        let b = _mm_setr_pd(1.0, 4.0);
        let e = _mm_setr_pd(14.0, 0.0);
        assert_eq_m128d(_mm_dp_pd::<0b00110001>(a, b), e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_dp_ps() {
        let a = _mm_setr_ps(2.0, 3.0, 1.0, 10.0);
        let b = _mm_setr_ps(1.0, 4.0, 0.5, 10.0);
        let e = _mm_setr_ps(14.5, 0.0, 14.5, 0.0);
        assert_eq_m128(_mm_dp_ps::<0b01110101>(a, b), e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_floor_pd() {
        let a = _mm_setr_pd(2.5, 4.5);
        let r = _mm_floor_pd(a);
        let e = _mm_setr_pd(2.0, 4.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_floor_ps() {
        let a = _mm_setr_ps(2.5, 4.5, 8.5, 16.5);
        let r = _mm_floor_ps(a);
        let e = _mm_setr_ps(2.0, 4.0, 8.0, 16.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_floor_sd() {
        let a = _mm_setr_pd(2.5, 4.5);
        let b = _mm_setr_pd(-1.5, -3.5);
        let r = _mm_floor_sd(a, b);
        let e = _mm_setr_pd(-2.0, 4.5);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_floor_ss() {
        let a = _mm_setr_ps(2.5, 4.5, 8.5, 16.5);
        let b = _mm_setr_ps(-1.5, -3.5, -7.5, -15.5);
        let r = _mm_floor_ss(a, b);
        let e = _mm_setr_ps(-2.0, 4.5, 8.5, 16.5);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_ceil_pd() {
        let a = _mm_setr_pd(1.5, 3.5);
        let r = _mm_ceil_pd(a);
        let e = _mm_setr_pd(2.0, 4.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_ceil_ps() {
        let a = _mm_setr_ps(1.5, 3.5, 7.5, 15.5);
        let r = _mm_ceil_ps(a);
        let e = _mm_setr_ps(2.0, 4.0, 8.0, 16.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_ceil_sd() {
        let a = _mm_setr_pd(1.5, 3.5);
        let b = _mm_setr_pd(-2.5, -4.5);
        let r = _mm_ceil_sd(a, b);
        let e = _mm_setr_pd(-2.0, 3.5);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_ceil_ss() {
        let a = _mm_setr_ps(1.5, 3.5, 7.5, 15.5);
        let b = _mm_setr_ps(-2.5, -4.5, -8.5, -16.5);
        let r = _mm_ceil_ss(a, b);
        let e = _mm_setr_ps(-2.0, 3.5, 7.5, 15.5);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_round_pd() {
        let a = _mm_setr_pd(1.25, 3.75);
        let r = _mm_round_pd::<_MM_FROUND_TO_NEAREST_INT>(a);
        let e = _mm_setr_pd(1.0, 4.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_round_ps() {
        let a = _mm_setr_ps(2.25, 4.75, -1.75, -4.25);
        let r = _mm_round_ps::<_MM_FROUND_TO_ZERO>(a);
        let e = _mm_setr_ps(2.0, 4.0, -1.0, -4.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_round_sd() {
        let a = _mm_setr_pd(1.5, 3.5);
        let b = _mm_setr_pd(-2.5, -4.5);
        let old_mode = _MM_GET_ROUNDING_MODE();
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        let r = _mm_round_sd::<_MM_FROUND_CUR_DIRECTION>(a, b);
        _MM_SET_ROUNDING_MODE(old_mode);
        let e = _mm_setr_pd(-2.0, 3.5);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_round_ss() {
        let a = _mm_setr_ps(1.5, 3.5, 7.5, 15.5);
        let b = _mm_setr_ps(-1.75, -4.5, -8.5, -16.5);
        let old_mode = _MM_GET_ROUNDING_MODE();
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        let r = _mm_round_ss::<_MM_FROUND_CUR_DIRECTION>(a, b);
        _MM_SET_ROUNDING_MODE(old_mode);
        let e = _mm_setr_ps(-2.0, 3.5, 7.5, 15.5);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_minpos_epu16_1() {
        let a = _mm_setr_epi16(23, 18, 44, 97, 50, 13, 67, 66);
        let r = _mm_minpos_epu16(a);
        let e = _mm_setr_epi16(13, 5, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_minpos_epu16_2() {
        let a = _mm_setr_epi16(0, 18, 44, 97, 50, 13, 67, 66);
        let r = _mm_minpos_epu16(a);
        let e = _mm_setr_epi16(0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_mul_epi32() {
        {
            let a = _mm_setr_epi32(1, 1, 1, 1);
            let b = _mm_setr_epi32(1, 2, 3, 4);
            let r = _mm_mul_epi32(a, b);
            let e = _mm_setr_epi64x(1, 3);
            assert_eq_m128i(r, e);
        }
        {
            let a = _mm_setr_epi32(15, 2 /* ignored */, 1234567, 4 /* ignored */);
            let b = _mm_setr_epi32(
                -20, -256, /* ignored */
                666666, 666666, /* ignored */
            );
            let r = _mm_mul_epi32(a, b);
            let e = _mm_setr_epi64x(-300, 823043843622);
            assert_eq_m128i(r, e);
        }
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_mullo_epi32() {
        {
            let a = _mm_setr_epi32(1, 1, 1, 1);
            let b = _mm_setr_epi32(1, 2, 3, 4);
            let r = _mm_mullo_epi32(a, b);
            let e = _mm_setr_epi32(1, 2, 3, 4);
            assert_eq_m128i(r, e);
        }
        {
            let a = _mm_setr_epi32(15, -2, 1234567, 99999);
            let b = _mm_setr_epi32(-20, -256, 666666, -99999);
            let r = _mm_mullo_epi32(a, b);
            // Attention, most significant bit in r[2] is treated
            // as a sign bit:
            // 1234567 * 666666 = -1589877210
            let e = _mm_setr_epi32(-300, 512, -1589877210, -1409865409);
            assert_eq_m128i(r, e);
        }
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_minpos_epu16() {
        let a = _mm_setr_epi16(8, 7, 6, 5, 4, 1, 2, 3);
        let r = _mm_minpos_epu16(a);
        let e = _mm_setr_epi16(1, 5, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_mpsadbw_epu8() {
        #[rustfmt::skip]
        let a = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );

        let r = _mm_mpsadbw_epu8::<0b000>(a, a);
        let e = _mm_setr_epi16(0, 4, 8, 12, 16, 20, 24, 28);
        assert_eq_m128i(r, e);

        let r = _mm_mpsadbw_epu8::<0b001>(a, a);
        let e = _mm_setr_epi16(16, 12, 8, 4, 0, 4, 8, 12);
        assert_eq_m128i(r, e);

        let r = _mm_mpsadbw_epu8::<0b100>(a, a);
        let e = _mm_setr_epi16(16, 20, 24, 28, 32, 36, 40, 44);
        assert_eq_m128i(r, e);

        let r = _mm_mpsadbw_epu8::<0b101>(a, a);
        let e = _mm_setr_epi16(0, 4, 8, 12, 16, 20, 24, 28);
        assert_eq_m128i(r, e);

        let r = _mm_mpsadbw_epu8::<0b111>(a, a);
        let e = _mm_setr_epi16(32, 28, 24, 20, 16, 12, 8, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_testz_si128() {
        let a = _mm_set1_epi8(1);
        let mask = _mm_set1_epi8(0);
        let r = _mm_testz_si128(a, mask);
        assert_eq!(r, 1);
        let a = _mm_set1_epi8(0b101);
        let mask = _mm_set1_epi8(0b110);
        let r = _mm_testz_si128(a, mask);
        assert_eq!(r, 0);
        let a = _mm_set1_epi8(0b011);
        let mask = _mm_set1_epi8(0b100);
        let r = _mm_testz_si128(a, mask);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_testc_si128() {
        let a = _mm_set1_epi8(-1);
        let mask = _mm_set1_epi8(0);
        let r = _mm_testc_si128(a, mask);
        assert_eq!(r, 1);
        let a = _mm_set1_epi8(0b101);
        let mask = _mm_set1_epi8(0b110);
        let r = _mm_testc_si128(a, mask);
        assert_eq!(r, 0);
        let a = _mm_set1_epi8(0b101);
        let mask = _mm_set1_epi8(0b100);
        let r = _mm_testc_si128(a, mask);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_testnzc_si128() {
        let a = _mm_set1_epi8(0);
        let mask = _mm_set1_epi8(1);
        let r = _mm_testnzc_si128(a, mask);
        assert_eq!(r, 0);
        let a = _mm_set1_epi8(-1);
        let mask = _mm_set1_epi8(0);
        let r = _mm_testnzc_si128(a, mask);
        assert_eq!(r, 0);
        let a = _mm_set1_epi8(0b101);
        let mask = _mm_set1_epi8(0b110);
        let r = _mm_testnzc_si128(a, mask);
        assert_eq!(r, 1);
        let a = _mm_set1_epi8(0b101);
        let mask = _mm_set1_epi8(0b101);
        let r = _mm_testnzc_si128(a, mask);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_test_all_zeros() {
        let a = _mm_set1_epi8(1);
        let mask = _mm_set1_epi8(0);
        let r = _mm_test_all_zeros(a, mask);
        assert_eq!(r, 1);
        let a = _mm_set1_epi8(0b101);
        let mask = _mm_set1_epi8(0b110);
        let r = _mm_test_all_zeros(a, mask);
        assert_eq!(r, 0);
        let a = _mm_set1_epi8(0b011);
        let mask = _mm_set1_epi8(0b100);
        let r = _mm_test_all_zeros(a, mask);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_test_all_ones() {
        let a = _mm_set1_epi8(-1);
        let r = _mm_test_all_ones(a);
        assert_eq!(r, 1);
        let a = _mm_set1_epi8(0b101);
        let r = _mm_test_all_ones(a);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_test_mix_ones_zeros() {
        let a = _mm_set1_epi8(0);
        let mask = _mm_set1_epi8(1);
        let r = _mm_test_mix_ones_zeros(a, mask);
        assert_eq!(r, 0);
        let a = _mm_set1_epi8(-1);
        let mask = _mm_set1_epi8(0);
        let r = _mm_test_mix_ones_zeros(a, mask);
        assert_eq!(r, 0);
        let a = _mm_set1_epi8(0b101);
        let mask = _mm_set1_epi8(0b110);
        let r = _mm_test_mix_ones_zeros(a, mask);
        assert_eq!(r, 1);
        let a = _mm_set1_epi8(0b101);
        let mask = _mm_set1_epi8(0b101);
        let r = _mm_test_mix_ones_zeros(a, mask);
        assert_eq!(r, 0);
    }
}
