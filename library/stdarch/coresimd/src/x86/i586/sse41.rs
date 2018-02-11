//! Streaming SIMD Extensions 4.1 (SSE4.1)

use core::mem;

#[cfg(test)]
use stdsimd_test::assert_instr;

use simd_llvm::*;
use v128::*;
use x86::*;

// SSE4 rounding constans
/// round to nearest
pub const _MM_FROUND_TO_NEAREST_INT: i32 = 0x00;
/// round down
pub const _MM_FROUND_TO_NEG_INF: i32 = 0x01;
/// round up
pub const _MM_FROUND_TO_POS_INF: i32 = 0x02;
/// truncate
pub const _MM_FROUND_TO_ZERO: i32 = 0x03;
/// use MXCSR.RC; see `vendor::_MM_SET_ROUNDING_MODE`
pub const _MM_FROUND_CUR_DIRECTION: i32 = 0x04;
/// do not suppress exceptions
pub const _MM_FROUND_RAISE_EXC: i32 = 0x00;
/// suppress exceptions
pub const _MM_FROUND_NO_EXC: i32 = 0x08;
/// round to nearest and do not suppress exceptions
pub const _MM_FROUND_NINT: i32 = 0x00;
/// round down and do not suppress exceptions
pub const _MM_FROUND_FLOOR: i32 =
    (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_NEG_INF);
/// round up and do not suppress exceptions
pub const _MM_FROUND_CEIL: i32 =
    (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_POS_INF);
/// truncate and do not suppress exceptions
pub const _MM_FROUND_TRUNC: i32 = (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_ZERO);
/// use MXCSR.RC and do not suppress exceptions; see
/// `vendor::_MM_SET_ROUNDING_MODE`
pub const _MM_FROUND_RINT: i32 =
    (_MM_FROUND_RAISE_EXC | _MM_FROUND_CUR_DIRECTION);
/// use MXCSR.RC and suppress exceptions; see `vendor::_MM_SET_ROUNDING_MODE`
pub const _MM_FROUND_NEARBYINT: i32 =
    (_MM_FROUND_NO_EXC | _MM_FROUND_CUR_DIRECTION);

/// Blend packed 8-bit integers from `a` and `b` using `mask`
///
/// The high bit of each corresponding mask byte determines the selection.
/// If the high bit is set the element of `a` is selected. The element
/// of `b` is selected otherwise.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pblendvb))]
pub unsafe fn _mm_blendv_epi8(
    a: __m128i, b: __m128i, mask: __m128i
) -> __m128i {
    mem::transmute(pblendvb(a.as_i8x16(), b.as_i8x16(), mask.as_i8x16()))
}

/// Blend packed 16-bit integers from `a` and `b` using the mask `imm8`.
///
/// The mask bits determine the selection. A clear bit selects the
/// corresponding element of `a`, and a set bit the corresponding
/// element of `b`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pblendw, imm8 = 0xF0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_blend_epi16(a: __m128i, b: __m128i, imm8: i32) -> __m128i {
    let a = a.as_i16x8();
    let b = b.as_i16x8();
    macro_rules! call {
        ($imm8:expr) => { pblendw(a, b, $imm8) }
    }
    mem::transmute(constify_imm8!(imm8, call))
}

/// Blend packed double-precision (64-bit) floating-point elements from `a`
/// and `b` using `mask`
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(blendvpd))]
pub unsafe fn _mm_blendv_pd(a: __m128d, b: __m128d, mask: __m128d) -> __m128d {
    blendvpd(a, b, mask)
}

/// Blend packed single-precision (32-bit) floating-point elements from `a`
/// and `b` using `mask`
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(blendvps))]
pub unsafe fn _mm_blendv_ps(a: __m128, b: __m128, mask: __m128) -> __m128 {
    blendvps(a, b, mask)
}

/// Blend packed double-precision (64-bit) floating-point elements from `a`
/// and `b` using control mask `imm2`
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(blendpd, imm2 = 0b10))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_blend_pd(a: __m128d, b: __m128d, imm2: i32) -> __m128d {
    macro_rules! call {
        ($imm2:expr) => { blendpd(a, b, $imm2) }
    }
    constify_imm2!(imm2, call)
}

/// Blend packed single-precision (32-bit) floating-point elements from `a`
/// and `b` using mask `imm4`
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(blendps, imm4 = 0b0101))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_blend_ps(a: __m128, b: __m128, imm4: i32) -> __m128 {
    macro_rules! call {
        ($imm4:expr) => { blendps(a, b, $imm4) }
    }
    constify_imm4!(imm4, call)
}

/// Extract a single-precision (32-bit) floating-point element from `a`,
/// selected with `imm8`
#[inline]
#[target_feature(enable = "sse4.1")]
// TODO: Add test for Windows
#[cfg_attr(test, assert_instr(extractps, imm8 = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_extract_ps(a: __m128, imm8: i32) -> i32 {
    mem::transmute(simd_extract::<_, f32>(a, imm8 as u32 & 0b11))
}

/// Extract an 8-bit integer from `a`, selected with `imm8`. Returns a 32-bit
/// integer containing the zero-extended integer data.
///
/// See [LLVM commit D20468][https://reviews.llvm.org/D20468].
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pextrb, imm8 = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_extract_epi8(a: __m128i, imm8: i32) -> i32 {
    let imm8 = (imm8 & 15) as u32;
    simd_extract::<_, u8>(a.as_u8x16(), imm8) as i32
}

/// Extract an 32-bit integer from `a` selected with `imm8`
#[inline]
#[target_feature(enable = "sse4.1")]
// TODO: Add test for Windows
#[cfg_attr(test, assert_instr(extractps, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_extract_epi32(a: __m128i, imm8: i32) -> i32 {
    let imm8 = (imm8 & 3) as u32;
    simd_extract::<_, i32>(a.as_i32x4(), imm8)
}

/// Select a single value in `a` to store at some position in `b`,
/// Then zero elements according to `imm8`.
///
/// `imm8` specifies which bits from operand `a` will be copied, which bits in
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
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(insertps, imm8 = 0b1010))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_insert_ps(a: __m128, b: __m128, imm8: i32) -> __m128 {
    macro_rules! call {
        ($imm8:expr) => { insertps(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Return a copy of `a` with the 8-bit integer from `i` inserted at a
/// location specified by `imm8`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pinsrb, imm8 = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_insert_epi8(a: __m128i, i: i32, imm8: i32) -> __m128i {
    mem::transmute(simd_insert(a.as_i8x16(), (imm8 & 0b1111) as u32, i as i8))
}

/// Return a copy of `a` with the 32-bit integer from `i` inserted at a
/// location specified by `imm8`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pinsrd, imm8 = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_insert_epi32(a: __m128i, i: i32, imm8: i32) -> __m128i {
    mem::transmute(simd_insert(a.as_i32x4(), (imm8 & 0b11) as u32, i))
}

/// Compare packed 8-bit integers in `a` and `b` and return packed maximum
/// values in dst.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmaxsb))]
pub unsafe fn _mm_max_epi8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pmaxsb(a.as_i8x16(), b.as_i8x16()))
}

/// Compare packed unsigned 16-bit integers in `a` and `b`, and return packed
/// maximum.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmaxuw))]
pub unsafe fn _mm_max_epu16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pmaxuw(a.as_u16x8(), b.as_u16x8()))
}

/// Compare packed 32-bit integers in `a` and `b`, and return packed maximum
/// values.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmaxsd))]
pub unsafe fn _mm_max_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pmaxsd(a.as_i32x4(), b.as_i32x4()))
}

/// Compare packed unsigned 32-bit integers in `a` and `b`, and return packed
/// maximum values.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmaxud))]
pub unsafe fn _mm_max_epu32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pmaxud(a.as_u32x4(), b.as_u32x4()))
}

/// Compare packed 8-bit integers in `a` and `b` and return packed minimum
/// values in dst.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pminsb))]
pub unsafe fn _mm_min_epi8(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pminsb(a.as_i8x16(), b.as_i8x16()))
}

/// Compare packed unsigned 16-bit integers in `a` and `b`, and return packed
/// minimum.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pminuw))]
pub unsafe fn _mm_min_epu16(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pminuw(a.as_u16x8(), b.as_u16x8()))
}

/// Compare packed 32-bit integers in `a` and `b`, and return packed minimum
/// values.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pminsd))]
pub unsafe fn _mm_min_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pminsd(a.as_i32x4(), b.as_i32x4()))
}

/// Compare packed unsigned 32-bit integers in `a` and `b`, and return packed
/// minimum values.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pminud))]
pub unsafe fn _mm_min_epu32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pminud(a.as_u32x4(), b.as_u32x4()))
}

/// Convert packed 32-bit integers from `a` and `b` to packed 16-bit integers
/// using unsigned saturation
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(packusdw))]
pub unsafe fn _mm_packus_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(packusdw(a.as_i32x4(), b.as_i32x4()))
}

/// Compare packed 64-bit integers in `a` and `b` for equality
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pcmpeqq))]
pub unsafe fn _mm_cmpeq_epi64(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(simd_eq::<_, i64x2>(a.as_i64x2(), b.as_i64x2()))
}

/// Sign extend packed 8-bit integers in `a` to packed 16-bit integers
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovsxbw))]
pub unsafe fn _mm_cvtepi8_epi16(a: __m128i) -> __m128i {
    let a = a.as_i8x16();
    let a = simd_shuffle8::<_, ::v64::i8x8>(a, a, [0, 1, 2, 3, 4, 5, 6, 7]);
    mem::transmute(simd_cast::<_, i16x8>(a))
}

/// Sign extend packed 8-bit integers in `a` to packed 32-bit integers
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovsxbd))]
pub unsafe fn _mm_cvtepi8_epi32(a: __m128i) -> __m128i {
    let a = a.as_i8x16();
    let a = simd_shuffle4::<_, ::v32::i8x4>(a, a, [0, 1, 2, 3]);
    mem::transmute(simd_cast::<_, i32x4>(a))
}

/// Sign extend packed 8-bit integers in the low 8 bytes of `a` to packed
/// 64-bit integers
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovsxbq))]
pub unsafe fn _mm_cvtepi8_epi64(a: __m128i) -> __m128i {
    let a = a.as_i8x16();
    let a = simd_shuffle2::<_, ::v16::i8x2>(a, a, [0, 1]);
    mem::transmute(simd_cast::<_, i64x2>(a))
}

/// Sign extend packed 16-bit integers in `a` to packed 32-bit integers
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovsxwd))]
pub unsafe fn _mm_cvtepi16_epi32(a: __m128i) -> __m128i {
    let a = a.as_i16x8();
    let a = simd_shuffle4::<_, ::v64::i16x4>(a, a, [0, 1, 2, 3]);
    mem::transmute(simd_cast::<_, i32x4>(a))
}

/// Sign extend packed 16-bit integers in `a` to packed 64-bit integers
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovsxwq))]
pub unsafe fn _mm_cvtepi16_epi64(a: __m128i) -> __m128i {
    let a = a.as_i16x8();
    let a = simd_shuffle2::<_, ::v32::i16x2>(a, a, [0, 1]);
    mem::transmute(simd_cast::<_, i64x2>(a))
}

/// Sign extend packed 32-bit integers in `a` to packed 64-bit integers
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovsxdq))]
pub unsafe fn _mm_cvtepi32_epi64(a: __m128i) -> __m128i {
    let a = a.as_i32x4();
    let a = simd_shuffle2::<_, ::v64::i32x2>(a, a, [0, 1]);
    mem::transmute(simd_cast::<_, i64x2>(a))
}

/// Zero extend packed unsigned 8-bit integers in `a` to packed 16-bit integers
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovzxbw))]
pub unsafe fn _mm_cvtepu8_epi16(a: __m128i) -> __m128i {
    let a = a.as_u8x16();
    let a = simd_shuffle8::<_, ::v64::u8x8>(a, a, [0, 1, 2, 3, 4, 5, 6, 7]);
    mem::transmute(simd_cast::<_, i16x8>(a))
}

/// Zero extend packed unsigned 8-bit integers in `a` to packed 32-bit integers
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovzxbd))]
pub unsafe fn _mm_cvtepu8_epi32(a: __m128i) -> __m128i {
    let a = a.as_u8x16();
    let a = simd_shuffle4::<_, ::v32::u8x4>(a, a, [0, 1, 2, 3]);
    mem::transmute(simd_cast::<_, i32x4>(a))
}

/// Zero extend packed unsigned 8-bit integers in `a` to packed 64-bit integers
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovzxbq))]
pub unsafe fn _mm_cvtepu8_epi64(a: __m128i) -> __m128i {
    let a = a.as_u8x16();
    let a = simd_shuffle2::<_, ::v16::u8x2>(a, a, [0, 1]);
    mem::transmute(simd_cast::<_, i64x2>(a))
}

/// Zero extend packed unsigned 16-bit integers in `a`
/// to packed 32-bit integers
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovzxwd))]
pub unsafe fn _mm_cvtepu16_epi32(a: __m128i) -> __m128i {
    let a = a.as_u16x8();
    let a = simd_shuffle4::<_, ::v64::u16x4>(a, a, [0, 1, 2, 3]);
    mem::transmute(simd_cast::<_, i32x4>(a))
}

/// Zero extend packed unsigned 16-bit integers in `a`
/// to packed 64-bit integers
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovzxwq))]
pub unsafe fn _mm_cvtepu16_epi64(a: __m128i) -> __m128i {
    let a = a.as_u16x8();
    let a = simd_shuffle2::<_, ::v32::u16x2>(a, a, [0, 1]);
    mem::transmute(simd_cast::<_, i64x2>(a))
}

/// Zero extend packed unsigned 32-bit integers in `a`
/// to packed 64-bit integers
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmovzxdq))]
pub unsafe fn _mm_cvtepu32_epi64(a: __m128i) -> __m128i {
    let a = a.as_u32x4();
    let a = simd_shuffle2::<_, ::v64::u32x2>(a, a, [0, 1]);
    mem::transmute(simd_cast::<_, i64x2>(a))
}

/// Returns the dot product of two __m128d vectors.
///
/// `imm8[1:0]` is the broadcast mask, and `imm8[5:4]` is the condition mask.
/// If a condition mask bit is zero, the corresponding multiplication is
/// replaced by a value of `0.0`. If a broadcast mask bit is one, the result of
/// the dot product will be stored in the return value component. Otherwise if
/// the broadcast mask bit is zero then the return component will be zero.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(dppd, imm8 = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_dp_pd(a: __m128d, b: __m128d, imm8: i32) -> __m128d {
    macro_rules! call {
        ($imm8:expr) => { dppd(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Returns the dot product of two __m128 vectors.
///
/// `imm8[3:0]` is the broadcast mask, and `imm8[7:4]` is the condition mask.
/// If a condition mask bit is zero, the corresponding multiplication is
/// replaced by a value of `0.0`. If a broadcast mask bit is one, the result of
/// the dot product will be stored in the return value component. Otherwise if
/// the broadcast mask bit is zero then the return component will be zero.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(dpps, imm8 = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_dp_ps(a: __m128, b: __m128, imm8: i32) -> __m128 {
    macro_rules! call {
        ($imm8:expr) => { dpps(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Round the packed double-precision (64-bit) floating-point elements in `a`
/// down to an integer value, and store the results as packed double-precision
/// floating-point elements.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundpd))]
pub unsafe fn _mm_floor_pd(a: __m128d) -> __m128d {
    roundpd(a, _MM_FROUND_FLOOR)
}

/// Round the packed single-precision (32-bit) floating-point elements in `a`
/// down to an integer value, and store the results as packed single-precision
/// floating-point elements.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundps))]
pub unsafe fn _mm_floor_ps(a: __m128) -> __m128 {
    roundps(a, _MM_FROUND_FLOOR)
}

/// Round the lower double-precision (64-bit) floating-point element in `b`
/// down to an integer value, store the result as a double-precision
/// floating-point element in the lower element of the intrinsic result,
/// and copy the upper element from `a` to the upper element of the intrinsic
/// result.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundsd))]
pub unsafe fn _mm_floor_sd(a: __m128d, b: __m128d) -> __m128d {
    roundsd(a, b, _MM_FROUND_FLOOR)
}

/// Round the lower single-precision (32-bit) floating-point element in `b`
/// down to an integer value, store the result as a single-precision
/// floating-point element in the lower element of the intrinsic result,
/// and copy the upper 3 packed elements from `a` to the upper elements
/// of the intrinsic result.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundss))]
pub unsafe fn _mm_floor_ss(a: __m128, b: __m128) -> __m128 {
    roundss(a, b, _MM_FROUND_FLOOR)
}

/// Round the packed double-precision (64-bit) floating-point elements in `a`
/// up to an integer value, and store the results as packed double-precision
/// floating-point elements.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundpd))]
pub unsafe fn _mm_ceil_pd(a: __m128d) -> __m128d {
    roundpd(a, _MM_FROUND_CEIL)
}

/// Round the packed single-precision (32-bit) floating-point elements in `a`
/// up to an integer value, and store the results as packed single-precision
/// floating-point elements.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundps))]
pub unsafe fn _mm_ceil_ps(a: __m128) -> __m128 {
    roundps(a, _MM_FROUND_CEIL)
}

/// Round the lower double-precision (64-bit) floating-point element in `b`
/// up to an integer value, store the result as a double-precision
/// floating-point element in the lower element of the intrisic result,
/// and copy the upper element from `a` to the upper element
/// of the intrinsic result.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundsd))]
pub unsafe fn _mm_ceil_sd(a: __m128d, b: __m128d) -> __m128d {
    roundsd(a, b, _MM_FROUND_CEIL)
}

/// Round the lower single-precision (32-bit) floating-point element in `b`
/// up to an integer value, store the result as a single-precision
/// floating-point element in the lower element of the intrinsic result,
/// and copy the upper 3 packed elements from `a` to the upper elements
/// of the intrinsic result.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundss))]
pub unsafe fn _mm_ceil_ss(a: __m128, b: __m128) -> __m128 {
    roundss(a, b, _MM_FROUND_CEIL)
}

/// Round the packed double-precision (64-bit) floating-point elements in `a`
/// using the `rounding` parameter, and store the results as packed
/// double-precision floating-point elements.
/// Rounding is done according to the rounding parameter, which can be one of:
///
/// ```
/// use coresimd::vendor;
///
/// // round to nearest, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_NEAREST_INT | vendor::_MM_FROUND_NO_EXC);
/// // round down, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_NEG_INF | vendor::_MM_FROUND_NO_EXC);
/// // round up, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_POS_INF | vendor::_MM_FROUND_NO_EXC);
/// // truncate, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_ZERO | vendor::_MM_FROUND_NO_EXC);
/// // use MXCSR.RC; see `vendor::_MM_SET_ROUNDING_MODE`:
/// vendor::_MM_FROUND_CUR_DIRECTION;
/// ```
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundpd, rounding = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_round_pd(a: __m128d, rounding: i32) -> __m128d {
    macro_rules! call {
        ($imm4:expr) => { roundpd(a, $imm4) }
    }
    constify_imm4!(rounding, call)
}

/// Round the packed single-precision (32-bit) floating-point elements in `a`
/// using the `rounding` parameter, and store the results as packed
/// single-precision floating-point elements.
/// Rounding is done according to the rounding parameter, which can be one of:
///
/// ```
/// use coresimd::vendor;
///
/// // round to nearest, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_NEAREST_INT | vendor::_MM_FROUND_NO_EXC);
/// // round down, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_NEG_INF | vendor::_MM_FROUND_NO_EXC);
/// // round up, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_POS_INF | vendor::_MM_FROUND_NO_EXC);
/// // truncate, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_ZERO | vendor::_MM_FROUND_NO_EXC);
/// // use MXCSR.RC; see `vendor::_MM_SET_ROUNDING_MODE`:
/// vendor::_MM_FROUND_CUR_DIRECTION;
/// ```
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundps, rounding = 0))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm_round_ps(a: __m128, rounding: i32) -> __m128 {
    macro_rules! call {
        ($imm4:expr) => { roundps(a, $imm4) }
    }
    constify_imm4!(rounding, call)
}

/// Round the lower double-precision (64-bit) floating-point element in `b`
/// using the `rounding` parameter, store the result as a double-precision
/// floating-point element in the lower element of the intrinsic result,
/// and copy the upper element from `a` to the upper element of the intrinsic
/// result.
/// Rounding is done according to the rounding parameter, which can be one of:
///
/// ```
/// use coresimd::vendor;
///
/// // round to nearest, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_NEAREST_INT | vendor::_MM_FROUND_NO_EXC);
/// // round down, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_NEG_INF | vendor::_MM_FROUND_NO_EXC);
/// // round up, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_POS_INF | vendor::_MM_FROUND_NO_EXC);
/// // truncate, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_ZERO | vendor::_MM_FROUND_NO_EXC);
/// // use MXCSR.RC; see `vendor::_MM_SET_ROUNDING_MODE`:
/// vendor::_MM_FROUND_CUR_DIRECTION;
/// ```
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundsd, rounding = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_round_sd(a: __m128d, b: __m128d, rounding: i32) -> __m128d {
    macro_rules! call {
        ($imm4:expr) => { roundsd(a, b, $imm4) }
    }
    constify_imm4!(rounding, call)
}

/// Round the lower single-precision (32-bit) floating-point element in `b`
/// using the `rounding` parameter, store the result as a single-precision
/// floating-point element in the lower element of the intrinsic result,
/// and copy the upper 3 packed elements from `a` to the upper elements
/// of the instrinsic result.
/// Rounding is done according to the rounding parameter, which can be one of:
///
/// ```
/// use coresimd::vendor;
///
/// // round to nearest, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_NEAREST_INT | vendor::_MM_FROUND_NO_EXC);
/// // round down, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_NEG_INF | vendor::_MM_FROUND_NO_EXC);
/// // round up, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_POS_INF | vendor::_MM_FROUND_NO_EXC);
/// // truncate, and suppress exceptions:
/// (vendor::_MM_FROUND_TO_ZERO | vendor::_MM_FROUND_NO_EXC);
/// // use MXCSR.RC; see `vendor::_MM_SET_ROUNDING_MODE`:
/// vendor::_MM_FROUND_CUR_DIRECTION;
/// ```
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(roundss, rounding = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_round_ss(a: __m128, b: __m128, rounding: i32) -> __m128 {
    macro_rules! call {
        ($imm4:expr) => { roundss(a, b, $imm4) }
    }
    constify_imm4!(rounding, call)
}

/// Finds the minimum unsigned 16-bit element in the 128-bit __m128i vector,
/// returning a vector containing its value in its first position, and its
/// index
/// in its second position; all other elements are set to zero.
///
/// This intrinsic corresponds to the <c> VPHMINPOSUW / PHMINPOSUW </c>
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
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(phminposuw))]
pub unsafe fn _mm_minpos_epu16(a: __m128i) -> __m128i {
    mem::transmute(phminposuw(a.as_u16x8()))
}

/// Multiply the low 32-bit integers from each packed 64-bit
/// element in `a` and `b`, and return the signed 64-bit result.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmuldq))]
pub unsafe fn _mm_mul_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(pmuldq(a.as_i32x4(), b.as_i32x4()))
}

/// Multiply the packed 32-bit integers in `a` and `b`, producing intermediate
/// 64-bit integers, and returns the lowest 32-bit, whatever they might be,
/// reinterpreted as a signed integer. While `pmulld __m128i::splat(2),
/// __m128i::splat(2)` returns the obvious `__m128i::splat(4)`, due to wrapping
/// arithmetic `pmulld __m128i::splat(i32::MAX), __m128i::splat(2)` would
/// return a negative number.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pmulld))]
pub unsafe fn _mm_mullo_epi32(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(simd_mul(a.as_i32x4(), b.as_i32x4()))
}

/// Subtracts 8-bit unsigned integer values and computes the absolute
/// values of the differences to the corresponding bits in the destination.
/// Then sums of the absolute differences are returned according to the bit
/// fields in the immediate operand.
///
/// The following algorithm is performed:
///
/// ```ignore
/// i = imm8[2] * 4
/// j = imm8[1:0] * 4
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
/// * `imm8` - An 8-bit immediate operand specifying how the absolute
///            differences are to be calculated
///     * Bit `[2]` specify the offset for operand `a`
///     * Bits `[1:0]` specify the offset for operand `b`
///
/// Returns:
///
/// * A `__m128i` vector containing the sums of the sets of
///   absolute differences between both operands.
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(mpsadbw, imm8 = 0))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm_mpsadbw_epu8(a: __m128i, b: __m128i, imm8: i32) -> __m128i {
    let a = a.as_u8x16();
    let b = b.as_u8x16();
    macro_rules! call {
        ($imm8:expr) => { mpsadbw(a, b, $imm8) }
    }
    mem::transmute(constify_imm3!(imm8, call))
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
}

#[cfg(test)]
mod tests {
    use std::mem;
    use stdsimd_test::simd_test;
    use x86::*;

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_blendv_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm_setr_epi8(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let mask = _mm_setr_epi8(
            0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0, -1,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm_setr_epi8(
            0, 17, 2, 19, 4, 21, 6, 23, 8, 25, 10, 27, 12, 29, 14, 31,
        );
        assert_eq_m128i(_mm_blendv_epi8(a, b, mask), e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_blendv_pd() {
        let a = _mm_set1_pd(0.0);
        let b = _mm_set1_pd(1.0);
        let mask = mem::transmute(_mm_setr_epi64x(0, -1));
        let r = _mm_blendv_pd(a, b, mask);
        let e = _mm_setr_pd(0.0, 1.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_blendv_ps() {
        let a = _mm_set1_ps(0.0);
        let b = _mm_set1_ps(1.0);
        let mask = mem::transmute(_mm_setr_epi32(0, -1, 0, -1));
        let r = _mm_blendv_ps(a, b, mask);
        let e = _mm_setr_ps(0.0, 1.0, 0.0, 1.0);
        assert_eq_m128(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_blend_pd() {
        let a = _mm_set1_pd(0.0);
        let b = _mm_set1_pd(1.0);
        let r = _mm_blend_pd(a, b, 0b10);
        let e = _mm_setr_pd(0.0, 1.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_blend_ps() {
        let a = _mm_set1_ps(0.0);
        let b = _mm_set1_ps(1.0);
        let r = _mm_blend_ps(a, b, 0b1010);
        let e = _mm_setr_ps(0.0, 1.0, 0.0, 1.0);
        assert_eq_m128(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_blend_epi16() {
        let a = _mm_set1_epi16(0);
        let b = _mm_set1_epi16(1);
        let r = _mm_blend_epi16(a, b, 0b1010_1100);
        let e = _mm_setr_epi16(0, 0, 1, 1, 0, 1, 0, 1);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_extract_ps() {
        let a = _mm_setr_ps(0.0, 1.0, 2.0, 3.0);
        let r: f32 = mem::transmute(_mm_extract_ps(a, 1));
        assert_eq!(r, 1.0);
        let r: f32 = mem::transmute(_mm_extract_ps(a, 5));
        assert_eq!(r, 1.0);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_extract_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            -1, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15
        );
        let r1 = _mm_extract_epi8(a, 0);
        let r2 = _mm_extract_epi8(a, 19);
        assert_eq!(r1, 0xFF);
        assert_eq!(r2, 3);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_extract_epi32() {
        let a = _mm_setr_epi32(0, 1, 2, 3);
        let r = _mm_extract_epi32(a, 1);
        assert_eq!(r, 1);
        let r = _mm_extract_epi32(a, 5);
        assert_eq!(r, 1);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_insert_ps() {
        let a = _mm_set1_ps(1.0);
        let b = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let r = _mm_insert_ps(a, b, 0b11_00_1100);
        let e = _mm_setr_ps(4.0, 1.0, 0.0, 0.0);
        assert_eq_m128(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_insert_epi8() {
        let a = _mm_set1_epi8(0);
        let e = _mm_setr_epi8(0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r = _mm_insert_epi8(a, 32, 1);
        assert_eq_m128i(r, e);
        let r = _mm_insert_epi8(a, 32, 17);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_insert_epi32() {
        let a = _mm_set1_epi32(0);
        let e = _mm_setr_epi32(0, 32, 0, 0);
        let r = _mm_insert_epi32(a, 32, 1);
        assert_eq_m128i(r, e);
        let r = _mm_insert_epi32(a, 32, 5);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_max_epi8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 4, 5, 8, 9, 12, 13, 16,
            17, 20, 21, 24, 25, 28, 29, 32,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm_setr_epi8(
            2, 3, 6, 7, 10, 11, 14, 15,
            18, 19, 22, 23, 26, 27, 30, 31,
        );
        let r = _mm_max_epi8(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm_setr_epi8(
            2, 4, 6, 8, 10, 12, 14, 16,
            18, 20, 22, 24, 26, 28, 30, 32,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_max_epu16() {
        let a = _mm_setr_epi16(1, 4, 5, 8, 9, 12, 13, 16);
        let b = _mm_setr_epi16(2, 3, 6, 7, 10, 11, 14, 15);
        let r = _mm_max_epu16(a, b);
        let e = _mm_setr_epi16(2, 4, 6, 8, 10, 12, 14, 16);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_max_epi32() {
        let a = _mm_setr_epi32(1, 4, 5, 8);
        let b = _mm_setr_epi32(2, 3, 6, 7);
        let r = _mm_max_epi32(a, b);
        let e = _mm_setr_epi32(2, 4, 6, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_max_epu32() {
        let a = _mm_setr_epi32(1, 4, 5, 8);
        let b = _mm_setr_epi32(2, 3, 6, 7);
        let r = _mm_max_epu32(a, b);
        let e = _mm_setr_epi32(2, 4, 6, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_min_epi8_1() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, 4, 5, 8, 9, 12, 13, 16,
            17, 20, 21, 24, 25, 28, 29, 32,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm_setr_epi8(
            2, 3, 6, 7, 10, 11, 14, 15,
            18, 19, 22, 23, 26, 27, 30, 31,
        );
        let r = _mm_min_epi8(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm_setr_epi8(
            1, 3, 5, 7, 9, 11, 13, 15,
            17, 19, 21, 23, 25, 27, 29, 31,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_min_epi8_2() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            1, -4, -5, 8, -9, -12, 13, -16,
            17, 20, 21, 24, 25, 28, 29, 32,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = _mm_setr_epi8(
            2, -3, -6, 7, -10, -11, 14, -15,
            18, 19, 22, 23, 26, 27, 30, 31,
        );
        let r = _mm_min_epi8(a, b);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let e = _mm_setr_epi8(
            1, -4, -6, 7, -10, -12, 13, -16,
            17, 19, 21, 23, 25, 27, 29, 31,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_min_epu16() {
        let a = _mm_setr_epi16(1, 4, 5, 8, 9, 12, 13, 16);
        let b = _mm_setr_epi16(2, 3, 6, 7, 10, 11, 14, 15);
        let r = _mm_min_epu16(a, b);
        let e = _mm_setr_epi16(1, 3, 5, 7, 9, 11, 13, 15);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_min_epi32_1() {
        let a = _mm_setr_epi32(1, 4, 5, 8);
        let b = _mm_setr_epi32(2, 3, 6, 7);
        let r = _mm_min_epi32(a, b);
        let e = _mm_setr_epi32(1, 3, 5, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_min_epi32_2() {
        let a = _mm_setr_epi32(-1, 4, 5, -7);
        let b = _mm_setr_epi32(-2, 3, -6, 8);
        let r = _mm_min_epi32(a, b);
        let e = _mm_setr_epi32(-2, 3, -6, -7);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_min_epu32() {
        let a = _mm_setr_epi32(1, 4, 5, 8);
        let b = _mm_setr_epi32(2, 3, 6, 7);
        let r = _mm_min_epu32(a, b);
        let e = _mm_setr_epi32(1, 3, 5, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_packus_epi32() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let b = _mm_setr_epi32(-1, -2, -3, -4);
        let r = _mm_packus_epi32(a, b);
        let e = _mm_setr_epi16(1, 2, 3, 4, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_cmpeq_epi64() {
        let a = _mm_setr_epi64x(0, 1);
        let b = _mm_setr_epi64x(0, 0);
        let r = _mm_cmpeq_epi64(a, b);
        let e = _mm_setr_epi64x(-1, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
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

    #[simd_test = "sse4.1"]
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

    #[simd_test = "sse4.1"]
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

    #[simd_test = "sse4.1"]
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

    #[simd_test = "sse4.1"]
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

    #[simd_test = "sse4.1"]
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

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_cvtepu8_epi16() {
        let a = _mm_set1_epi8(10);
        let r = _mm_cvtepu8_epi16(a);
        let e = _mm_set1_epi16(10);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_cvtepu8_epi32() {
        let a = _mm_set1_epi8(10);
        let r = _mm_cvtepu8_epi32(a);
        let e = _mm_set1_epi32(10);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_cvtepu8_epi64() {
        let a = _mm_set1_epi8(10);
        let r = _mm_cvtepu8_epi64(a);
        let e = _mm_set1_epi64x(10);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_cvtepu16_epi32() {
        let a = _mm_set1_epi16(10);
        let r = _mm_cvtepu16_epi32(a);
        let e = _mm_set1_epi32(10);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_cvtepu16_epi64() {
        let a = _mm_set1_epi16(10);
        let r = _mm_cvtepu16_epi64(a);
        let e = _mm_set1_epi64x(10);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_cvtepu32_epi64() {
        let a = _mm_set1_epi32(10);
        let r = _mm_cvtepu32_epi64(a);
        let e = _mm_set1_epi64x(10);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_dp_pd() {
        let a = _mm_setr_pd(2.0, 3.0);
        let b = _mm_setr_pd(1.0, 4.0);
        let e = _mm_setr_pd(14.0, 0.0);
        assert_eq_m128d(_mm_dp_pd(a, b, 0b00110001), e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_dp_ps() {
        let a = _mm_setr_ps(2.0, 3.0, 1.0, 10.0);
        let b = _mm_setr_ps(1.0, 4.0, 0.5, 10.0);
        let e = _mm_setr_ps(14.5, 0.0, 14.5, 0.0);
        assert_eq_m128(_mm_dp_ps(a, b, 0b01110101), e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_floor_pd() {
        let a = _mm_setr_pd(2.5, 4.5);
        let r = _mm_floor_pd(a);
        let e = _mm_setr_pd(2.0, 4.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_floor_ps() {
        let a = _mm_setr_ps(2.5, 4.5, 8.5, 16.5);
        let r = _mm_floor_ps(a);
        let e = _mm_setr_ps(2.0, 4.0, 8.0, 16.0);
        assert_eq_m128(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_floor_sd() {
        let a = _mm_setr_pd(2.5, 4.5);
        let b = _mm_setr_pd(-1.5, -3.5);
        let r = _mm_floor_sd(a, b);
        let e = _mm_setr_pd(-2.0, 4.5);
        assert_eq_m128d(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_floor_ss() {
        let a = _mm_setr_ps(2.5, 4.5, 8.5, 16.5);
        let b = _mm_setr_ps(-1.5, -3.5, -7.5, -15.5);
        let r = _mm_floor_ss(a, b);
        let e = _mm_setr_ps(-2.0, 4.5, 8.5, 16.5);
        assert_eq_m128(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_ceil_pd() {
        let a = _mm_setr_pd(1.5, 3.5);
        let r = _mm_ceil_pd(a);
        let e = _mm_setr_pd(2.0, 4.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_ceil_ps() {
        let a = _mm_setr_ps(1.5, 3.5, 7.5, 15.5);
        let r = _mm_ceil_ps(a);
        let e = _mm_setr_ps(2.0, 4.0, 8.0, 16.0);
        assert_eq_m128(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_ceil_sd() {
        let a = _mm_setr_pd(1.5, 3.5);
        let b = _mm_setr_pd(-2.5, -4.5);
        let r = _mm_ceil_sd(a, b);
        let e = _mm_setr_pd(-2.0, 3.5);
        assert_eq_m128d(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_ceil_ss() {
        let a = _mm_setr_ps(1.5, 3.5, 7.5, 15.5);
        let b = _mm_setr_ps(-2.5, -4.5, -8.5, -16.5);
        let r = _mm_ceil_ss(a, b);
        let e = _mm_setr_ps(-2.0, 3.5, 7.5, 15.5);
        assert_eq_m128(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_round_pd() {
        let a = _mm_setr_pd(1.25, 3.75);
        let r = _mm_round_pd(a, _MM_FROUND_TO_NEAREST_INT);
        let e = _mm_setr_pd(1.0, 4.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_round_ps() {
        let a = _mm_setr_ps(2.25, 4.75, -1.75, -4.25);
        let r = _mm_round_ps(a, _MM_FROUND_TO_ZERO);
        let e = _mm_setr_ps(2.0, 4.0, -1.0, -4.0);
        assert_eq_m128(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_round_sd() {
        let a = _mm_setr_pd(1.5, 3.5);
        let b = _mm_setr_pd(-2.5, -4.5);
        let old_mode = _MM_GET_ROUNDING_MODE();
        _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
        let r = _mm_round_sd(a, b, _MM_FROUND_CUR_DIRECTION);
        _MM_SET_ROUNDING_MODE(old_mode);
        let e = _mm_setr_pd(-2.0, 3.5);
        assert_eq_m128d(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_round_ss() {
        let a = _mm_setr_ps(1.5, 3.5, 7.5, 15.5);
        let b = _mm_setr_ps(-1.75, -4.5, -8.5, -16.5);
        let old_mode = _MM_GET_ROUNDING_MODE();
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        let r = _mm_round_ss(a, b, _MM_FROUND_CUR_DIRECTION);
        _MM_SET_ROUNDING_MODE(old_mode);
        let e = _mm_setr_ps(-2.0, 3.5, 7.5, 15.5);
        assert_eq_m128(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_minpos_epu16_1() {
        let a = _mm_setr_epi16(23, 18, 44, 97, 50, 13, 67, 66);
        let r = _mm_minpos_epu16(a);
        let e = _mm_setr_epi16(13, 5, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_minpos_epu16_2() {
        let a = _mm_setr_epi16(0, 18, 44, 97, 50, 13, 67, 66);
        let r = _mm_minpos_epu16(a);
        let e = _mm_setr_epi16(0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_mul_epi32() {
        {
            let a = _mm_setr_epi32(1, 1, 1, 1);
            let b = _mm_setr_epi32(1, 2, 3, 4);
            let r = _mm_mul_epi32(a, b);
            let e = _mm_setr_epi64x(1, 3);
            assert_eq_m128i(r, e);
        }
        {
            let a = _mm_setr_epi32(
                15,
                2, /* ignored */
                1234567,
                4, /* ignored */
            );
            let b = _mm_setr_epi32(
                -20,
                -256, /* ignored */
                666666,
                666666, /* ignored */
            );
            let r = _mm_mul_epi32(a, b);
            let e = _mm_setr_epi64x(-300, 823043843622);
            assert_eq_m128i(r, e);
        }
    }

    #[simd_test = "sse4.1"]
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

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_minpos_epu16() {
        let a = _mm_setr_epi16(8, 7, 6, 5, 4, 1, 2, 3);
        let r = _mm_minpos_epu16(a);
        let e = _mm_setr_epi16(1, 5, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn test_mm_mpsadbw_epu8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
        );

        let r = _mm_mpsadbw_epu8(a, a, 0b000);
        let e = _mm_setr_epi16(0, 4, 8, 12, 16, 20, 24, 28);
        assert_eq_m128i(r, e);

        let r = _mm_mpsadbw_epu8(a, a, 0b001);
        let e = _mm_setr_epi16(16, 12, 8, 4, 0, 4, 8, 12);
        assert_eq_m128i(r, e);

        let r = _mm_mpsadbw_epu8(a, a, 0b100);
        let e = _mm_setr_epi16(16, 20, 24, 28, 32, 36, 40, 44);
        assert_eq_m128i(r, e);

        let r = _mm_mpsadbw_epu8(a, a, 0b101);
        let e = _mm_setr_epi16(0, 4, 8, 12, 16, 20, 24, 28);
        assert_eq_m128i(r, e);

        let r = _mm_mpsadbw_epu8(a, a, 0b111);
        let e = _mm_setr_epi16(32, 28, 24, 20, 16, 12, 8, 4);
        assert_eq_m128i(r, e);
    }
}
