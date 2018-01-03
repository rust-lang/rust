//! `i686` Streaming SIMD Extensions (SSE)

use v128::f32x4;
use v64::*;
use core::mem;
use x86::i586;
use x86::i686::mmx;

#[cfg(test)]
use stdsimd_test::assert_instr;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.sse.cvtpi2ps"]
    fn cvtpi2ps(a: f32x4, b: __m64) -> f32x4;
    #[link_name = "llvm.x86.mmx.maskmovq"]
    fn maskmovq(a: __m64, mask: __m64, mem_addr: *mut i8);
    #[link_name = "llvm.x86.mmx.pextr.w"]
    fn pextrw(a: __m64, imm8: i32) -> i32;
    #[link_name = "llvm.x86.mmx.pinsr.w"]
    fn pinsrw(a: __m64, d: i32, imm8: i32) -> __m64;
    #[link_name = "llvm.x86.mmx.pmovmskb"]
    fn pmovmskb(a: __m64) -> i32;
    #[link_name = "llvm.x86.sse.pshuf.w"]
    fn pshufw(a: __m64, imm8: i8) -> __m64;
    #[link_name = "llvm.x86.mmx.pmaxs.w"]
    fn pmaxsw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pmaxu.b"]
    fn pmaxub(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pmins.w"]
    fn pminsw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pminu.b"]
    fn pminub(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pmulhu.w"]
    fn pmulhuw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pavg.b"]
    fn pavgb(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pavg.w"]
    fn pavgw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.psad.bw"]
    fn psadbw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.sse.cvtps2pi"]
    fn cvtps2pi(a: f32x4) -> __m64;
    #[link_name = "llvm.x86.sse.cvttps2pi"]
    fn cvttps2pi(a: f32x4) -> __m64;
}

/// Compares the packed 16-bit signed integers of `a` and `b` writing the
/// greatest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pmaxsw))]
pub unsafe fn _mm_max_pi16(a: __m64, b: __m64) -> __m64 {
    pmaxsw(a, b)
}

/// Compares the packed 16-bit signed integers of `a` and `b` writing the
/// greatest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pmaxsw))]
pub unsafe fn _m_pmaxsw(a: __m64, b: __m64) -> __m64 {
    _mm_max_pi16(a, b)
}

/// Compares the packed 8-bit signed integers of `a` and `b` writing the
/// greatest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pmaxub))]
pub unsafe fn _mm_max_pu8(a: __m64, b: __m64) -> __m64 {
    pmaxub(a, b)
}

/// Compares the packed 8-bit signed integers of `a` and `b` writing the
/// greatest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pmaxub))]
pub unsafe fn _m_pmaxub(a: __m64, b: __m64) -> __m64 {
    _mm_max_pu8(a, b)
}

/// Compares the packed 16-bit signed integers of `a` and `b` writing the
/// smallest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pminsw))]
pub unsafe fn _mm_min_pi16(a: __m64, b: __m64) -> __m64 {
    pminsw(a, b)
}

/// Compares the packed 16-bit signed integers of `a` and `b` writing the
/// smallest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pminsw))]
pub unsafe fn _m_pminsw(a: __m64, b: __m64) -> __m64 {
    _mm_min_pi16(a, b)
}

/// Compares the packed 8-bit signed integers of `a` and `b` writing the
/// smallest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pminub))]
pub unsafe fn _mm_min_pu8(a: __m64, b: __m64) -> __m64 {
    pminub(a, b)
}

/// Compares the packed 8-bit signed integers of `a` and `b` writing the
/// smallest value into the result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pminub))]
pub unsafe fn _m_pminub(a: __m64, b: __m64) -> __m64 {
    _mm_min_pu8(a, b)
}

/// Multiplies packed 16-bit unsigned integer values and writes the
/// high-order 16 bits of each 32-bit product to the corresponding bits in
/// the destination.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pmulhuw))]
pub unsafe fn _mm_mulhi_pu16(a: __m64, b: __m64) -> __m64 {
    pmulhuw(a, b)
}

/// Multiplies packed 16-bit unsigned integer values and writes the
/// high-order 16 bits of each 32-bit product to the corresponding bits in
/// the destination.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pmulhuw))]
pub unsafe fn _m_pmulhuw(a: __m64, b: __m64) -> __m64 {
    _mm_mulhi_pu16(a, b)
}

/// Computes the rounded averages of the packed unsigned 8-bit integer
/// values and writes the averages to the corresponding bits in the
/// destination.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pavgb))]
pub unsafe fn _mm_avg_pu8(a: __m64, b: __m64) -> __m64 {
    pavgb(a, b)
}

/// Computes the rounded averages of the packed unsigned 8-bit integer
/// values and writes the averages to the corresponding bits in the
/// destination.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pavgb))]
pub unsafe fn _m_pavgb(a: __m64, b: __m64) -> __m64 {
    _mm_avg_pu8(a, b)
}

/// Computes the rounded averages of the packed unsigned 16-bit integer
/// values and writes the averages to the corresponding bits in the
/// destination.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pavgw))]
pub unsafe fn _mm_avg_pu16(a: __m64, b: __m64) -> __m64 {
    pavgw(a, b)
}

/// Computes the rounded averages of the packed unsigned 16-bit integer
/// values and writes the averages to the corresponding bits in the
/// destination.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pavgw))]
pub unsafe fn _m_pavgw(a: __m64, b: __m64) -> __m64 {
    _mm_avg_pu16(a, b)
}

/// Subtracts the corresponding 8-bit unsigned integer values of the two
/// 64-bit vector operands and computes the absolute value for each of the
/// difference. Then sum of the 8 absolute differences is written to the
/// bits [15:0] of the destination; the remaining bits [63:16] are cleared.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(psadbw))]
pub unsafe fn _mm_sad_pu8(a: __m64, b: __m64) -> __m64 {
    psadbw(a, b)
}

/// Subtracts the corresponding 8-bit unsigned integer values of the two
/// 64-bit vector operands and computes the absolute value for each of the
/// difference. Then sum of the 8 absolute differences is written to the
/// bits [15:0] of the destination; the remaining bits [63:16] are cleared.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(psadbw))]
pub unsafe fn _m_psadbw(a: __m64, b: __m64) -> __m64 {
    _mm_sad_pu8(a, b)
}

/// Converts two elements of a 64-bit vector of [2 x i32] into two
/// floating point values and writes them to the lower 64-bits of the
/// destination. The remaining higher order elements of the destination are
/// copied from the corresponding elements in the first operand.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(cvtpi2ps))]
pub unsafe fn _mm_cvtpi32_ps(a: f32x4, b: i32x2) -> f32x4 {
    cvtpi2ps(a, mem::transmute(b))
}

/// Converts two elements of a 64-bit vector of [2 x i32] into two
/// floating point values and writes them to the lower 64-bits of the
/// destination. The remaining higher order elements of the destination are
/// copied from the corresponding elements in the first operand.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(cvtpi2ps))]
pub unsafe fn _mm_cvt_pi2ps(a: f32x4, b: i32x2) -> f32x4 {
    _mm_cvtpi32_ps(a, b)
}

/// Converts a 64-bit vector of [4 x i16] into a 128-bit vector of [4 x
/// float].
#[inline(always)]
#[target_feature = "+sse"]
pub unsafe fn _mm_cvtpi16_ps(a: __m64) -> f32x4 {
    let b = mmx::_mm_setzero_si64();
    let b = mmx::_mm_cmpgt_pi16(mem::transmute(b), a);
    let c = mmx::_mm_unpackhi_pi16(a, b);
    let r = i586::_mm_setzero_ps();
    let r = cvtpi2ps(r, mem::transmute(c));
    let r = i586::_mm_movelh_ps(r, r);
    let c = mmx::_mm_unpacklo_pi16(a, b);
    cvtpi2ps(r, mem::transmute(c))
}

/// Converts a 64-bit vector of 16-bit unsigned integer values into a
/// 128-bit vector of [4 x float].
#[inline(always)]
#[target_feature = "+sse"]
pub unsafe fn _mm_cvtpu16_ps(a: __m64) -> f32x4 {
    let b = mmx::_mm_setzero_si64();
    let c = mmx::_mm_unpackhi_pi16(a, b);
    let r = i586::_mm_setzero_ps();
    let r = cvtpi2ps(r, c);
    let r = i586::_mm_movelh_ps(r, r);
    let c = mmx::_mm_unpacklo_pi16(a, b);
    cvtpi2ps(r, c)
}

/// Converts the lower four 8-bit values from a 64-bit vector of [8 x i8]
/// into a 128-bit vector of [4 x float].
#[inline(always)]
#[target_feature = "+sse"]
pub unsafe fn _mm_cvtpi8_ps(a: __m64) -> f32x4 {
    let b = mmx::_mm_setzero_si64();
    let b = mmx::_mm_cmpgt_pi8(b, a);
    let b = mmx::_mm_unpacklo_pi8(a, b);
    _mm_cvtpi16_ps(b)
}

/// Converts the lower four unsigned 8-bit integer values from a 64-bit
/// vector of [8 x u8] into a 128-bit vector of [4 x float].
#[inline(always)]
#[target_feature = "+sse"]
pub unsafe fn _mm_cvtpu8_ps(a: __m64) -> f32x4 {
    let b = mmx::_mm_setzero_si64();
    let b = mmx::_mm_unpacklo_pi8(a, b);
    _mm_cvtpi16_ps(b)
}

/// Converts the two 32-bit signed integer values from each 64-bit vector
/// operand of [2 x i32] into a 128-bit vector of [4 x float].
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(cvtpi2ps))]
pub unsafe fn _mm_cvtpi32x2_ps(a: i32x2, b: i32x2) -> f32x4 {
    let c = i586::_mm_setzero_ps();
    let c = _mm_cvtpi32_ps(c, b);
    let c = i586::_mm_movelh_ps(c, c);
    _mm_cvtpi32_ps(c, a)
}

/// Conditionally copies the values from each 8-bit element in the first
/// 64-bit integer vector operand to the specified memory location, as
/// specified by the most significant bit in the corresponding element in the
/// second 64-bit integer vector operand.
///
/// To minimize caching, the data is flagged as non-temporal
/// (unlikely to be used again soon).
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(maskmovq))]
pub unsafe fn _mm_maskmove_si64(a: __m64, mask: __m64, mem_addr: *mut i8) {
    maskmovq(a, mask, mem_addr)
}

/// Conditionally copies the values from each 8-bit element in the first
/// 64-bit integer vector operand to the specified memory location, as
/// specified by the most significant bit in the corresponding element in the
/// second 64-bit integer vector operand.
///
/// To minimize caching, the data is flagged as non-temporal
/// (unlikely to be used again soon).
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(maskmovq))]
pub unsafe fn _m_maskmovq(a: __m64, mask: __m64, mem_addr: *mut i8) {
    _mm_maskmove_si64(a, mask, mem_addr)
}

/// Extracts 16-bit element from a 64-bit vector of [4 x i16] and
/// returns it, as specified by the immediate integer operand.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pextrw, imm2 = 0))]
pub unsafe fn _mm_extract_pi16(a: i16x4, imm2: i32) -> i16 {
    macro_rules! call {
        ($imm2:expr) => { pextrw(mem::transmute(a), $imm2) as i16 }
    }
    constify_imm2!(imm2, call)
}

/// Extracts 16-bit element from a 64-bit vector of [4 x i16] and
/// returns it, as specified by the immediate integer operand.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pextrw, imm2 = 0))]
pub unsafe fn _m_pextrw(a: i16x4, imm2: i32) -> i16 {
    _mm_extract_pi16(a, imm2)
}

/// Copies data from the 64-bit vector of [4 x i16] to the destination,
/// and inserts the lower 16-bits of an integer operand at the 16-bit offset
/// specified by the immediate operand `n`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pinsrw, imm2 = 0))]
pub unsafe fn _mm_insert_pi16(a: __m64, d: i32, imm2: i32) -> __m64 {
    macro_rules! call {
        ($imm2:expr) => { pinsrw(a, d, $imm2) }
    }
    constify_imm2!(imm2, call)
}

/// Copies data from the 64-bit vector of [4 x i16] to the destination,
/// and inserts the lower 16-bits of an integer operand at the 16-bit offset
/// specified by the immediate operand `n`.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pinsrw, imm2 = 0))]
pub unsafe fn _m_pinsrw(a: __m64, d: i32, imm2: i32) -> __m64 {
    _mm_insert_pi16(a, d, imm2)
}

/// Takes the most significant bit from each 8-bit element in a 64-bit
/// integer vector to create a 16-bit mask value. Zero-extends the value to
/// 32-bit integer and writes it to the destination.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pmovmskb))]
pub unsafe fn _mm_movemask_pi8(a: i16x4) -> i32 {
    pmovmskb(mem::transmute(a))
}

/// Takes the most significant bit from each 8-bit element in a 64-bit
/// integer vector to create a 16-bit mask value. Zero-extends the value to
/// 32-bit integer and writes it to the destination.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pmovmskb))]
pub unsafe fn _m_pmovmskb(a: i16x4) -> i32 {
    _mm_movemask_pi8(a)
}

/// Shuffles the 4 16-bit integers from a 64-bit integer vector to the
/// destination, as specified by the immediate value operand.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pshufw, imm8 = 0))]
pub unsafe fn _mm_shuffle_pi16(a: __m64, imm8: i32) -> __m64 {
    macro_rules! call {
        ($imm8:expr) => { pshufw(a, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Shuffles the 4 16-bit integers from a 64-bit integer vector to the
/// destination, as specified by the immediate value operand.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(pshufw, imm8 = 0))]
pub unsafe fn _m_pshufw(a: __m64, imm8: i32) -> __m64 {
    _mm_shuffle_pi16(a, imm8)
}

/// Convert the two lower packed single-precision (32-bit) floating-point
/// elements in `a` to packed 32-bit integers with truncation.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(cvttps2pi))]
pub unsafe fn _mm_cvttps_pi32(a: f32x4) -> i32x2 {
    mem::transmute(cvttps2pi(a))
}

/// Convert the two lower packed single-precision (32-bit) floating-point
/// elements in `a` to packed 32-bit integers with truncation.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(cvttps2pi))]
pub unsafe fn _mm_cvtt_ps2pi(a: f32x4) -> i32x2 {
    _mm_cvttps_pi32(a)
}

/// Convert the two lower packed single-precision (32-bit) floating-point
/// elements in `a` to packed 32-bit integers.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(cvtps2pi))]
pub unsafe fn _mm_cvtps_pi32(a: f32x4) -> __m64 {
    cvtps2pi(a)
}

/// Convert the two lower packed single-precision (32-bit) floating-point
/// elements in `a` to packed 32-bit integers.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(cvtps2pi))]
pub unsafe fn _mm_cvt_ps2pi(a: f32x4) -> __m64 {
    _mm_cvtps_pi32(a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to
/// packed 16-bit integers.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(cvtps2pi))]
pub unsafe fn _mm_cvtps_pi16(a: f32x4) -> __m64 {
    let b = _mm_cvtps_pi32(a);
    let a = i586::_mm_movehl_ps(a, a);
    let c = _mm_cvtps_pi32(a);
    mmx::_mm_packs_pi32(b, c)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to
/// packed 8-bit integers, and returns theem in the lower 4 elements of the
/// result.
#[inline(always)]
#[target_feature = "+sse"]
#[cfg_attr(test, assert_instr(cvtps2pi))]
pub unsafe fn _mm_cvtps_pi8(a: f32x4) -> __m64 {
    let b = _mm_cvtps_pi16(a);
    let c = mmx::_mm_setzero_si64();
    mmx::_mm_packs_pi16(b, c)
}

#[cfg(test)]
mod tests {
    #[cfg(not(windows))]
    use std::mem;

    use v128::f32x4;
    use v64::{i16x4, i32x2, i8x8, u16x4, u8x8};
    use x86::i686::sse;
    use stdsimd_test::simd_test;

    #[simd_test = "sse"]
    unsafe fn _mm_max_pi16() {
        let a = i16x4::new(-1, 6, -3, 8);
        let b = i16x4::new(5, -2, 7, -4);
        let r = i16x4::new(5, 6, 7, 8);

        assert_eq!(r, i16x4::from(sse::_mm_max_pi16(a.into(), b.into())));
        assert_eq!(r, i16x4::from(sse::_m_pmaxsw(a.into(), b.into())));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_max_pu8() {
        let a = u8x8::new(2, 6, 3, 8, 2, 6, 3, 8);
        let b = u8x8::new(5, 2, 7, 4, 5, 2, 7, 4);
        let r = u8x8::new(5, 6, 7, 8, 5, 6, 7, 8);

        assert_eq!(r, u8x8::from(sse::_mm_max_pu8(a.into(), b.into())));
        assert_eq!(r, u8x8::from(sse::_m_pmaxub(a.into(), b.into())));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_min_pi16() {
        let a = i16x4::new(-1, 6, -3, 8);
        let b = i16x4::new(5, -2, 7, -4);
        let r = i16x4::new(-1, -2, -3, -4);

        assert_eq!(r, i16x4::from(sse::_mm_min_pi16(a.into(), b.into())));
        assert_eq!(r, i16x4::from(sse::_m_pminsw(a.into(), b.into())));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_min_pu8() {
        let a = u8x8::new(2, 6, 3, 8, 2, 6, 3, 8);
        let b = u8x8::new(5, 2, 7, 4, 5, 2, 7, 4);
        let r = u8x8::new(2, 2, 3, 4, 2, 2, 3, 4);

        assert_eq!(r, u8x8::from(sse::_mm_min_pu8(a.into(), b.into())));
        assert_eq!(r, u8x8::from(sse::_m_pminub(a.into(), b.into())));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_mulhi_pu16() {
        let (a, b) = (u16x4::splat(1000), u16x4::splat(1001));
        let r = u16x4::from(sse::_mm_mulhi_pu16(a.into(), b.into()));
        assert_eq!(r, u16x4::splat(15));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_avg_pu8() {
        let (a, b) = (u8x8::splat(3), u8x8::splat(9));
        let r = u8x8::from(sse::_mm_avg_pu8(a.into(), b.into()));
        assert_eq!(r, u8x8::splat(6));

        let r = u8x8::from(sse::_m_pavgb(a.into(), b.into()));
        assert_eq!(r, u8x8::splat(6));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_avg_pu16() {
        let (a, b) = (u16x4::splat(3), u16x4::splat(9));
        let r = u16x4::from(sse::_mm_avg_pu16(a.into(), b.into()));
        assert_eq!(r, u16x4::splat(6));

        let r = u16x4::from(sse::_m_pavgw(a.into(), b.into()));
        assert_eq!(r, u16x4::splat(6));
    }

    #[simd_test = "sse"]
    #[cfg(not(windows))] // FIXME "unknown codeview register" in LLVM
    unsafe fn _mm_sad_pu8() {
        let a = u8x8::new(255, 254, 253, 252, 1, 2, 3, 4);
        let b = u8x8::new(0, 0, 0, 0, 2, 1, 2, 1);
        let r = sse::_mm_sad_pu8(a.into(), b.into());
        assert_eq!(r, mem::transmute(u16x4::new(1020, 0, 0, 0)));

        let r = sse::_m_psadbw(a.into(), b.into());
        assert_eq!(r, mem::transmute(u16x4::new(1020, 0, 0, 0)));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_cvtpi32_ps() {
        let a = f32x4::new(0., 0., 3., 4.);
        let b = i32x2::new(1, 2);
        let expected = f32x4::new(1., 2., 3., 4.);
        let r = sse::_mm_cvtpi32_ps(a, b);
        assert_eq!(r, expected);

        let r = sse::_mm_cvt_pi2ps(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_cvtpi16_ps() {
        let a = i16x4::new(1, 2, 3, 4);
        let expected = f32x4::new(1., 2., 3., 4.);
        let r = sse::_mm_cvtpi16_ps(a.into());
        assert_eq!(r, expected);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_cvtpu16_ps() {
        let a = u16x4::new(1, 2, 3, 4);
        let expected = f32x4::new(1., 2., 3., 4.);
        let r = sse::_mm_cvtpu16_ps(a.into());
        assert_eq!(r, expected);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_cvtpi8_ps() {
        let a = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let expected = f32x4::new(1., 2., 3., 4.);
        let r = sse::_mm_cvtpi8_ps(a.into());
        assert_eq!(r, expected);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_cvtpu8_ps() {
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let expected = f32x4::new(1., 2., 3., 4.);
        let r = sse::_mm_cvtpu8_ps(a.into());
        assert_eq!(r, expected);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_cvtpi32x2_ps() {
        let a = i32x2::new(1, 2);
        let b = i32x2::new(3, 4);
        let expected = f32x4::new(1., 2., 3., 4.);
        let r = sse::_mm_cvtpi32x2_ps(a, b);
        assert_eq!(r, expected);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_maskmove_si64() {
        let a = i8x8::splat(9);
        let mask = i8x8::splat(0).replace(2, 0x80u8 as i8);
        let mut r = i8x8::splat(0);
        sse::_mm_maskmove_si64(a.into(), mask.into(), &mut r as *mut _ as *mut i8);
        assert_eq!(r, i8x8::splat(0).replace(2, 9));

        let mut r = i8x8::splat(0);
        sse::_m_maskmovq(a.into(), mask.into(), &mut r as *mut _ as *mut i8);
        assert_eq!(r, i8x8::splat(0).replace(2, 9));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_extract_pi16() {
        let a = i16x4::new(1, 2, 3, 4);
        let r = sse::_mm_extract_pi16(a, 0);
        assert_eq!(r, 1);
        let r = sse::_mm_extract_pi16(a, 1);
        assert_eq!(r, 2);

        let r = sse::_m_pextrw(a, 1);
        assert_eq!(r, 2);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_insert_pi16() {
        let a = i16x4::new(1, 2, 3, 4);
        let r = i16x4::from(sse::_mm_insert_pi16(a.into(), 0, 0b0));
        let expected = i16x4::new(0, 2, 3, 4);
        assert_eq!(r, expected);
        let r = i16x4::from(sse::_mm_insert_pi16(a.into(), 0, 0b10));
        let expected = i16x4::new(1, 2, 0, 4);
        assert_eq!(r, expected);

        let r = i16x4::from(sse::_m_pinsrw(a.into(), 0, 0b10));
        assert_eq!(r, expected);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_movemask_pi8() {
        let a = i16x4::new(0b1000_0000, 0b0100_0000, 0b1000_0000, 0b0100_0000);
        let r = sse::_mm_movemask_pi8(a);
        assert_eq!(r, 0b10001);

        let r = sse::_m_pmovmskb(a);
        assert_eq!(r, 0b10001);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_shuffle_pi16() {
        let a = i16x4::new(1, 2, 3, 4);
        let r = i16x4::from(sse::_mm_shuffle_pi16(a.into(), 0b00_01_01_11));
        let expected = i16x4::new(4, 2, 2, 1);
        assert_eq!(r, expected);

        let r = i16x4::from(sse::_m_pshufw(a.into(), 0b00_01_01_11));
        assert_eq!(r, expected);
    }

    #[simd_test = "sse"]
    unsafe fn _mm_cvtps_pi32() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let r = i32x2::new(1, 2);

        assert_eq!(r, i32x2::from(sse::_mm_cvtps_pi32(a)));
        assert_eq!(r, i32x2::from(sse::_mm_cvt_ps2pi(a)));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_cvttps_pi32() {
        let a = f32x4::new(7.0, 2.0, 3.0, 4.0);
        let r = i32x2::new(7, 2);

        assert_eq!(r, sse::_mm_cvttps_pi32(a));
        assert_eq!(r, sse::_mm_cvtt_ps2pi(a));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_cvtps_pi16() {
        let a = f32x4::new(7.0, 2.0, 3.0, 4.0);
        let r = i16x4::new(7, 2, 3, 4);
        assert_eq!(r, i16x4::from(sse::_mm_cvtps_pi16(a)));
    }

    #[simd_test = "sse"]
    unsafe fn _mm_cvtps_pi8() {
        let a = f32x4::new(7.0, 2.0, 3.0, 4.0);
        let r = i8x8::new(7, 2, 3, 4, 0, 0, 0, 0);
        assert_eq!(r, i8x8::from(sse::_mm_cvtps_pi8(a)));
    }
}
