//! `i686` Streaming SIMD Extensions (SSE)

use x86::*;

#[cfg(test)]
use stdsimd_test::assert_instr;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.sse.cvtpi2ps"]
    fn cvtpi2ps(a: __m128, b: __m64) -> __m128;
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
    fn cvtps2pi(a: __m128) -> __m64;
    #[link_name = "llvm.x86.sse.cvttps2pi"]
    fn cvttps2pi(a: __m128) -> __m64;
}

/// Compares the packed 16-bit signed integers of `a` and `b` writing the
/// greatest value into the result.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pmaxsw))]
pub unsafe fn _mm_max_pi16(a: __m64, b: __m64) -> __m64 {
    pmaxsw(a, b)
}

/// Compares the packed 16-bit signed integers of `a` and `b` writing the
/// greatest value into the result.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pmaxsw))]
pub unsafe fn _m_pmaxsw(a: __m64, b: __m64) -> __m64 {
    _mm_max_pi16(a, b)
}

/// Compares the packed 8-bit signed integers of `a` and `b` writing the
/// greatest value into the result.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pmaxub))]
pub unsafe fn _mm_max_pu8(a: __m64, b: __m64) -> __m64 {
    pmaxub(a, b)
}

/// Compares the packed 8-bit signed integers of `a` and `b` writing the
/// greatest value into the result.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pmaxub))]
pub unsafe fn _m_pmaxub(a: __m64, b: __m64) -> __m64 {
    _mm_max_pu8(a, b)
}

/// Compares the packed 16-bit signed integers of `a` and `b` writing the
/// smallest value into the result.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pminsw))]
pub unsafe fn _mm_min_pi16(a: __m64, b: __m64) -> __m64 {
    pminsw(a, b)
}

/// Compares the packed 16-bit signed integers of `a` and `b` writing the
/// smallest value into the result.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pminsw))]
pub unsafe fn _m_pminsw(a: __m64, b: __m64) -> __m64 {
    _mm_min_pi16(a, b)
}

/// Compares the packed 8-bit signed integers of `a` and `b` writing the
/// smallest value into the result.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pminub))]
pub unsafe fn _mm_min_pu8(a: __m64, b: __m64) -> __m64 {
    pminub(a, b)
}

/// Compares the packed 8-bit signed integers of `a` and `b` writing the
/// smallest value into the result.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pminub))]
pub unsafe fn _m_pminub(a: __m64, b: __m64) -> __m64 {
    _mm_min_pu8(a, b)
}

/// Multiplies packed 16-bit unsigned integer values and writes the
/// high-order 16 bits of each 32-bit product to the corresponding bits in
/// the destination.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pmulhuw))]
pub unsafe fn _mm_mulhi_pu16(a: __m64, b: __m64) -> __m64 {
    pmulhuw(a, b)
}

/// Multiplies packed 16-bit unsigned integer values and writes the
/// high-order 16 bits of each 32-bit product to the corresponding bits in
/// the destination.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pmulhuw))]
pub unsafe fn _m_pmulhuw(a: __m64, b: __m64) -> __m64 {
    _mm_mulhi_pu16(a, b)
}

/// Computes the rounded averages of the packed unsigned 8-bit integer
/// values and writes the averages to the corresponding bits in the
/// destination.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pavgb))]
pub unsafe fn _mm_avg_pu8(a: __m64, b: __m64) -> __m64 {
    pavgb(a, b)
}

/// Computes the rounded averages of the packed unsigned 8-bit integer
/// values and writes the averages to the corresponding bits in the
/// destination.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pavgb))]
pub unsafe fn _m_pavgb(a: __m64, b: __m64) -> __m64 {
    _mm_avg_pu8(a, b)
}

/// Computes the rounded averages of the packed unsigned 16-bit integer
/// values and writes the averages to the corresponding bits in the
/// destination.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pavgw))]
pub unsafe fn _mm_avg_pu16(a: __m64, b: __m64) -> __m64 {
    pavgw(a, b)
}

/// Computes the rounded averages of the packed unsigned 16-bit integer
/// values and writes the averages to the corresponding bits in the
/// destination.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pavgw))]
pub unsafe fn _m_pavgw(a: __m64, b: __m64) -> __m64 {
    _mm_avg_pu16(a, b)
}

/// Subtracts the corresponding 8-bit unsigned integer values of the two
/// 64-bit vector operands and computes the absolute value for each of the
/// difference. Then sum of the 8 absolute differences is written to the
/// bits [15:0] of the destination; the remaining bits [63:16] are cleared.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(psadbw))]
pub unsafe fn _mm_sad_pu8(a: __m64, b: __m64) -> __m64 {
    psadbw(a, b)
}

/// Subtracts the corresponding 8-bit unsigned integer values of the two
/// 64-bit vector operands and computes the absolute value for each of the
/// difference. Then sum of the 8 absolute differences is written to the
/// bits [15:0] of the destination; the remaining bits [63:16] are cleared.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(psadbw))]
pub unsafe fn _m_psadbw(a: __m64, b: __m64) -> __m64 {
    _mm_sad_pu8(a, b)
}

/// Converts two elements of a 64-bit vector of [2 x i32] into two
/// floating point values and writes them to the lower 64-bits of the
/// destination. The remaining higher order elements of the destination are
/// copied from the corresponding elements in the first operand.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(cvtpi2ps))]
pub unsafe fn _mm_cvtpi32_ps(a: __m128, b: __m64) -> __m128 {
    cvtpi2ps(a, b)
}

/// Converts two elements of a 64-bit vector of [2 x i32] into two
/// floating point values and writes them to the lower 64-bits of the
/// destination. The remaining higher order elements of the destination are
/// copied from the corresponding elements in the first operand.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(cvtpi2ps))]
pub unsafe fn _mm_cvt_pi2ps(a: __m128, b: __m64) -> __m128 {
    _mm_cvtpi32_ps(a, b)
}

/// Converts the lower 4 8-bit values of `a` into a 128-bit vector of 4 `f32`s.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(cvtpi2ps))]
pub unsafe fn _mm_cvtpi8_ps(a: __m64) -> __m128 {
    let b = _mm_setzero_si64();
    let b = _mm_cmpgt_pi8(b, a);
    let b = _mm_unpacklo_pi8(a, b);
    _mm_cvtpi16_ps(b)
}

/// Converts the lower 4 8-bit values of `a` into a 128-bit vector of 4 `f32`s.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(cvtpi2ps))]
pub unsafe fn _mm_cvtpu8_ps(a: __m64) -> __m128 {
    let b = _mm_setzero_si64();
    let b = _mm_unpacklo_pi8(a, b);
    _mm_cvtpi16_ps(b)
}

/// Converts a 64-bit vector of `i16`s into a 128-bit vector of 4 `f32`s.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(cvtpi2ps))]
pub unsafe fn _mm_cvtpi16_ps(a: __m64) -> __m128 {
    let b = _mm_setzero_si64();
    let b = _mm_cmpgt_pi16(b, a);
    let c = _mm_unpackhi_pi16(a, b);
    let r = _mm_setzero_ps();
    let r = cvtpi2ps(r, c);
    let r = _mm_movelh_ps(r, r);
    let c = _mm_unpacklo_pi16(a, b);
    cvtpi2ps(r, c)
}

/// Converts a 64-bit vector of `i16`s into a 128-bit vector of 4 `f32`s.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(cvtpi2ps))]
pub unsafe fn _mm_cvtpu16_ps(a: __m64) -> __m128 {
    let b = _mm_setzero_si64();
    let c = _mm_unpackhi_pi16(a, b);
    let r = _mm_setzero_ps();
    let r = cvtpi2ps(r, c);
    let r = _mm_movelh_ps(r, r);
    let c = _mm_unpacklo_pi16(a, b);
    cvtpi2ps(r, c)
}

/// Converts the two 32-bit signed integer values from each 64-bit vector
/// operand of [2 x i32] into a 128-bit vector of [4 x float].
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(cvtpi2ps))]
pub unsafe fn _mm_cvtpi32x2_ps(a: __m64, b: __m64) -> __m128 {
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
#[inline]
#[target_feature(enable = "sse,mmx")]
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
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(maskmovq))]
pub unsafe fn _m_maskmovq(a: __m64, mask: __m64, mem_addr: *mut i8) {
    _mm_maskmove_si64(a, mask, mem_addr)
}

/// Extracts 16-bit element from a 64-bit vector of [4 x i16] and
/// returns it, as specified by the immediate integer operand.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pextrw, imm2 = 0))]
pub unsafe fn _mm_extract_pi16(a: __m64, imm2: i32) -> i16 {
    macro_rules! call {
        ($imm2:expr) => { pextrw(a, $imm2) as i16 }
    }
    constify_imm2!(imm2, call)
}

/// Extracts 16-bit element from a 64-bit vector of [4 x i16] and
/// returns it, as specified by the immediate integer operand.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pextrw, imm2 = 0))]
pub unsafe fn _m_pextrw(a: __m64, imm2: i32) -> i16 {
    _mm_extract_pi16(a, imm2)
}

/// Copies data from the 64-bit vector of [4 x i16] to the destination,
/// and inserts the lower 16-bits of an integer operand at the 16-bit offset
/// specified by the immediate operand `n`.
#[inline]
#[target_feature(enable = "sse,mmx")]
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
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pinsrw, imm2 = 0))]
pub unsafe fn _m_pinsrw(a: __m64, d: i32, imm2: i32) -> __m64 {
    _mm_insert_pi16(a, d, imm2)
}

/// Takes the most significant bit from each 8-bit element in a 64-bit
/// integer vector to create a 16-bit mask value. Zero-extends the value to
/// 32-bit integer and writes it to the destination.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pmovmskb))]
pub unsafe fn _mm_movemask_pi8(a: __m64) -> i32 {
    pmovmskb(a)
}

/// Takes the most significant bit from each 8-bit element in a 64-bit
/// integer vector to create a 16-bit mask value. Zero-extends the value to
/// 32-bit integer and writes it to the destination.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pmovmskb))]
pub unsafe fn _m_pmovmskb(a: __m64) -> i32 {
    _mm_movemask_pi8(a)
}

/// Shuffles the 4 16-bit integers from a 64-bit integer vector to the
/// destination, as specified by the immediate value operand.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pshufw, imm8 = 0))]
pub unsafe fn _mm_shuffle_pi16(a: __m64, imm8: i32) -> __m64 {
    macro_rules! call {
        ($imm8:expr) => { pshufw(a, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Shuffles the 4 16-bit integers from a 64-bit integer vector to the
/// destination, as specified by the immediate value operand.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(pshufw, imm8 = 0))]
pub unsafe fn _m_pshufw(a: __m64, imm8: i32) -> __m64 {
    _mm_shuffle_pi16(a, imm8)
}

/// Convert the two lower packed single-precision (32-bit) floating-point
/// elements in `a` to packed 32-bit integers with truncation.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(cvttps2pi))]
pub unsafe fn _mm_cvttps_pi32(a: __m128) -> __m64 {
    cvttps2pi(a)
}

/// Convert the two lower packed single-precision (32-bit) floating-point
/// elements in `a` to packed 32-bit integers with truncation.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(cvttps2pi))]
pub unsafe fn _mm_cvtt_ps2pi(a: __m128) -> __m64 {
    _mm_cvttps_pi32(a)
}

/// Convert the two lower packed single-precision (32-bit) floating-point
/// elements in `a` to packed 32-bit integers.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(cvtps2pi))]
pub unsafe fn _mm_cvtps_pi32(a: __m128) -> __m64 {
    cvtps2pi(a)
}

/// Convert the two lower packed single-precision (32-bit) floating-point
/// elements in `a` to packed 32-bit integers.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(cvtps2pi))]
pub unsafe fn _mm_cvt_ps2pi(a: __m128) -> __m64 {
    _mm_cvtps_pi32(a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to
/// packed 16-bit integers.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(cvtps2pi))]
pub unsafe fn _mm_cvtps_pi16(a: __m128) -> __m64 {
    let b = _mm_cvtps_pi32(a);
    let a = _mm_movehl_ps(a, a);
    let c = _mm_cvtps_pi32(a);
    _mm_packs_pi32(b, c)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to
/// packed 8-bit integers, and returns theem in the lower 4 elements of the
/// result.
#[inline]
#[target_feature(enable = "sse,mmx")]
#[cfg_attr(test, assert_instr(cvtps2pi))]
pub unsafe fn _mm_cvtps_pi8(a: __m128) -> __m64 {
    let b = _mm_cvtps_pi16(a);
    let c = _mm_setzero_si64();
    _mm_packs_pi16(b, c)
}

#[cfg(test)]
mod tests {
    use x86::*;
    use stdsimd_test::simd_test;

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_max_pi16() {
        let a = _mm_setr_pi16(-1, 6, -3, 8);
        let b = _mm_setr_pi16(5, -2, 7, -4);
        let r = _mm_setr_pi16(5, 6, 7, 8);

        assert_eq_m64(r, _mm_max_pi16(a, b));
        assert_eq_m64(r, _m_pmaxsw(a, b));
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_max_pu8() {
        let a = _mm_setr_pi8(2, 6, 3, 8, 2, 6, 3, 8);
        let b = _mm_setr_pi8(5, 2, 7, 4, 5, 2, 7, 4);
        let r = _mm_setr_pi8(5, 6, 7, 8, 5, 6, 7, 8);

        assert_eq_m64(r, _mm_max_pu8(a, b));
        assert_eq_m64(r, _m_pmaxub(a, b));
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_min_pi16() {
        let a = _mm_setr_pi16(-1, 6, -3, 8);
        let b = _mm_setr_pi16(5, -2, 7, -4);
        let r = _mm_setr_pi16(-1, -2, -3, -4);

        assert_eq_m64(r, _mm_min_pi16(a, b));
        assert_eq_m64(r, _m_pminsw(a, b));
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_min_pu8() {
        let a = _mm_setr_pi8(2, 6, 3, 8, 2, 6, 3, 8);
        let b = _mm_setr_pi8(5, 2, 7, 4, 5, 2, 7, 4);
        let r = _mm_setr_pi8(2, 2, 3, 4, 2, 2, 3, 4);

        assert_eq_m64(r, _mm_min_pu8(a, b));
        assert_eq_m64(r, _m_pminub(a, b));
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_mulhi_pu16() {
        let (a, b) = (_mm_set1_pi16(1000), _mm_set1_pi16(1001));
        let r = _mm_mulhi_pu16(a, b);
        assert_eq_m64(r, _mm_set1_pi16(15));
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_m_pmulhuw() {
        let (a, b) = (_mm_set1_pi16(1000), _mm_set1_pi16(1001));
        let r = _m_pmulhuw(a, b);
        assert_eq_m64(r, _mm_set1_pi16(15));
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_avg_pu8() {
        let (a, b) = (_mm_set1_pi8(3), _mm_set1_pi8(9));
        let r = _mm_avg_pu8(a, b);
        assert_eq_m64(r, _mm_set1_pi8(6));

        let r = _m_pavgb(a, b);
        assert_eq_m64(r, _mm_set1_pi8(6));
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_avg_pu16() {
        let (a, b) = (_mm_set1_pi16(3), _mm_set1_pi16(9));
        let r = _mm_avg_pu16(a, b);
        assert_eq_m64(r, _mm_set1_pi16(6));

        let r = _m_pavgw(a, b);
        assert_eq_m64(r, _mm_set1_pi16(6));
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_sad_pu8() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = _mm_setr_pi8(
            255u8 as i8, 254u8 as i8, 253u8 as i8, 252u8 as i8,
            1, 2, 3, 4,
        );
        let b = _mm_setr_pi8(0, 0, 0, 0, 2, 1, 2, 1);
        let r = _mm_sad_pu8(a, b);
        assert_eq_m64(r, _mm_setr_pi16(1020, 0, 0, 0));

        let r = _m_psadbw(a, b);
        assert_eq_m64(r, _mm_setr_pi16(1020, 0, 0, 0));
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_cvtpi32_ps() {
        let a = _mm_setr_ps(0., 0., 3., 4.);
        let b = _mm_setr_pi32(1, 2);
        let expected = _mm_setr_ps(1., 2., 3., 4.);
        let r = _mm_cvtpi32_ps(a, b);
        assert_eq_m128(r, expected);

        let r = _mm_cvt_pi2ps(a, b);
        assert_eq_m128(r, expected);
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_cvtpi16_ps() {
        let a = _mm_setr_pi16(1, 2, 3, 4);
        let expected = _mm_setr_ps(1., 2., 3., 4.);
        let r = _mm_cvtpi16_ps(a);
        assert_eq_m128(r, expected);
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_cvtpu16_ps() {
        let a = _mm_setr_pi16(1, 2, 3, 4);
        let expected = _mm_setr_ps(1., 2., 3., 4.);
        let r = _mm_cvtpu16_ps(a);
        assert_eq_m128(r, expected);
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_cvtpi8_ps() {
        let a = _mm_setr_pi8(1, 2, 3, 4, 5, 6, 7, 8);
        let expected = _mm_setr_ps(1., 2., 3., 4.);
        let r = _mm_cvtpi8_ps(a);
        assert_eq_m128(r, expected);
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_cvtpu8_ps() {
        let a = _mm_setr_pi8(1, 2, 3, 4, 5, 6, 7, 8);
        let expected = _mm_setr_ps(1., 2., 3., 4.);
        let r = _mm_cvtpu8_ps(a);
        assert_eq_m128(r, expected);
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_cvtpi32x2_ps() {
        let a = _mm_setr_pi32(1, 2);
        let b = _mm_setr_pi32(3, 4);
        let expected = _mm_setr_ps(1., 2., 3., 4.);
        let r = _mm_cvtpi32x2_ps(a, b);
        assert_eq_m128(r, expected);
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_maskmove_si64() {
        let a = _mm_set1_pi8(9);
        let mask = _mm_setr_pi8(0, 0, 0x80u8 as i8, 0, 0, 0, 0, 0);
        let mut r = _mm_set1_pi8(0);
        _mm_maskmove_si64(a, mask, &mut r as *mut _ as *mut i8);
        let e = _mm_setr_pi8(0, 0, 9, 0, 0, 0, 0, 0);
        assert_eq_m64(r, e);

        let mut r = _mm_set1_pi8(0);
        _m_maskmovq(a, mask, &mut r as *mut _ as *mut i8);
        assert_eq_m64(r, e);
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_extract_pi16() {
        let a = _mm_setr_pi16(1, 2, 3, 4);
        let r = _mm_extract_pi16(a, 0);
        assert_eq!(r, 1);
        let r = _mm_extract_pi16(a, 1);
        assert_eq!(r, 2);

        let r = _m_pextrw(a, 1);
        assert_eq!(r, 2);
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_insert_pi16() {
        let a = _mm_setr_pi16(1, 2, 3, 4);
        let r = _mm_insert_pi16(a, 0, 0b0);
        let expected = _mm_setr_pi16(0, 2, 3, 4);
        assert_eq_m64(r, expected);
        let r = _mm_insert_pi16(a, 0, 0b10);
        let expected = _mm_setr_pi16(1, 2, 0, 4);
        assert_eq_m64(r, expected);

        let r = _m_pinsrw(a, 0, 0b10);
        assert_eq_m64(r, expected);
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_movemask_pi8() {
        let a =
            _mm_setr_pi16(0b1000_0000, 0b0100_0000, 0b1000_0000, 0b0100_0000);
        let r = _mm_movemask_pi8(a);
        assert_eq!(r, 0b10001);

        let r = _m_pmovmskb(a);
        assert_eq!(r, 0b10001);
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_shuffle_pi16() {
        let a = _mm_setr_pi16(1, 2, 3, 4);
        let r = _mm_shuffle_pi16(a, 0b00_01_01_11);
        let expected = _mm_setr_pi16(4, 2, 2, 1);
        assert_eq_m64(r, expected);

        let r = _m_pshufw(a, 0b00_01_01_11);
        assert_eq_m64(r, expected);
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_cvtps_pi32() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let r = _mm_setr_pi32(1, 2);

        assert_eq_m64(r, _mm_cvtps_pi32(a));
        assert_eq_m64(r, _mm_cvt_ps2pi(a));
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_cvttps_pi32() {
        let a = _mm_setr_ps(7.0, 2.0, 3.0, 4.0);
        let r = _mm_setr_pi32(7, 2);

        assert_eq_m64(r, _mm_cvttps_pi32(a));
        assert_eq_m64(r, _mm_cvtt_ps2pi(a));
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_cvtps_pi16() {
        let a = _mm_setr_ps(7.0, 2.0, 3.0, 4.0);
        let r = _mm_setr_pi16(7, 2, 3, 4);
        assert_eq_m64(r, _mm_cvtps_pi16(a));
    }

    #[simd_test = "sse,mmx"]
    unsafe fn test_mm_cvtps_pi8() {
        let a = _mm_setr_ps(7.0, 2.0, 3.0, 4.0);
        let r = _mm_setr_pi8(7, 2, 3, 4, 0, 0, 0, 0);
        assert_eq_m64(r, _mm_cvtps_pi8(a));
    }
}
