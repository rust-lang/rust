//! `i686`'s Streaming SIMD Extensions 4.2 (SSE4.2)

use simd_llvm::*;
use v128::*;
use x86::*;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Compare packed 64-bit integers in `a` and `b` for greater-than,
/// return the results.
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpgtq))]
pub unsafe fn _mm_cmpgt_epi64(a: __m128i, b: __m128i) -> __m128i {
    mem::transmute(simd_gt::<_, i64x2>(a.as_i64x2(), b.as_i64x2()))
}

#[cfg(test)]
mod tests {
    use x86::*;

    use stdsimd_test::simd_test;

    #[simd_test = "sse4.2"]
    unsafe fn test_mm_cmpgt_epi64() {
        let a = _mm_setr_epi64x(0, 0x2a);
        let b = _mm_set1_epi64x(0x00);
        let i = _mm_cmpgt_epi64(a, b);
        assert_eq_m128i(i, _mm_setr_epi64x(0x00, 0xffffffffffffffffu64 as i64));
    }
}
