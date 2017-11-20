//! `i686`'s Streaming SIMD Extensions 4.2 (SSE4.2)

use v128::*;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Compare packed 64-bit integers in `a` and `b` for greater-than,
/// return the results.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpgtq))]
pub unsafe fn _mm_cmpgt_epi64(a: i64x2, b: i64x2) -> i64x2 {
    a.gt(b)
}

#[cfg(test)]
mod tests {
    use v128::*;
    use x86::i686::sse42;

    use stdsimd_test::simd_test;

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpgt_epi64() {
        let a = i64x2::splat(0x00).replace(1, 0x2a);
        let b = i64x2::splat(0x00);
        let i = sse42::_mm_cmpgt_epi64(a, b);
        assert_eq!(i, i64x2::new(0x00, 0xffffffffffffffffu64 as i64));
    }
}
