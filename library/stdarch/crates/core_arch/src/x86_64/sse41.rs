//! `i686`'s Streaming SIMD Extensions 4.1 (SSE4.1)

use crate::{
    core_arch::{simd_llvm::*, x86::*},
    mem::transmute,
};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Extracts an 64-bit integer from `a` selected with `IMM1`
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_extract_epi64)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(all(test, not(target_os = "windows")), assert_instr(pextrq, IMM1 = 1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_extract_epi64<const IMM1: i32>(a: __m128i) -> i64 {
    static_assert_imm1!(IMM1);
    simd_extract(a.as_i64x2(), IMM1 as u32)
}

/// Returns a copy of `a` with the 64-bit integer from `i` inserted at a
/// location specified by `IMM1`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_insert_epi64)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pinsrq, IMM1 = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_insert_epi64<const IMM1: i32>(a: __m128i, i: i64) -> __m128i {
    static_assert_imm1!(IMM1);
    transmute(simd_insert(a.as_i64x2(), IMM1 as u32, i))
}

#[cfg(test)]
mod tests {
    use crate::core_arch::arch::x86_64::*;
    use stdarch_test::simd_test;

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_extract_epi64() {
        let a = _mm_setr_epi64x(0, 1);
        let r = _mm_extract_epi64::<1>(a);
        assert_eq!(r, 1);
        let r = _mm_extract_epi64::<0>(a);
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_insert_epi64() {
        let a = _mm_set1_epi64x(0);
        let e = _mm_setr_epi64x(0, 32);
        let r = _mm_insert_epi64::<1>(a, 32);
        assert_eq_m128i(r, e);
        let e = _mm_setr_epi64x(32, 0);
        let r = _mm_insert_epi64::<0>(a, 32);
        assert_eq_m128i(r, e);
    }
}
