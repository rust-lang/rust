//! `i686`'s Streaming SIMD Extensions 4.1 (SSE4.1)

use core_arch::simd_llvm::*;
use core_arch::x86::*;
use mem;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Extract an 64-bit integer from `a` selected with `imm8`
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_extract_epi64)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(all(test, not(target_os = "windows")), assert_instr(pextrq, imm8 = 1))]
#[rustc_args_required_const(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_extract_epi64(a: __m128i, imm8: i32) -> i64 {
    let imm8 = (imm8 & 1) as u32;
    simd_extract(a.as_i64x2(), imm8)
}

/// Return a copy of `a` with the 64-bit integer from `i` inserted at a
/// location specified by `imm8`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_insert_epi64)
#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(test, assert_instr(pinsrq, imm8 = 0))]
#[rustc_args_required_const(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_insert_epi64(a: __m128i, i: i64, imm8: i32) -> __m128i {
    mem::transmute(simd_insert(a.as_i64x2(), (imm8 & 1) as u32, i))
}

#[cfg(test)]
mod tests {
    use core_arch::arch::x86_64::*;
    use stdsimd_test::simd_test;

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_extract_epi64() {
        let a = _mm_setr_epi64x(0, 1);
        let r = _mm_extract_epi64(a, 1);
        assert_eq!(r, 1);
        let r = _mm_extract_epi64(a, 3);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "sse4.1")]
    unsafe fn test_mm_insert_epi64() {
        let a = _mm_set1_epi64x(0);
        let e = _mm_setr_epi64x(0, 32);
        let r = _mm_insert_epi64(a, 32, 1);
        assert_eq_m128i(r, e);
        let r = _mm_insert_epi64(a, 32, 3);
        assert_eq_m128i(r, e);
    }
}
