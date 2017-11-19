//! `i686`'s Streaming SIMD Extensions 2 (SSE2)

use v128::*;

#[cfg(all(test, not(target_arch = "x86")))]
use stdsimd_test::assert_instr;

/// Return `a` with its lower element replaced by `b` after converting it to
/// an `f64`.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(all(test, not(target_arch = "x86")), assert_instr(cvtsi2sd))]
pub unsafe fn _mm_cvtsi64_sd(a: f64x2, b: i64) -> f64x2 {
    a.replace(0, b as f64)
}

/// Return `a` with its lower element replaced by `b` after converting it to
/// an `f64`.
#[inline(always)]
#[target_feature = "+sse2"]
#[cfg_attr(all(test, not(target_arch = "x86")), assert_instr(cvtsi2sd))]
pub unsafe fn _mm_cvtsi64x_sd(a: f64x2, b: i64) -> f64x2 {
    _mm_cvtsi64_sd(a, b)
}

/// Return a vector whose lowest element is `a` and all higher elements are
/// `0`.
#[inline(always)]
#[target_feature = "+sse2"]
// no particular instruction to test
pub unsafe fn _mm_cvtsi64_si128(a: i64) -> i64x2 {
    i64x2::new(a, 0)
}

/// Return a vector whose lowest element is `a` and all higher elements are
/// `0`.
#[inline(always)]
#[target_feature = "+sse2"]
// no particular instruction to test
pub unsafe fn _mm_cvtsi64x_si128(a: i64) -> i64x2 {
    _mm_cvtsi64_si128(a)
}

/// Return the lowest element of `a`.
#[inline(always)]
#[target_feature = "+sse2"]
// no particular instruction to test
pub unsafe fn _mm_cvtsi128_si64(a: i64x2) -> i64 {
    a.extract(0)
}

/// Return the lowest element of `a`.
#[inline(always)]
#[target_feature = "+sse2"]
// no particular instruction to test
pub unsafe fn _mm_cvtsi128_si64x(a: i64x2) -> i64 {
    _mm_cvtsi128_si64(a)
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use v128::*;
    use x86::i686::sse2;

    #[simd_test = "sse2"]
    unsafe fn _mm_cvtsi64_sd() {
        let a = f64x2::splat(3.5);
        let r = sse2::_mm_cvtsi64_sd(a, 5);
        assert_eq!(r, f64x2::new(5.0, 3.5));
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_cvtsi64_si128() {
        let r = sse2::_mm_cvtsi64_si128(5);
        assert_eq!(r, i64x2::new(5, 0));
    }

    #[simd_test = "sse2"]
    unsafe fn _mm_cvtsi128_si64() {
        let r = sse2::_mm_cvtsi128_si64(i64x2::new(5, 0));
        assert_eq!(r, 5);
    }
}
