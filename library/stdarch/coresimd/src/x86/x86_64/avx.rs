use core::mem;

use simd_llvm::*;
use x86::*;

/// Copy `a` to result, and insert the 64-bit integer `i` into result
/// at the location specified by `index`.
#[inline(always)]
#[target_feature(enable = "avx")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm256_insert_epi64(a: __m256i, i: i64, index: i32) -> __m256i {
    mem::transmute(simd_insert(a.as_i64x4(), (index as u32) & 3, i))
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use x86::*;

    #[simd_test = "avx"]
    unsafe fn test_mm256_insert_epi64() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let r = _mm256_insert_epi64(a, 0, 3);
        let e = _mm256_setr_epi64x(1, 2, 3, 0);
        assert_eq_m256i(r, e);
    }
}
