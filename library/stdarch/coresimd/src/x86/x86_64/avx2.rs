use simd_llvm::*;
use x86::*;

/// Extract a 64-bit integer from `a`, selected with `imm8`.
#[inline(always)]
#[target_feature(enable = "avx2")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm256_extract_epi64(a: __m256i, imm8: i32) -> i64 {
    let imm8 = (imm8 & 3) as u32;
    simd_extract(a.as_i64x4(), imm8)
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use x86::*;

    #[simd_test = "avx2"]
    unsafe fn test_mm256_extract_epi64() {
        let a = _mm256_setr_epi64x(0, 1, 2, 3);
        let r = _mm256_extract_epi64(a, 3);
        assert_eq!(r, 3);
    }
}
