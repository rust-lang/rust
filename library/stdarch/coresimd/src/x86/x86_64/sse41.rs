use v128::*;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Extract an 64-bit integer from `a` selected with `imm8`
#[inline(always)]
#[target_feature = "+sse4.1"]
// TODO: Add test for Windows
#[cfg_attr(all(test, not(windows)), assert_instr(pextrq, imm8 = 1))]
pub unsafe fn _mm_extract_epi64(a: i64x2, imm8: i32) -> i64 {
    let imm8 = (imm8 & 1) as u32;
    a.extract_unchecked(imm8)
}

/// Return a copy of `a` with the 64-bit integer from `i` inserted at a
/// location specified by `imm8`.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pinsrq, imm8 = 0))]
pub unsafe fn _mm_insert_epi64(a: i64x2, i: i64, imm8: i32) -> i64x2 {
    a.replace((imm8 & 0b1) as u32, i)
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;
    use x86::x86_64::sse41;
    use v128::*;

    #[simd_test = "sse4.1"]
    unsafe fn _mm_extract_epi64() {
        let a = i64x2::new(0, 1);
        let r = sse41::_mm_extract_epi64(a, 1);
        assert_eq!(r, 1);
        let r = sse41::_mm_extract_epi64(a, 3);
        assert_eq!(r, 1);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_insert_epi64() {
        let a = i64x2::splat(0);
        let e = i64x2::splat(0).replace(1, 32);
        let r = sse41::_mm_insert_epi64(a, 32, 1);
        assert_eq!(r, e);
        let r = sse41::_mm_insert_epi64(a, 32, 3);
        assert_eq!(r, e);
    }
}
