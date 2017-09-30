use x86::__m128i;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Load 128-bits of integer data from unaligned memory.
/// This intrinsic may perform better than `_mm_loadu_si128`
/// when the data crosses a cache line boundary.
#[inline(always)]
#[target_feature = "+sse3"]
#[cfg_attr(test, assert_instr(lddqu))]
pub unsafe fn _mm_lddqu_si128(mem_addr: *const __m128i) -> __m128i {
    lddqu(mem_addr as *const _)
}

#[allow(improper_ctypes)]
extern {
    #[link_name = "llvm.x86.sse3.ldu.dq"]
    fn lddqu(mem_addr: *const i8) -> __m128i;
}


#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use v128::*;
    use x86::sse3 as sse3;

    #[simd_test = "sse3"]
    unsafe fn _mm_lddqu_si128() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = sse3::_mm_lddqu_si128(&a);
        assert_eq!(a, r);
    }
}