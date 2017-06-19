// use v128::*;
use x86::__m128i;

#[inline(always)]
#[target_feature = "+sse4.1"]
pub fn _mm_blendv_epi8(
    a: __m128i,
    b: __m128i,
    mask: __m128i,
) -> __m128i {
    unsafe { pblendvb(a, b, mask) }
}

#[allow(improper_ctypes)]
extern {
    #[link_name = "llvm.x86.sse41.pblendvb"]
    fn pblendvb(a: __m128i, b: __m128i, mask: __m128i) -> __m128i;
}

#[cfg(test)]
mod tests {
    use v128::*;
    use x86::sse41;

    #[test]
    #[target_feature = "+sse4.2"]
    fn _mm_blendv_epi8() {
        let a = i8x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = i8x16::new(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let mask = i8x16::new(
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1);
        let e = i8x16::new(
            0, 17, 2, 19, 4, 21, 6, 23, 8, 25, 10, 27, 12, 29, 14, 31);
        assert_eq!(sse41::_mm_blendv_epi8(a, b, mask), e);
    }
}
