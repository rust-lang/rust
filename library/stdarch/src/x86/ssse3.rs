use v128::*;

/// Compute the absolute value of packed 8-bit signed integers in `a` and
/// return the unsigned results.
#[inline]
#[target_feature = "+ssse3"]
pub unsafe fn _mm_abs_epi8(a: __m128i) -> __m128i {
    pabsb128(i8x16::from(a)).as_m128i()
}

#[allow(improper_ctypes)]
extern {
    #[link_name = "llvm.x86.ssse3.pabs.b.128"]
    pub fn pabsb128(a: i8x16) -> u8x16;
}

#[cfg(test)]
mod tests {
    use v128::*;
    use x86::ssse3 as ssse3;

    #[test]
    fn _mm_abs_epi8() {
        unsafe {
            // let a = sse2::_mm_set1_epi8(-5);
            let a = i8x16::splat(-5);
            let r = ssse3::_mm_abs_epi8(a.as_m128i());
            assert_eq!(u8x16::from(r), u8x16::splat(5));
        }
    }
}
