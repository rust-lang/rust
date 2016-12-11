use v128::*;

/// Compute the absolute value of packed 8-bit signed integers in `a` and
/// return the unsigned results.
#[inline(always)]
#[target_feature = "+ssse3"]
pub fn _mm_abs_epi8(a: i8x16) -> u8x16 {
    unsafe { pabsb128(a) }
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
    #[target_feature = "+ssse3"]
    fn _mm_abs_epi8() {
        let r = ssse3::_mm_abs_epi8(i8x16::splat(-5));
        assert_eq!(r, u8x16::splat(5));
    }
}
