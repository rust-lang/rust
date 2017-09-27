#[cfg(test)]
use stdsimd_test::assert_instr;

use v128::*;

/// Compute the absolute value of packed 8-bit signed integers in `a` and
/// return the unsigned results.
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(pabsb))]
pub unsafe fn _mm_abs_epi8(a: i8x16) -> u8x16 {
    pabsb128(a)
}

/// Shuffle bytes from `a` according to the content of `b`.
///
/// The last 4 bits of each byte of `b` are used as addresses
/// into the 16 bytes of `a`.
///
/// In addition, if the highest significant bit of a byte of `b`
/// is set, the respective destination byte is set to 0.
///
/// Picturing `a` and `b` as `[u8; 16]`, `_mm_shuffle_epi8` is
/// logically equivalent to:
///
/// ```
/// fn mm_shuffle_epi8(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
///     let mut r = [0u8; 16];
///     for i in 0..16 {
///         // if the most significant bit of b is set,
///         // then the destination byte is set to 0.
///         if b[i] & 0x80 == 0u8 {
///             r[i] = a[(b[i] % 16) as usize];
///         }
///     }
///     r
/// }
/// ```
#[inline(always)]
#[target_feature = "+ssse3"]
#[cfg_attr(test, assert_instr(pshufb))]
pub unsafe fn _mm_shuffle_epi8(a: u8x16, b: u8x16) -> u8x16 {
    pshufb128(a, b)
}


#[allow(improper_ctypes)]
extern {
    #[link_name = "llvm.x86.ssse3.pabs.b.128"]
    fn pabsb128(a: i8x16) -> u8x16;
    #[link_name = "llvm.x86.ssse3.pshuf.b.128"]
    fn pshufb128(a: u8x16, b: u8x16) -> u8x16;
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use v128::*;
    use x86::ssse3 as ssse3;

    #[simd_test = "ssse3"]
    unsafe fn _mm_abs_epi8() {
        let r = ssse3::_mm_abs_epi8(i8x16::splat(-5));
        assert_eq!(r, u8x16::splat(5));
    }

    #[simd_test = "ssse3"]
    unsafe fn _mm_shuffle_epi8() {
        let a = u8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
        );
        let b = u8x16::new(
            4, 128, 4, 3,
            24, 12, 6, 19,
            12, 5, 5, 10,
            4, 1, 8, 0,
        );
        let expected = u8x16::new(
            5, 0, 5, 4,
            9, 13, 7, 4,
            13, 6, 6, 11,
            5, 2, 9, 1,
        );
        let r = ssse3::_mm_shuffle_epi8(a, b);
        assert_eq!(r, expected);
    }
}
