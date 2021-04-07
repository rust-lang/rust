//! Carry-less Multiplication (CLMUL)
//!
//! The reference is [Intel 64 and IA-32 Architectures Software Developer's
//! Manual Volume 2: Instruction Set Reference, A-Z][intel64_ref] (p. 4-241).
//!
//! [intel64_ref]: http://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf

use crate::core_arch::x86::__m128i;

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.pclmulqdq"]
    fn pclmulqdq(a: __m128i, round_key: __m128i, imm8: u8) -> __m128i;
}

/// Performs a carry-less multiplication of two 64-bit polynomials over the
/// finite field GF(2^k).
///
/// The immediate byte is used for determining which halves of `a` and `b`
/// should be used. Immediate bits other than 0 and 4 are ignored.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_clmulepi64_si128)
#[inline]
#[target_feature(enable = "pclmulqdq")]
#[cfg_attr(all(test, not(target_os = "linux")), assert_instr(pclmulqdq, IMM8 = 0))]
#[cfg_attr(all(test, target_os = "linux"), assert_instr(pclmullqlqdq, IMM8 = 0))]
#[cfg_attr(all(test, target_os = "linux"), assert_instr(pclmulhqlqdq, IMM8 = 1))]
#[cfg_attr(all(test, target_os = "linux"), assert_instr(pclmullqhqdq, IMM8 = 16))]
#[cfg_attr(all(test, target_os = "linux"), assert_instr(pclmulhqhqdq, IMM8 = 17))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _mm_clmulepi64_si128<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    static_assert_imm8!(IMM8);
    pclmulqdq(a, b, IMM8 as u8)
}

#[cfg(test)]
mod tests {
    // The constants in the tests below are just bit patterns. They should not
    // be interpreted as integers; signedness does not make sense for them, but
    // __m128i happens to be defined in terms of signed integers.
    #![allow(overflowing_literals)]

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "pclmulqdq")]
    unsafe fn test_mm_clmulepi64_si128() {
        // Constants taken from https://software.intel.com/sites/default/files/managed/72/cc/clmul-wp-rev-2.02-2014-04-20.pdf
        let a = _mm_set_epi64x(0x7b5b546573745665, 0x63746f725d53475d);
        let b = _mm_set_epi64x(0x4869285368617929, 0x5b477565726f6e5d);
        let r00 = _mm_set_epi64x(0x1d4d84c85c3440c0, 0x929633d5d36f0451);
        let r01 = _mm_set_epi64x(0x1bd17c8d556ab5a1, 0x7fa540ac2a281315);
        let r10 = _mm_set_epi64x(0x1a2bf6db3a30862f, 0xbabf262df4b7d5c9);
        let r11 = _mm_set_epi64x(0x1d1e1f2c592e7c45, 0xd66ee03e410fd4ed);

        assert_eq_m128i(_mm_clmulepi64_si128::<0x00>(a, b), r00);
        assert_eq_m128i(_mm_clmulepi64_si128::<0x10>(a, b), r01);
        assert_eq_m128i(_mm_clmulepi64_si128::<0x01>(a, b), r10);
        assert_eq_m128i(_mm_clmulepi64_si128::<0x11>(a, b), r11);

        let a0 = _mm_set_epi64x(0x0000000000000000, 0x8000000000000000);
        let r = _mm_set_epi64x(0x4000000000000000, 0x0000000000000000);
        assert_eq_m128i(_mm_clmulepi64_si128::<0x00>(a0, a0), r);
    }
}
