//! AES New Instructions (AES-NI)
//!
//! The intrinsics here correspond to those in the `wmmintrin.h` C header.
//!
//! The reference is [Intel 64 and IA-32 Architectures Software Developer's
//! Manual Volume 2: Instruction Set Reference, A-Z][intel64_ref].
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf

use crate::core_arch::x86::__m128i;

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.x86.aesni.aesdec"]
    fn aesdec(a: __m128i, round_key: __m128i) -> __m128i;
    #[link_name = "llvm.x86.aesni.aesdeclast"]
    fn aesdeclast(a: __m128i, round_key: __m128i) -> __m128i;
    #[link_name = "llvm.x86.aesni.aesenc"]
    fn aesenc(a: __m128i, round_key: __m128i) -> __m128i;
    #[link_name = "llvm.x86.aesni.aesenclast"]
    fn aesenclast(a: __m128i, round_key: __m128i) -> __m128i;
    #[link_name = "llvm.x86.aesni.aesimc"]
    fn aesimc(a: __m128i) -> __m128i;
    #[link_name = "llvm.x86.aesni.aeskeygenassist"]
    fn aeskeygenassist(a: __m128i, imm8: u8) -> __m128i;
}

/// Performs one round of an AES decryption flow on data (state) in `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesdec_si128)
#[inline]
#[target_feature(enable = "aes")]
#[cfg_attr(test, assert_instr(aesdec))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_aesdec_si128(a: __m128i, round_key: __m128i) -> __m128i {
    unsafe { aesdec(a, round_key) }
}

/// Performs the last round of an AES decryption flow on data (state) in `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesdeclast_si128)
#[inline]
#[target_feature(enable = "aes")]
#[cfg_attr(test, assert_instr(aesdeclast))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_aesdeclast_si128(a: __m128i, round_key: __m128i) -> __m128i {
    unsafe { aesdeclast(a, round_key) }
}

/// Performs one round of an AES encryption flow on data (state) in `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesenc_si128)
#[inline]
#[target_feature(enable = "aes")]
#[cfg_attr(test, assert_instr(aesenc))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_aesenc_si128(a: __m128i, round_key: __m128i) -> __m128i {
    unsafe { aesenc(a, round_key) }
}

/// Performs the last round of an AES encryption flow on data (state) in `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesenclast_si128)
#[inline]
#[target_feature(enable = "aes")]
#[cfg_attr(test, assert_instr(aesenclast))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_aesenclast_si128(a: __m128i, round_key: __m128i) -> __m128i {
    unsafe { aesenclast(a, round_key) }
}

/// Performs the `InvMixColumns` transformation on `a`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesimc_si128)
#[inline]
#[target_feature(enable = "aes")]
#[cfg_attr(test, assert_instr(aesimc))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_aesimc_si128(a: __m128i) -> __m128i {
    unsafe { aesimc(a) }
}

/// Assist in expanding the AES cipher key.
///
/// Assist in expanding the AES cipher key by computing steps towards
/// generating a round key for encryption cipher using data from `a` and an
/// 8-bit round constant `IMM8`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aeskeygenassist_si128)
#[inline]
#[target_feature(enable = "aes")]
#[cfg_attr(test, assert_instr(aeskeygenassist, IMM8 = 0))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_aeskeygenassist_si128<const IMM8: i32>(a: __m128i) -> __m128i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { aeskeygenassist(a, IMM8 as u8) }
}

#[cfg(test)]
mod tests {
    // The constants in the tests below are just bit patterns. They should not
    // be interpreted as integers; signedness does not make sense for them, but
    // __m128i happens to be defined in terms of signed integers.
    #![allow(overflowing_literals)]

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "aes")]
    unsafe fn test_mm_aesdec_si128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc664949.aspx.
        let a = _mm_set_epi64x(0x0123456789abcdef, 0x8899aabbccddeeff);
        let k = _mm_set_epi64x(0x1133557799bbddff, 0x0022446688aaccee);
        let e = _mm_set_epi64x(0x044e4f5176fec48f, 0xb57ecfa381da39ee);
        let r = _mm_aesdec_si128(a, k);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "aes")]
    unsafe fn test_mm_aesdeclast_si128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc714178.aspx.
        let a = _mm_set_epi64x(0x0123456789abcdef, 0x8899aabbccddeeff);
        let k = _mm_set_epi64x(0x1133557799bbddff, 0x0022446688aaccee);
        let e = _mm_set_epi64x(0x36cad57d9072bf9e, 0xf210dd981fa4a493);
        let r = _mm_aesdeclast_si128(a, k);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "aes")]
    unsafe fn test_mm_aesenc_si128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc664810.aspx.
        let a = _mm_set_epi64x(0x0123456789abcdef, 0x8899aabbccddeeff);
        let k = _mm_set_epi64x(0x1133557799bbddff, 0x0022446688aaccee);
        let e = _mm_set_epi64x(0x16ab0e57dfc442ed, 0x28e4ee1884504333);
        let r = _mm_aesenc_si128(a, k);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "aes")]
    unsafe fn test_mm_aesenclast_si128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc714136.aspx.
        let a = _mm_set_epi64x(0x0123456789abcdef, 0x8899aabbccddeeff);
        let k = _mm_set_epi64x(0x1133557799bbddff, 0x0022446688aaccee);
        let e = _mm_set_epi64x(0xb6dd7df25d7ab320, 0x4b04f98cf4c860f8);
        let r = _mm_aesenclast_si128(a, k);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "aes")]
    unsafe fn test_mm_aesimc_si128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc714195.aspx.
        let a = _mm_set_epi64x(0x0123456789abcdef, 0x8899aabbccddeeff);
        let e = _mm_set_epi64x(0xc66c82284ee40aa0, 0x6633441122770055);
        let r = _mm_aesimc_si128(a);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "aes")]
    unsafe fn test_mm_aeskeygenassist_si128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc714138.aspx.
        let a = _mm_set_epi64x(0x0123456789abcdef, 0x8899aabbccddeeff);
        let e = _mm_set_epi64x(0x857c266b7c266e85, 0xeac4eea9c4eeacea);
        let r = _mm_aeskeygenassist_si128::<5>(a);
        assert_eq_m128i(r, e);
    }
}
