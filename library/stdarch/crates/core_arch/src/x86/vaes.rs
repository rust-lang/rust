//! Vectorized AES Instructions (VAES)
//!
//! The intrinsics here correspond to those in the `immintrin.h` C header.
//!
//! The reference is [Intel 64 and IA-32 Architectures Software Developer's
//! Manual Volume 2: Instruction Set Reference, A-Z][intel64_ref].
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf

use crate::core_arch::x86::__m256i;
use crate::core_arch::x86::__m512i;

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.x86.aesni.aesenc.256"]
    fn aesenc_256(a: __m256i, round_key: __m256i) -> __m256i;
    #[link_name = "llvm.x86.aesni.aesenclast.256"]
    fn aesenclast_256(a: __m256i, round_key: __m256i) -> __m256i;
    #[link_name = "llvm.x86.aesni.aesdec.256"]
    fn aesdec_256(a: __m256i, round_key: __m256i) -> __m256i;
    #[link_name = "llvm.x86.aesni.aesdeclast.256"]
    fn aesdeclast_256(a: __m256i, round_key: __m256i) -> __m256i;
    #[link_name = "llvm.x86.aesni.aesenc.512"]
    fn aesenc_512(a: __m512i, round_key: __m512i) -> __m512i;
    #[link_name = "llvm.x86.aesni.aesenclast.512"]
    fn aesenclast_512(a: __m512i, round_key: __m512i) -> __m512i;
    #[link_name = "llvm.x86.aesni.aesdec.512"]
    fn aesdec_512(a: __m512i, round_key: __m512i) -> __m512i;
    #[link_name = "llvm.x86.aesni.aesdeclast.512"]
    fn aesdeclast_512(a: __m512i, round_key: __m512i) -> __m512i;
}

/// Performs one round of an AES encryption flow on each 128-bit word (state) in `a` using
/// the corresponding 128-bit word (key) in `round_key`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_aesenc_epi128)
#[inline]
#[target_feature(enable = "vaes")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vaesenc))]
pub fn _mm256_aesenc_epi128(a: __m256i, round_key: __m256i) -> __m256i {
    unsafe { aesenc_256(a, round_key) }
}

/// Performs the last round of an AES encryption flow on each 128-bit word (state) in `a` using
/// the corresponding 128-bit word (key) in `round_key`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_aesenclast_epi128)
#[inline]
#[target_feature(enable = "vaes")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vaesenclast))]
pub fn _mm256_aesenclast_epi128(a: __m256i, round_key: __m256i) -> __m256i {
    unsafe { aesenclast_256(a, round_key) }
}

/// Performs one round of an AES decryption flow on each 128-bit word (state) in `a` using
/// the corresponding 128-bit word (key) in `round_key`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_aesdec_epi128)
#[inline]
#[target_feature(enable = "vaes")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vaesdec))]
pub fn _mm256_aesdec_epi128(a: __m256i, round_key: __m256i) -> __m256i {
    unsafe { aesdec_256(a, round_key) }
}

/// Performs the last round of an AES decryption flow on each 128-bit word (state) in `a` using
/// the corresponding 128-bit word (key) in `round_key`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_aesdeclast_epi128)
#[inline]
#[target_feature(enable = "vaes")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vaesdeclast))]
pub fn _mm256_aesdeclast_epi128(a: __m256i, round_key: __m256i) -> __m256i {
    unsafe { aesdeclast_256(a, round_key) }
}

/// Performs one round of an AES encryption flow on each 128-bit word (state) in `a` using
/// the corresponding 128-bit word (key) in `round_key`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_aesenc_epi128)
#[inline]
#[target_feature(enable = "vaes,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vaesenc))]
pub fn _mm512_aesenc_epi128(a: __m512i, round_key: __m512i) -> __m512i {
    unsafe { aesenc_512(a, round_key) }
}

/// Performs the last round of an AES encryption flow on each 128-bit word (state) in `a` using
/// the corresponding 128-bit word (key) in `round_key`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_aesenclast_epi128)
#[inline]
#[target_feature(enable = "vaes,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vaesenclast))]
pub fn _mm512_aesenclast_epi128(a: __m512i, round_key: __m512i) -> __m512i {
    unsafe { aesenclast_512(a, round_key) }
}

/// Performs one round of an AES decryption flow on each 128-bit word (state) in `a` using
/// the corresponding 128-bit word (key) in `round_key`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_aesdec_epi128)
#[inline]
#[target_feature(enable = "vaes,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vaesdec))]
pub fn _mm512_aesdec_epi128(a: __m512i, round_key: __m512i) -> __m512i {
    unsafe { aesdec_512(a, round_key) }
}

/// Performs the last round of an AES decryption flow on each 128-bit word (state) in `a` using
/// the corresponding 128-bit word (key) in `round_key`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_aesdeclast_epi128)
#[inline]
#[target_feature(enable = "vaes,avx512f")]
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
#[cfg_attr(test, assert_instr(vaesdeclast))]
pub fn _mm512_aesdeclast_epi128(a: __m512i, round_key: __m512i) -> __m512i {
    unsafe { aesdeclast_512(a, round_key) }
}

#[cfg(test)]
mod tests {
    // The constants in the tests below are just bit patterns. They should not
    // be interpreted as integers; signedness does not make sense for them, but
    // __mXXXi happens to be defined in terms of signed integers.
    #![allow(overflowing_literals)]

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    // the first parts of these tests are straight ports from the AES-NI tests
    // the second parts directly compare the two, for inputs that are different across lanes
    // and "more random" than the standard test vectors
    // ideally we'd be using quickcheck here instead

    #[target_feature(enable = "avx2")]
    unsafe fn helper_for_256_vaes(
        linear: unsafe fn(__m128i, __m128i) -> __m128i,
        vectorized: unsafe fn(__m256i, __m256i) -> __m256i,
    ) {
        let a = _mm256_set_epi64x(
            0xDCB4DB3657BF0B7D,
            0x18DB0601068EDD9F,
            0xB76B908233200DC5,
            0xE478235FA8E22D5E,
        );
        let k = _mm256_set_epi64x(
            0x672F6F105A94CEA7,
            0x8298B8FFCA5F829C,
            0xA3927047B3FB61D8,
            0x978093862CDE7187,
        );
        let mut a_decomp = [_mm_setzero_si128(); 2];
        a_decomp[0] = _mm256_extracti128_si256::<0>(a);
        a_decomp[1] = _mm256_extracti128_si256::<1>(a);
        let mut k_decomp = [_mm_setzero_si128(); 2];
        k_decomp[0] = _mm256_extracti128_si256::<0>(k);
        k_decomp[1] = _mm256_extracti128_si256::<1>(k);
        let r = vectorized(a, k);
        let mut e_decomp = [_mm_setzero_si128(); 2];
        for i in 0..2 {
            e_decomp[i] = linear(a_decomp[i], k_decomp[i]);
        }
        assert_eq_m128i(_mm256_extracti128_si256::<0>(r), e_decomp[0]);
        assert_eq_m128i(_mm256_extracti128_si256::<1>(r), e_decomp[1]);
    }

    #[target_feature(enable = "sse2")]
    unsafe fn setup_state_key<T>(broadcast: unsafe fn(__m128i) -> T) -> (T, T) {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc664949.aspx.
        let a = _mm_set_epi64x(0x0123456789abcdef, 0x8899aabbccddeeff);
        let k = _mm_set_epi64x(0x1133557799bbddff, 0x0022446688aaccee);
        (broadcast(a), broadcast(k))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn setup_state_key_256() -> (__m256i, __m256i) {
        setup_state_key(_mm256_broadcastsi128_si256)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn setup_state_key_512() -> (__m512i, __m512i) {
        setup_state_key(_mm512_broadcast_i32x4)
    }

    #[simd_test(enable = "vaes,avx512vl")]
    unsafe fn test_mm256_aesdec_epi128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc664949.aspx.
        let (a, k) = setup_state_key_256();
        let e = _mm_set_epi64x(0x044e4f5176fec48f, 0xb57ecfa381da39ee);
        let e = _mm256_broadcastsi128_si256(e);
        let r = _mm256_aesdec_epi128(a, k);
        assert_eq_m256i(r, e);

        helper_for_256_vaes(_mm_aesdec_si128, _mm256_aesdec_epi128);
    }

    #[simd_test(enable = "vaes,avx512vl")]
    unsafe fn test_mm256_aesdeclast_epi128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc714178.aspx.
        let (a, k) = setup_state_key_256();
        let e = _mm_set_epi64x(0x36cad57d9072bf9e, 0xf210dd981fa4a493);
        let e = _mm256_broadcastsi128_si256(e);
        let r = _mm256_aesdeclast_epi128(a, k);
        assert_eq_m256i(r, e);

        helper_for_256_vaes(_mm_aesdeclast_si128, _mm256_aesdeclast_epi128);
    }

    #[simd_test(enable = "vaes,avx512vl")]
    unsafe fn test_mm256_aesenc_epi128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc664810.aspx.
        // they are repeated appropriately
        let (a, k) = setup_state_key_256();
        let e = _mm_set_epi64x(0x16ab0e57dfc442ed, 0x28e4ee1884504333);
        let e = _mm256_broadcastsi128_si256(e);
        let r = _mm256_aesenc_epi128(a, k);
        assert_eq_m256i(r, e);

        helper_for_256_vaes(_mm_aesenc_si128, _mm256_aesenc_epi128);
    }

    #[simd_test(enable = "vaes,avx512vl")]
    unsafe fn test_mm256_aesenclast_epi128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc714136.aspx.
        let (a, k) = setup_state_key_256();
        let e = _mm_set_epi64x(0xb6dd7df25d7ab320, 0x4b04f98cf4c860f8);
        let e = _mm256_broadcastsi128_si256(e);
        let r = _mm256_aesenclast_epi128(a, k);
        assert_eq_m256i(r, e);

        helper_for_256_vaes(_mm_aesenclast_si128, _mm256_aesenclast_epi128);
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn helper_for_512_vaes(
        linear: unsafe fn(__m128i, __m128i) -> __m128i,
        vectorized: unsafe fn(__m512i, __m512i) -> __m512i,
    ) {
        let a = _mm512_set_epi64(
            0xDCB4DB3657BF0B7D,
            0x18DB0601068EDD9F,
            0xB76B908233200DC5,
            0xE478235FA8E22D5E,
            0xAB05CFFA2621154C,
            0x1171B47A186174C9,
            0x8C6B6C0E7595CEC9,
            0xBE3E7D4934E961BD,
        );
        let k = _mm512_set_epi64(
            0x672F6F105A94CEA7,
            0x8298B8FFCA5F829C,
            0xA3927047B3FB61D8,
            0x978093862CDE7187,
            0xB1927AB22F31D0EC,
            0xA9A5DA619BE4D7AF,
            0xCA2590F56884FDC6,
            0x19BE9F660038BDB5,
        );
        let mut a_decomp = [_mm_setzero_si128(); 4];
        a_decomp[0] = _mm512_extracti32x4_epi32::<0>(a);
        a_decomp[1] = _mm512_extracti32x4_epi32::<1>(a);
        a_decomp[2] = _mm512_extracti32x4_epi32::<2>(a);
        a_decomp[3] = _mm512_extracti32x4_epi32::<3>(a);
        let mut k_decomp = [_mm_setzero_si128(); 4];
        k_decomp[0] = _mm512_extracti32x4_epi32::<0>(k);
        k_decomp[1] = _mm512_extracti32x4_epi32::<1>(k);
        k_decomp[2] = _mm512_extracti32x4_epi32::<2>(k);
        k_decomp[3] = _mm512_extracti32x4_epi32::<3>(k);
        let r = vectorized(a, k);
        let mut e_decomp = [_mm_setzero_si128(); 4];
        for i in 0..4 {
            e_decomp[i] = linear(a_decomp[i], k_decomp[i]);
        }
        assert_eq_m128i(_mm512_extracti32x4_epi32::<0>(r), e_decomp[0]);
        assert_eq_m128i(_mm512_extracti32x4_epi32::<1>(r), e_decomp[1]);
        assert_eq_m128i(_mm512_extracti32x4_epi32::<2>(r), e_decomp[2]);
        assert_eq_m128i(_mm512_extracti32x4_epi32::<3>(r), e_decomp[3]);
    }

    #[simd_test(enable = "vaes,avx512f")]
    unsafe fn test_mm512_aesdec_epi128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc664949.aspx.
        let (a, k) = setup_state_key_512();
        let e = _mm_set_epi64x(0x044e4f5176fec48f, 0xb57ecfa381da39ee);
        let e = _mm512_broadcast_i32x4(e);
        let r = _mm512_aesdec_epi128(a, k);
        assert_eq_m512i(r, e);

        helper_for_512_vaes(_mm_aesdec_si128, _mm512_aesdec_epi128);
    }

    #[simd_test(enable = "vaes,avx512f")]
    unsafe fn test_mm512_aesdeclast_epi128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc714178.aspx.
        let (a, k) = setup_state_key_512();
        let e = _mm_set_epi64x(0x36cad57d9072bf9e, 0xf210dd981fa4a493);
        let e = _mm512_broadcast_i32x4(e);
        let r = _mm512_aesdeclast_epi128(a, k);
        assert_eq_m512i(r, e);

        helper_for_512_vaes(_mm_aesdeclast_si128, _mm512_aesdeclast_epi128);
    }

    #[simd_test(enable = "vaes,avx512f")]
    unsafe fn test_mm512_aesenc_epi128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc664810.aspx.
        let (a, k) = setup_state_key_512();
        let e = _mm_set_epi64x(0x16ab0e57dfc442ed, 0x28e4ee1884504333);
        let e = _mm512_broadcast_i32x4(e);
        let r = _mm512_aesenc_epi128(a, k);
        assert_eq_m512i(r, e);

        helper_for_512_vaes(_mm_aesenc_si128, _mm512_aesenc_epi128);
    }

    #[simd_test(enable = "vaes,avx512f")]
    unsafe fn test_mm512_aesenclast_epi128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc714136.aspx.
        let (a, k) = setup_state_key_512();
        let e = _mm_set_epi64x(0xb6dd7df25d7ab320, 0x4b04f98cf4c860f8);
        let e = _mm512_broadcast_i32x4(e);
        let r = _mm512_aesenclast_epi128(a, k);
        assert_eq_m512i(r, e);

        helper_for_512_vaes(_mm_aesenclast_si128, _mm512_aesenclast_epi128);
    }
}
