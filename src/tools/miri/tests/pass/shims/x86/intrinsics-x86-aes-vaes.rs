// We're testing x86 target specific features
//@only-target: x86_64 i686
//@compile-flags: -C target-feature=+aes,+vaes,+avx512f

#![feature(stdarch_x86_avx512)]

use core::mem::transmute;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn main() {
    assert!(is_x86_feature_detected!("aes"));
    assert!(is_x86_feature_detected!("vaes"));
    assert!(is_x86_feature_detected!("avx512f"));

    unsafe {
        test_aes();
        test_vaes();
    }
}

// The constants in the tests below are just bit patterns. They should not
// be interpreted as integers; signedness does not make sense for them, but
// __m128i happens to be defined in terms of signed integers.
#[allow(overflowing_literals)]
#[target_feature(enable = "aes")]
unsafe fn test_aes() {
    // Mostly copied from library/stdarch/crates/core_arch/src/x86/aes.rs

    #[target_feature(enable = "aes")]
    unsafe fn test_mm_aesdec_si128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc664949.aspx.
        let a = _mm_set_epi64x(0x0123456789abcdef, 0x8899aabbccddeeff);
        let k = _mm_set_epi64x(0x1133557799bbddff, 0x0022446688aaccee);
        let e = _mm_set_epi64x(0x044e4f5176fec48f, 0xb57ecfa381da39ee);
        let r = _mm_aesdec_si128(a, k);
        assert_eq_m128i(r, e);
    }
    test_mm_aesdec_si128();

    #[target_feature(enable = "aes")]
    unsafe fn test_mm_aesdeclast_si128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc714178.aspx.
        let a = _mm_set_epi64x(0x0123456789abcdef, 0x8899aabbccddeeff);
        let k = _mm_set_epi64x(0x1133557799bbddff, 0x0022446688aaccee);
        let e = _mm_set_epi64x(0x36cad57d9072bf9e, 0xf210dd981fa4a493);
        let r = _mm_aesdeclast_si128(a, k);
        assert_eq_m128i(r, e);
    }
    test_mm_aesdeclast_si128();

    #[target_feature(enable = "aes")]
    unsafe fn test_mm_aesenc_si128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc664810.aspx.
        let a = _mm_set_epi64x(0x0123456789abcdef, 0x8899aabbccddeeff);
        let k = _mm_set_epi64x(0x1133557799bbddff, 0x0022446688aaccee);
        let e = _mm_set_epi64x(0x16ab0e57dfc442ed, 0x28e4ee1884504333);
        let r = _mm_aesenc_si128(a, k);
        assert_eq_m128i(r, e);
    }
    test_mm_aesenc_si128();

    #[target_feature(enable = "aes")]
    unsafe fn test_mm_aesenclast_si128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc714136.aspx.
        let a = _mm_set_epi64x(0x0123456789abcdef, 0x8899aabbccddeeff);
        let k = _mm_set_epi64x(0x1133557799bbddff, 0x0022446688aaccee);
        let e = _mm_set_epi64x(0xb6dd7df25d7ab320, 0x4b04f98cf4c860f8);
        let r = _mm_aesenclast_si128(a, k);
        assert_eq_m128i(r, e);
    }
    test_mm_aesenclast_si128();

    #[target_feature(enable = "aes")]
    unsafe fn test_mm_aesimc_si128() {
        // Constants taken from https://msdn.microsoft.com/en-us/library/cc714195.aspx.
        let a = _mm_set_epi64x(0x0123456789abcdef, 0x8899aabbccddeeff);
        let e = _mm_set_epi64x(0xc66c82284ee40aa0, 0x6633441122770055);
        let r = _mm_aesimc_si128(a);
        assert_eq_m128i(r, e);
    }
    test_mm_aesimc_si128();
}

// The constants in the tests below are just bit patterns. They should not
// be interpreted as integers; signedness does not make sense for them, but
// __m128i happens to be defined in terms of signed integers.
#[allow(overflowing_literals)]
#[target_feature(enable = "vaes,avx512f")]
unsafe fn test_vaes() {
    #[target_feature(enable = "avx")]
    unsafe fn get_a256() -> __m256i {
        // Constants are random
        _mm256_set_epi64x(
            0xb89f43a558d3cd51,
            0x57b3e81e369bd603,
            0xf177a1a626933fd6,
            0x50d8adbed1a2f9d7,
        )
    }
    #[target_feature(enable = "avx")]
    unsafe fn get_k256() -> __m256i {
        // Constants are random
        _mm256_set_epi64x(
            0x503ff704588b5627,
            0xe23d882ed9c3c146,
            0x2785e5b670155b3c,
            0xa750718e183549ff,
        )
    }

    #[target_feature(enable = "vaes")]
    unsafe fn test_mm256_aesdec_epi128() {
        let a = get_a256();
        let k = get_k256();
        let r = _mm256_aesdec_epi128(a, k);

        // Check results.
        let a: [u128; 2] = transmute(a);
        let k: [u128; 2] = transmute(k);
        let r: [u128; 2] = transmute(r);
        for i in 0..2 {
            let e: u128 = transmute(_mm_aesdec_si128(transmute(a[i]), transmute(k[i])));
            assert_eq!(r[i], e);
        }
    }
    test_mm256_aesdec_epi128();

    #[target_feature(enable = "vaes")]
    unsafe fn test_mm256_aesdeclast_epi128() {
        let a = get_a256();
        let k = get_k256();
        let r = _mm256_aesdeclast_epi128(a, k);

        // Check results.
        let a: [u128; 2] = transmute(a);
        let k: [u128; 2] = transmute(k);
        let r: [u128; 2] = transmute(r);
        for i in 0..2 {
            let e: u128 = transmute(_mm_aesdeclast_si128(transmute(a[i]), transmute(k[i])));
            assert_eq!(r[i], e);
        }
    }
    test_mm256_aesdeclast_epi128();

    #[target_feature(enable = "vaes")]
    unsafe fn test_mm256_aesenc_epi128() {
        let a = get_a256();
        let k = get_k256();
        let r = _mm256_aesenc_epi128(a, k);

        // Check results.
        let a: [u128; 2] = transmute(a);
        let k: [u128; 2] = transmute(k);
        let r: [u128; 2] = transmute(r);
        for i in 0..2 {
            let e: u128 = transmute(_mm_aesenc_si128(transmute(a[i]), transmute(k[i])));
            assert_eq!(r[i], e);
        }
    }
    test_mm256_aesenc_epi128();

    #[target_feature(enable = "vaes")]
    unsafe fn test_mm256_aesenclast_epi128() {
        let a = get_a256();
        let k = get_k256();
        let r = _mm256_aesenclast_epi128(a, k);

        // Check results.
        let a: [u128; 2] = transmute(a);
        let k: [u128; 2] = transmute(k);
        let r: [u128; 2] = transmute(r);
        for i in 0..2 {
            let e: u128 = transmute(_mm_aesenclast_si128(transmute(a[i]), transmute(k[i])));
            assert_eq!(r[i], e);
        }
    }
    test_mm256_aesenclast_epi128();

    #[target_feature(enable = "avx512f")]
    unsafe fn get_a512() -> __m512i {
        // Constants are random
        _mm512_set_epi64(
            0xb89f43a558d3cd51,
            0x57b3e81e369bd603,
            0xf177a1a626933fd6,
            0x50d8adbed1a2f9d7,
            0xfbfee3116629db78,
            0x6aef4a91f2ad50f4,
            0x4258bb51ff1d476d,
            0x31da65761c8016cf,
        )
    }
    #[target_feature(enable = "avx512f")]
    unsafe fn get_k512() -> __m512i {
        // Constants are random
        _mm512_set_epi64(
            0x503ff704588b5627,
            0xe23d882ed9c3c146,
            0x2785e5b670155b3c,
            0xa750718e183549ff,
            0xdfb408830a65d3d9,
            0x0de3d92adac81b0a,
            0xed2741fe12877cae,
            0x3251ddb5404e0974,
        )
    }

    #[target_feature(enable = "vaes,avx512f")]
    unsafe fn test_mm512_aesdec_epi128() {
        let a = get_a512();
        let k = get_k512();
        let r = _mm512_aesdec_epi128(a, k);

        // Check results.
        let a: [u128; 4] = transmute(a);
        let k: [u128; 4] = transmute(k);
        let r: [u128; 4] = transmute(r);
        for i in 0..4 {
            let e: u128 = transmute(_mm_aesdec_si128(transmute(a[i]), transmute(k[i])));
            assert_eq!(r[i], e);
        }
    }
    test_mm512_aesdec_epi128();

    #[target_feature(enable = "vaes,avx512f")]
    unsafe fn test_mm512_aesdeclast_epi128() {
        let a = get_a512();
        let k = get_k512();
        let r = _mm512_aesdeclast_epi128(a, k);

        // Check results.
        let a: [u128; 4] = transmute(a);
        let k: [u128; 4] = transmute(k);
        let r: [u128; 4] = transmute(r);
        for i in 0..4 {
            let e: u128 = transmute(_mm_aesdeclast_si128(transmute(a[i]), transmute(k[i])));
            assert_eq!(r[i], e);
        }
    }
    test_mm512_aesdeclast_epi128();

    #[target_feature(enable = "vaes,avx512f")]
    unsafe fn test_mm512_aesenc_epi128() {
        let a = get_a512();
        let k = get_k512();
        let r = _mm512_aesenc_epi128(a, k);

        // Check results.
        let a: [u128; 4] = transmute(a);
        let k: [u128; 4] = transmute(k);
        let r: [u128; 4] = transmute(r);
        for i in 0..4 {
            let e: u128 = transmute(_mm_aesenc_si128(transmute(a[i]), transmute(k[i])));
            assert_eq!(r[i], e);
        }
    }
    test_mm512_aesenc_epi128();

    #[target_feature(enable = "vaes,avx512f")]
    unsafe fn test_mm512_aesenclast_epi128() {
        let a = get_a512();
        let k = get_k512();
        let r = _mm512_aesenclast_epi128(a, k);

        // Check results.
        let a: [u128; 4] = transmute(a);
        let k: [u128; 4] = transmute(k);
        let r: [u128; 4] = transmute(r);
        for i in 0..4 {
            let e: u128 = transmute(_mm_aesenclast_si128(transmute(a[i]), transmute(k[i])));
            assert_eq!(r[i], e);
        }
    }
    test_mm512_aesenclast_epi128();
}

#[track_caller]
#[target_feature(enable = "sse2")]
unsafe fn assert_eq_m128i(a: __m128i, b: __m128i) {
    assert_eq!(transmute::<_, [u64; 2]>(a), transmute::<_, [u64; 2]>(b))
}
