// We're testing x86 target specific features
//@revisions: avx512 avx
//@only-target: x86_64 i686
//@[avx512]compile-flags: -C target-feature=+vpclmulqdq,+avx512f
//@[avx]compile-flags: -C target-feature=+vpclmulqdq,+avx2

// The constants in the tests below are just bit patterns. They should not
// be interpreted as integers; signedness does not make sense for them, but
// __mXXXi happens to be defined in terms of signed integers.
#![allow(overflowing_literals)]
#![feature(stdarch_x86_avx512)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::transmute;

fn main() {
    // Mostly copied from library/stdarch/crates/core_arch/src/x86/vpclmulqdq.rs

    assert!(is_x86_feature_detected!("pclmulqdq"));
    assert!(is_x86_feature_detected!("vpclmulqdq"));

    unsafe {
        test_mm256_clmulepi64_epi128();

        if is_x86_feature_detected!("avx512f") {
            test_mm512_clmulepi64_epi128();
        }
    }
}

macro_rules! verify_kat_pclmul {
    ($broadcast:ident, $clmul:ident, $assert:ident) => {
        // Constants taken from https://software.intel.com/sites/default/files/managed/72/cc/clmul-wp-rev-2.02-2014-04-20.pdf
        let a = _mm_set_epi64x(0x7b5b546573745665, 0x63746f725d53475d);
        let a = $broadcast(a);
        let b = _mm_set_epi64x(0x4869285368617929, 0x5b477565726f6e5d);
        let b = $broadcast(b);
        let r00 = _mm_set_epi64x(0x1d4d84c85c3440c0, 0x929633d5d36f0451);
        let r00 = $broadcast(r00);
        let r01 = _mm_set_epi64x(0x1bd17c8d556ab5a1, 0x7fa540ac2a281315);
        let r01 = $broadcast(r01);
        let r10 = _mm_set_epi64x(0x1a2bf6db3a30862f, 0xbabf262df4b7d5c9);
        let r10 = $broadcast(r10);
        let r11 = _mm_set_epi64x(0x1d1e1f2c592e7c45, 0xd66ee03e410fd4ed);
        let r11 = $broadcast(r11);

        $assert($clmul::<0x00>(a, b), r00);
        $assert($clmul::<0x10>(a, b), r01);
        $assert($clmul::<0x01>(a, b), r10);
        $assert($clmul::<0x11>(a, b), r11);

        let a0 = _mm_set_epi64x(0x0000000000000000, 0x8000000000000000);
        let a0 = $broadcast(a0);
        let r = _mm_set_epi64x(0x4000000000000000, 0x0000000000000000);
        let r = $broadcast(r);
        $assert($clmul::<0x00>(a0, a0), r);
    }
}

// this function tests one of the possible 4 instances
// with different inputs across lanes for the 512-bit version
#[target_feature(enable = "vpclmulqdq,avx512f")]
unsafe fn verify_512_helper(
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
    let b = _mm512_set_epi64(
        0x672F6F105A94CEA7,
        0x8298B8FFCA5F829C,
        0xA3927047B3FB61D8,
        0x978093862CDE7187,
        0xB1927AB22F31D0EC,
        0xA9A5DA619BE4D7AF,
        0xCA2590F56884FDC6,
        0x19BE9F660038BDB5,
    );

    let a_decomp = transmute::<_, [__m128i; 4]>(a);
    let b_decomp = transmute::<_, [__m128i; 4]>(b);

    let r = vectorized(a, b);

    let e_decomp = [
        linear(a_decomp[0], b_decomp[0]),
        linear(a_decomp[1], b_decomp[1]),
        linear(a_decomp[2], b_decomp[2]),
        linear(a_decomp[3], b_decomp[3]),
    ];
    let e = transmute::<_, __m512i>(e_decomp);

    assert_eq_m512i(r, e)
}

// this function tests one of the possible 4 instances
// with different inputs across lanes for the 256-bit version
#[target_feature(enable = "vpclmulqdq")]
unsafe fn verify_256_helper(
    linear: unsafe fn(__m128i, __m128i) -> __m128i,
    vectorized: unsafe fn(__m256i, __m256i) -> __m256i,
) {
    let a = _mm256_set_epi64x(
        0xDCB4DB3657BF0B7D,
        0x18DB0601068EDD9F,
        0xB76B908233200DC5,
        0xE478235FA8E22D5E,
    );
    let b = _mm256_set_epi64x(
        0x672F6F105A94CEA7,
        0x8298B8FFCA5F829C,
        0xA3927047B3FB61D8,
        0x978093862CDE7187,
    );

    let a_decomp = transmute::<_, [__m128i; 2]>(a);
    let b_decomp = transmute::<_, [__m128i; 2]>(b);

    let r = vectorized(a, b);

    let e_decomp = [linear(a_decomp[0], b_decomp[0]), linear(a_decomp[1], b_decomp[1])];
    let e = transmute::<_, __m256i>(e_decomp);

    assert_eq_m256i(r, e)
}

#[target_feature(enable = "vpclmulqdq,avx512f")]
unsafe fn test_mm512_clmulepi64_epi128() {
    verify_kat_pclmul!(_mm512_broadcast_i32x4, _mm512_clmulepi64_epi128, assert_eq_m512i);

    verify_512_helper(
        |a, b| _mm_clmulepi64_si128::<0x00>(a, b),
        |a, b| _mm512_clmulepi64_epi128::<0x00>(a, b),
    );
    verify_512_helper(
        |a, b| _mm_clmulepi64_si128::<0x01>(a, b),
        |a, b| _mm512_clmulepi64_epi128::<0x01>(a, b),
    );
    verify_512_helper(
        |a, b| _mm_clmulepi64_si128::<0x10>(a, b),
        |a, b| _mm512_clmulepi64_epi128::<0x10>(a, b),
    );
    verify_512_helper(
        |a, b| _mm_clmulepi64_si128::<0x11>(a, b),
        |a, b| _mm512_clmulepi64_epi128::<0x11>(a, b),
    );
}

#[target_feature(enable = "vpclmulqdq")]
unsafe fn test_mm256_clmulepi64_epi128() {
    verify_kat_pclmul!(_mm256_broadcastsi128_si256, _mm256_clmulepi64_epi128, assert_eq_m256i);

    verify_256_helper(
        |a, b| _mm_clmulepi64_si128::<0x00>(a, b),
        |a, b| _mm256_clmulepi64_epi128::<0x00>(a, b),
    );
    verify_256_helper(
        |a, b| _mm_clmulepi64_si128::<0x01>(a, b),
        |a, b| _mm256_clmulepi64_epi128::<0x01>(a, b),
    );
    verify_256_helper(
        |a, b| _mm_clmulepi64_si128::<0x10>(a, b),
        |a, b| _mm256_clmulepi64_epi128::<0x10>(a, b),
    );
    verify_256_helper(
        |a, b| _mm_clmulepi64_si128::<0x11>(a, b),
        |a, b| _mm256_clmulepi64_epi128::<0x11>(a, b),
    );
}

#[track_caller]
#[target_feature(enable = "avx512f")]
unsafe fn assert_eq_m512i(a: __m512i, b: __m512i) {
    assert_eq!(transmute::<_, [u64; 8]>(a), transmute::<_, [u64; 8]>(b))
}

#[track_caller]
#[target_feature(enable = "avx")]
unsafe fn assert_eq_m256i(a: __m256i, b: __m256i) {
    assert_eq!(transmute::<_, [u64; 4]>(a), transmute::<_, [u64; 4]>(b))
}
