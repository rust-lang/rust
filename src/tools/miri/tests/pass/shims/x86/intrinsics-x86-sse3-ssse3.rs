// We're testing x86 target specific features
//@only-target: x86_64 i686
// SSSE3 implicitly enables SSE3
//@compile-flags: -C target-feature=+ssse3

use core::mem::transmute;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn main() {
    // SSSE3 implicitly enables SSE3, still check it to be sure
    assert!(is_x86_feature_detected!("sse3"));
    assert!(is_x86_feature_detected!("ssse3"));

    unsafe {
        test_sse3();
        test_ssse3();
    }
}

#[target_feature(enable = "sse3")]
unsafe fn test_sse3() {
    // Mostly copied from library/stdarch/crates/core_arch/src/x86/sse3.rs

    #[target_feature(enable = "sse3")]
    unsafe fn test_mm_addsub_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_addsub_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(99.0, 25.0, 0.0, -15.0));
    }
    test_mm_addsub_ps();

    #[target_feature(enable = "sse3")]
    unsafe fn test_mm_addsub_pd() {
        let a = _mm_setr_pd(-1.0, 5.0);
        let b = _mm_setr_pd(-100.0, 20.0);
        let r = _mm_addsub_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(99.0, 25.0));
    }
    test_mm_addsub_pd();

    #[target_feature(enable = "sse3")]
    unsafe fn test_mm_hadd_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_hadd_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(4.0, -10.0, -80.0, -5.0));
    }
    test_mm_hadd_ps();

    #[target_feature(enable = "sse3")]
    unsafe fn test_mm_hadd_pd() {
        let a = _mm_setr_pd(-1.0, 5.0);
        let b = _mm_setr_pd(-100.0, 20.0);
        let r = _mm_hadd_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(4.0, -80.0));
    }
    test_mm_hadd_pd();

    #[target_feature(enable = "sse3")]
    unsafe fn test_mm_hsub_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_hsub_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(-6.0, 10.0, -120.0, 5.0));
    }
    test_mm_hsub_ps();

    #[target_feature(enable = "sse3")]
    unsafe fn test_mm_hsub_pd() {
        let a = _mm_setr_pd(-1.0, 5.0);
        let b = _mm_setr_pd(-100.0, 20.0);
        let r = _mm_hsub_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(-6.0, -120.0));
    }
    test_mm_hsub_pd();

    #[target_feature(enable = "sse3")]
    unsafe fn test_mm_lddqu_si128() {
        let a = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm_lddqu_si128(&a);
        assert_eq_m128i(a, r);
    }
    test_mm_lddqu_si128();
}

#[target_feature(enable = "ssse3")]
unsafe fn test_ssse3() {
    // Mostly copied from library/stdarch/crates/core_arch/src/x86/ssse3.rs

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_abs_epi8() {
        let r = _mm_abs_epi8(_mm_set1_epi8(-5));
        assert_eq_m128i(r, _mm_set1_epi8(5));
    }
    test_mm_abs_epi8();

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_abs_epi16() {
        let r = _mm_abs_epi16(_mm_set1_epi16(-5));
        assert_eq_m128i(r, _mm_set1_epi16(5));
    }
    test_mm_abs_epi16();

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_abs_epi32() {
        let r = _mm_abs_epi32(_mm_set1_epi32(-5));
        assert_eq_m128i(r, _mm_set1_epi32(5));
    }
    test_mm_abs_epi32();

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_shuffle_epi8() {
        let a = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm_setr_epi8(4, 128_u8 as i8, 4, 3, 24, 12, 6, 19, 12, 5, 5, 10, 4, 1, 8, 0);
        let expected = _mm_setr_epi8(5, 0, 5, 4, 9, 13, 7, 4, 13, 6, 6, 11, 5, 2, 9, 1);
        let r = _mm_shuffle_epi8(a, b);
        assert_eq_m128i(r, expected);

        // Test indices greater than 15 wrapping around
        let b = _mm_add_epi8(b, _mm_set1_epi8(32));
        let r = _mm_shuffle_epi8(a, b);
        assert_eq_m128i(r, expected);
    }
    test_mm_shuffle_epi8();

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_hadd_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 24, 12, 6, 19);
        let expected = _mm_setr_epi16(3, 7, 11, 15, 132, 7, 36, 25);
        let r = _mm_hadd_epi16(a, b);
        assert_eq_m128i(r, expected);

        // Test wrapping on overflow
        let a = _mm_setr_epi16(i16::MAX, 1, i16::MAX, 2, i16::MAX, 3, i16::MAX, 4);
        let b = _mm_setr_epi16(i16::MIN, -1, i16::MIN, -2, i16::MIN, -3, i16::MIN, -4);
        let expected = _mm_setr_epi16(
            i16::MIN,
            i16::MIN + 1,
            i16::MIN + 2,
            i16::MIN + 3,
            i16::MAX,
            i16::MAX - 1,
            i16::MAX - 2,
            i16::MAX - 3,
        );
        let r = _mm_hadd_epi16(a, b);
        assert_eq_m128i(r, expected);
    }
    test_mm_hadd_epi16();

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_hadds_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 32767, 1, -32768, -1);
        let expected = _mm_setr_epi16(3, 7, 11, 15, 132, 7, 32767, -32768);
        let r = _mm_hadds_epi16(a, b);
        assert_eq_m128i(r, expected);

        // Test saturating on overflow
        let a = _mm_setr_epi16(i16::MAX, 1, i16::MAX, 2, i16::MAX, 3, i16::MAX, 4);
        let b = _mm_setr_epi16(i16::MIN, -1, i16::MIN, -2, i16::MIN, -3, i16::MIN, -4);
        let expected = _mm_setr_epi16(
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MIN,
            i16::MIN,
        );
        let r = _mm_hadds_epi16(a, b);
        assert_eq_m128i(r, expected);
    }
    test_mm_hadds_epi16();

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_hadd_epi32() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let b = _mm_setr_epi32(4, 128, 4, 3);
        let expected = _mm_setr_epi32(3, 7, 132, 7);
        let r = _mm_hadd_epi32(a, b);
        assert_eq_m128i(r, expected);

        // Test wrapping on overflow
        let a = _mm_setr_epi32(i32::MAX, 1, i32::MAX, 2);
        let b = _mm_setr_epi32(i32::MIN, -1, i32::MIN, -2);
        let expected = _mm_setr_epi32(i32::MIN, i32::MIN + 1, i32::MAX, i32::MAX - 1);
        let r = _mm_hadd_epi32(a, b);
        assert_eq_m128i(r, expected);
    }
    test_mm_hadd_epi32();

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_hsub_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 24, 12, 6, 19);
        let expected = _mm_setr_epi16(-1, -1, -1, -1, -124, 1, 12, -13);
        let r = _mm_hsub_epi16(a, b);
        assert_eq_m128i(r, expected);

        // Test wrapping on overflow
        let a = _mm_setr_epi16(i16::MAX, -1, i16::MAX, -2, i16::MAX, -3, i16::MAX, -4);
        let b = _mm_setr_epi16(i16::MIN, 1, i16::MIN, 2, i16::MIN, 3, i16::MIN, 4);
        let expected = _mm_setr_epi16(
            i16::MIN,
            i16::MIN + 1,
            i16::MIN + 2,
            i16::MIN + 3,
            i16::MAX,
            i16::MAX - 1,
            i16::MAX - 2,
            i16::MAX - 3,
        );
        let r = _mm_hsub_epi16(a, b);
        assert_eq_m128i(r, expected);
    }
    test_mm_hsub_epi16();

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_hsubs_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 32767, -1, -32768, 1);
        let expected = _mm_setr_epi16(-1, -1, -1, -1, -124, 1, 32767, -32768);
        let r = _mm_hsubs_epi16(a, b);
        assert_eq_m128i(r, expected);

        // Test saturating on overflow
        let a = _mm_setr_epi16(i16::MAX, -1, i16::MAX, -2, i16::MAX, -3, i16::MAX, -4);
        let b = _mm_setr_epi16(i16::MIN, 1, i16::MIN, 2, i16::MIN, 3, i16::MIN, 4);
        let expected = _mm_setr_epi16(
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MIN,
            i16::MIN,
        );
        let r = _mm_hsubs_epi16(a, b);
        assert_eq_m128i(r, expected);
    }
    test_mm_hsubs_epi16();

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_hsub_epi32() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let b = _mm_setr_epi32(4, 128, 4, 3);
        let expected = _mm_setr_epi32(-1, -1, -124, 1);
        let r = _mm_hsub_epi32(a, b);
        assert_eq_m128i(r, expected);

        // Test wrapping on overflow
        let a = _mm_setr_epi32(i32::MAX, -1, i32::MAX, -2);
        let b = _mm_setr_epi32(i32::MIN, 1, i32::MIN, 2);
        let expected = _mm_setr_epi32(i32::MIN, i32::MIN + 1, i32::MAX, i32::MAX - 1);
        let r = _mm_hsub_epi32(a, b);
        assert_eq_m128i(r, expected);
    }
    test_mm_hsub_epi32();

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_maddubs_epi16() {
        let a = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm_setr_epi8(4, 63, 4, 3, 24, 12, 6, 19, 12, 5, 5, 10, 4, 1, 8, 0);
        let expected = _mm_setr_epi16(130, 24, 192, 194, 158, 175, 66, 120);
        let r = _mm_maddubs_epi16(a, b);
        assert_eq_m128i(r, expected);

        // Test widening and saturation
        let a = _mm_setr_epi8(
            u8::MAX as i8,
            u8::MAX as i8,
            u8::MAX as i8,
            u8::MAX as i8,
            u8::MAX as i8,
            u8::MAX as i8,
            100,
            100,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let b = _mm_setr_epi8(
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            50,
            15,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let expected = _mm_setr_epi16(i16::MAX, -255, i16::MIN, 6500, 0, 0, 0, 0);
        let r = _mm_maddubs_epi16(a, b);
        assert_eq_m128i(r, expected);
    }
    test_mm_maddubs_epi16();

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_mulhrs_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 4, 3, 32767, -1, -32768, 1);
        let expected = _mm_setr_epi16(0, 0, 0, 0, 5, 0, -7, 0);
        let r = _mm_mulhrs_epi16(a, b);
        assert_eq_m128i(r, expected);

        // Test extreme values
        let a = _mm_setr_epi16(i16::MAX, i16::MIN, i16::MIN, 0, 0, 0, 0, 0);
        let b = _mm_setr_epi16(i16::MAX, i16::MIN, i16::MAX, 0, 0, 0, 0, 0);
        let expected = _mm_setr_epi16(i16::MAX - 1, i16::MIN, -i16::MAX, 0, 0, 0, 0, 0);
        let r = _mm_mulhrs_epi16(a, b);
        assert_eq_m128i(r, expected);
    }
    test_mm_mulhrs_epi16();

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_sign_epi8() {
        let a = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -14, -15, 16);
        let b = _mm_setr_epi8(4, 63, -4, 3, 24, 12, -6, -19, 12, 5, -5, 10, 4, 1, -8, 0);
        let expected = _mm_setr_epi8(1, 2, -3, 4, 5, 6, -7, -8, 9, 10, -11, 12, 13, -14, 15, 0);
        let r = _mm_sign_epi8(a, b);
        assert_eq_m128i(r, expected);
    }
    test_mm_sign_epi8();

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_sign_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, -5, -6, 7, 8);
        let b = _mm_setr_epi16(4, 128, 0, 3, 1, -1, -2, 1);
        let expected = _mm_setr_epi16(1, 2, 0, 4, -5, 6, -7, 8);
        let r = _mm_sign_epi16(a, b);
        assert_eq_m128i(r, expected);
    }
    test_mm_sign_epi16();

    #[target_feature(enable = "ssse3")]
    unsafe fn test_mm_sign_epi32() {
        let a = _mm_setr_epi32(-1, 2, 3, 4);
        let b = _mm_setr_epi32(1, -1, 1, 0);
        let expected = _mm_setr_epi32(-1, -2, 3, 0);
        let r = _mm_sign_epi32(a, b);
        assert_eq_m128i(r, expected);
    }
    test_mm_sign_epi32();
}

#[track_caller]
#[target_feature(enable = "sse")]
unsafe fn assert_eq_m128(a: __m128, b: __m128) {
    let r = _mm_cmpeq_ps(a, b);
    if _mm_movemask_ps(r) != 0b1111 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[track_caller]
#[target_feature(enable = "sse2")]
unsafe fn assert_eq_m128d(a: __m128d, b: __m128d) {
    if _mm_movemask_pd(_mm_cmpeq_pd(a, b)) != 0b11 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[track_caller]
#[target_feature(enable = "sse2")]
pub unsafe fn assert_eq_m128i(a: __m128i, b: __m128i) {
    assert_eq!(transmute::<_, [u64; 2]>(a), transmute::<_, [u64; 2]>(b))
}
