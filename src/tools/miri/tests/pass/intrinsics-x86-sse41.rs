// Ignore everything except x86 and x86_64
// Any additional target are added to CI should be ignored here
// (We cannot use `cfg`-based tricks here since the `target-feature` flags below only work on x86.)
//@ignore-target-aarch64
//@ignore-target-arm
//@ignore-target-avr
//@ignore-target-s390x
//@ignore-target-thumbv7em
//@ignore-target-wasm32
//@compile-flags: -C target-feature=+sse4.1

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::transmute;

fn main() {
    assert!(is_x86_feature_detected!("sse4.1"));

    unsafe {
        test_sse41();
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn test_sse41() {
    // Mostly copied from library/stdarch/crates/core_arch/src/x86/sse41.rs

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_insert_ps() {
        let a = _mm_set1_ps(1.0);
        let b = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let r = _mm_insert_ps::<0b11_00_1100>(a, b);
        let e = _mm_setr_ps(4.0, 1.0, 0.0, 0.0);
        assert_eq_m128(r, e);

        // Zeroing takes precedence over copied value
        let a = _mm_set1_ps(1.0);
        let b = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let r = _mm_insert_ps::<0b11_00_0001>(a, b);
        let e = _mm_setr_ps(0.0, 1.0, 1.0, 1.0);
        assert_eq_m128(r, e);
    }
    test_mm_insert_ps();

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_packus_epi32() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let b = _mm_setr_epi32(-1, -2, -3, -4);
        let r = _mm_packus_epi32(a, b);
        let e = _mm_setr_epi16(1, 2, 3, 4, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }
    test_mm_packus_epi32();

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_dp_pd() {
        let a = _mm_setr_pd(2.0, 3.0);
        let b = _mm_setr_pd(1.0, 4.0);
        let e = _mm_setr_pd(14.0, 0.0);
        assert_eq_m128d(_mm_dp_pd::<0b00110001>(a, b), e);
    }
    test_mm_dp_pd();

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_dp_ps() {
        let a = _mm_setr_ps(2.0, 3.0, 1.0, 10.0);
        let b = _mm_setr_ps(1.0, 4.0, 0.5, 10.0);
        let e = _mm_setr_ps(14.5, 0.0, 14.5, 0.0);
        assert_eq_m128(_mm_dp_ps::<0b01110101>(a, b), e);
    }
    test_mm_dp_ps();

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_floor_sd() {
        let a = _mm_setr_pd(2.5, 4.5);
        let b = _mm_setr_pd(-1.5, -3.5);
        let r = _mm_floor_sd(a, b);
        let e = _mm_setr_pd(-2.0, 4.5);
        assert_eq_m128d(r, e);
    }
    test_mm_floor_sd();

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_floor_ss() {
        let a = _mm_setr_ps(2.5, 4.5, 8.5, 16.5);
        let b = _mm_setr_ps(-1.5, -3.5, -7.5, -15.5);
        let r = _mm_floor_ss(a, b);
        let e = _mm_setr_ps(-2.0, 4.5, 8.5, 16.5);
        assert_eq_m128(r, e);
    }
    test_mm_floor_ss();

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_ceil_sd() {
        let a = _mm_setr_pd(1.5, 3.5);
        let b = _mm_setr_pd(-2.5, -4.5);
        let r = _mm_ceil_sd(a, b);
        let e = _mm_setr_pd(-2.0, 3.5);
        assert_eq_m128d(r, e);
    }
    test_mm_ceil_sd();

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_ceil_ss() {
        let a = _mm_setr_ps(1.5, 3.5, 7.5, 15.5);
        let b = _mm_setr_ps(-2.5, -4.5, -8.5, -16.5);
        let r = _mm_ceil_ss(a, b);
        let e = _mm_setr_ps(-2.0, 3.5, 7.5, 15.5);
        assert_eq_m128(r, e);
    }
    test_mm_ceil_ss();

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_round_sd() {
        let a = _mm_setr_pd(1.5, 3.5);
        let b = _mm_setr_pd(-2.5, -4.5);
        let r = _mm_round_sd::<_MM_FROUND_TO_NEAREST_INT>(a, b);
        let e = _mm_setr_pd(-2.0, 3.5);
        assert_eq_m128d(r, e);

        let a = _mm_setr_pd(1.5, 3.5);
        let b = _mm_setr_pd(-2.5, -4.5);
        let r = _mm_round_sd::<_MM_FROUND_TO_NEG_INF>(a, b);
        let e = _mm_setr_pd(-3.0, 3.5);
        assert_eq_m128d(r, e);

        let a = _mm_setr_pd(1.5, 3.5);
        let b = _mm_setr_pd(-2.5, -4.5);
        let r = _mm_round_sd::<_MM_FROUND_TO_POS_INF>(a, b);
        let e = _mm_setr_pd(-2.0, 3.5);
        assert_eq_m128d(r, e);

        let a = _mm_setr_pd(1.5, 3.5);
        let b = _mm_setr_pd(-2.5, -4.5);
        let r = _mm_round_sd::<_MM_FROUND_TO_ZERO>(a, b);
        let e = _mm_setr_pd(-2.0, 3.5);
        assert_eq_m128d(r, e);

        // Assume round-to-nearest by default
        let a = _mm_setr_pd(1.5, 3.5);
        let b = _mm_setr_pd(-2.5, -4.5);
        let r = _mm_round_sd::<_MM_FROUND_CUR_DIRECTION>(a, b);
        let e = _mm_setr_pd(-2.0, 3.5);
        assert_eq_m128d(r, e);
    }
    test_mm_round_sd();

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_round_ss() {
        let a = _mm_setr_ps(1.5, 3.5, 7.5, 15.5);
        let b = _mm_setr_ps(-1.75, -4.5, -8.5, -16.5);
        let r = _mm_round_ss::<_MM_FROUND_TO_NEAREST_INT>(a, b);
        let e = _mm_setr_ps(-2.0, 3.5, 7.5, 15.5);
        assert_eq_m128(r, e);

        let a = _mm_setr_ps(1.5, 3.5, 7.5, 15.5);
        let b = _mm_setr_ps(-1.75, -4.5, -8.5, -16.5);
        let r = _mm_round_ss::<_MM_FROUND_TO_NEG_INF>(a, b);
        let e = _mm_setr_ps(-2.0, 3.5, 7.5, 15.5);
        assert_eq_m128(r, e);

        let a = _mm_setr_ps(1.5, 3.5, 7.5, 15.5);
        let b = _mm_setr_ps(-1.75, -4.5, -8.5, -16.5);
        let r = _mm_round_ss::<_MM_FROUND_TO_POS_INF>(a, b);
        let e = _mm_setr_ps(-1.0, 3.5, 7.5, 15.5);
        assert_eq_m128(r, e);

        let a = _mm_setr_ps(1.5, 3.5, 7.5, 15.5);
        let b = _mm_setr_ps(-1.75, -4.5, -8.5, -16.5);
        let r = _mm_round_ss::<_MM_FROUND_TO_ZERO>(a, b);
        let e = _mm_setr_ps(-1.0, 3.5, 7.5, 15.5);
        assert_eq_m128(r, e);

        // Assume round-to-nearest by default
        let a = _mm_setr_ps(1.5, 3.5, 7.5, 15.5);
        let b = _mm_setr_ps(-1.75, -4.5, -8.5, -16.5);
        let r = _mm_round_ss::<_MM_FROUND_CUR_DIRECTION>(a, b);
        let e = _mm_setr_ps(-2.0, 3.5, 7.5, 15.5);
        assert_eq_m128(r, e);
    }
    test_mm_round_ss();

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_minpos_epu16() {
        let a = _mm_setr_epi16(23, 18, 44, 97, 50, 13, 67, 66);
        let r = _mm_minpos_epu16(a);
        let e = _mm_setr_epi16(13, 5, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);

        let a = _mm_setr_epi16(0, 18, 44, 97, 50, 13, 67, 66);
        let r = _mm_minpos_epu16(a);
        let e = _mm_setr_epi16(0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);

        // Case where the minimum value is repeated
        let a = _mm_setr_epi16(23, 18, 44, 97, 50, 13, 67, 13);
        let r = _mm_minpos_epu16(a);
        let e = _mm_setr_epi16(13, 5, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(r, e);
    }
    test_mm_minpos_epu16();

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_mpsadbw_epu8() {
        let a = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

        let r = _mm_mpsadbw_epu8::<0b000>(a, a);
        let e = _mm_setr_epi16(0, 4, 8, 12, 16, 20, 24, 28);
        assert_eq_m128i(r, e);

        let r = _mm_mpsadbw_epu8::<0b001>(a, a);
        let e = _mm_setr_epi16(16, 12, 8, 4, 0, 4, 8, 12);
        assert_eq_m128i(r, e);

        let r = _mm_mpsadbw_epu8::<0b100>(a, a);
        let e = _mm_setr_epi16(16, 20, 24, 28, 32, 36, 40, 44);
        assert_eq_m128i(r, e);

        let r = _mm_mpsadbw_epu8::<0b101>(a, a);
        let e = _mm_setr_epi16(0, 4, 8, 12, 16, 20, 24, 28);
        assert_eq_m128i(r, e);

        let r = _mm_mpsadbw_epu8::<0b111>(a, a);
        let e = _mm_setr_epi16(32, 28, 24, 20, 16, 12, 8, 4);
        assert_eq_m128i(r, e);
    }
    test_mm_mpsadbw_epu8();

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_testz_si128() {
        let a = _mm_set1_epi8(1);
        let mask = _mm_set1_epi8(0);
        let r = _mm_testz_si128(a, mask);
        assert_eq!(r, 1);

        let a = _mm_set1_epi8(0b101);
        let mask = _mm_set1_epi8(0b110);
        let r = _mm_testz_si128(a, mask);
        assert_eq!(r, 0);

        let a = _mm_set1_epi8(0b011);
        let mask = _mm_set1_epi8(0b100);
        let r = _mm_testz_si128(a, mask);
        assert_eq!(r, 1);
    }
    test_mm_testz_si128();

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_testc_si128() {
        let a = _mm_set1_epi8(-1);
        let mask = _mm_set1_epi8(0);
        let r = _mm_testc_si128(a, mask);
        assert_eq!(r, 1);

        let a = _mm_set1_epi8(0b101);
        let mask = _mm_set1_epi8(0b110);
        let r = _mm_testc_si128(a, mask);
        assert_eq!(r, 0);

        let a = _mm_set1_epi8(0b101);
        let mask = _mm_set1_epi8(0b100);
        let r = _mm_testc_si128(a, mask);
        assert_eq!(r, 1);
    }
    test_mm_testc_si128();

    #[target_feature(enable = "sse4.1")]
    unsafe fn test_mm_testnzc_si128() {
        let a = _mm_set1_epi8(0);
        let mask = _mm_set1_epi8(1);
        let r = _mm_testnzc_si128(a, mask);
        assert_eq!(r, 0);

        let a = _mm_set1_epi8(-1);
        let mask = _mm_set1_epi8(0);
        let r = _mm_testnzc_si128(a, mask);
        assert_eq!(r, 0);

        let a = _mm_set1_epi8(0b101);
        let mask = _mm_set1_epi8(0b110);
        let r = _mm_testnzc_si128(a, mask);
        assert_eq!(r, 1);

        let a = _mm_set1_epi8(0b101);
        let mask = _mm_set1_epi8(0b101);
        let r = _mm_testnzc_si128(a, mask);
        assert_eq!(r, 0);
    }
    test_mm_testnzc_si128();
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
pub unsafe fn assert_eq_m128d(a: __m128d, b: __m128d) {
    if _mm_movemask_pd(_mm_cmpeq_pd(a, b)) != 0b11 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[track_caller]
#[target_feature(enable = "sse2")]
pub unsafe fn assert_eq_m128i(a: __m128i, b: __m128i) {
    assert_eq!(transmute::<_, [u64; 2]>(a), transmute::<_, [u64; 2]>(b))
}
