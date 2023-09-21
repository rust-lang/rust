// Ignore everything except x86 and x86_64
// Any additional target are added to CI should be ignored here
//@ignore-target-aarch64
//@ignore-target-arm
//@ignore-target-avr
//@ignore-target-s390x
//@ignore-target-thumbv7em
//@ignore-target-wasm32
//@compile-flags: -C target-feature=+avx512vpopcntdq,+avx512f,+avx512vl

#![feature(avx512_target_feature)]
#![feature(stdsimd)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::transmute;

fn main() {
    assert!(is_x86_feature_detected!("avx512vpopcntdq"));
    assert!(is_x86_feature_detected!("avx512f"));
    assert!(is_x86_feature_detected!("avx512vl"));

    unsafe {
        test_avx512vpopcntdq();
    }
}

#[target_feature(enable = "avx512vpopcntdq,avx512f,avx512vl")]
unsafe fn test_avx512vpopcntdq() {
    // Mostly copied from library/stdarch/crates/core_arch/src/x86/avx512vpopcntdq.rs

    #[target_feature(enable = "avx512vpopcntdq,avx512f")]
    unsafe fn test_mm512_popcnt_epi32() {
        let test_data = _mm512_set_epi32(
            0,
            1,
            -1,
            2,
            7,
            0xFF_FE,
            0x7F_FF_FF_FF,
            -100,
            0x40_00_00_00,
            103,
            371,
            552,
            432_948,
            818_826_998,
            255,
            256,
        );
        let actual_result = _mm512_popcnt_epi32(test_data);
        let reference_result =
            _mm512_set_epi32(0, 1, 32, 1, 3, 15, 31, 28, 1, 5, 6, 3, 10, 17, 8, 1);
        assert_eq_m512i(actual_result, reference_result);
    }
    test_mm512_popcnt_epi32();

    #[target_feature(enable = "avx512vpopcntdq,avx512f,avx512vl")]
    unsafe fn test_mm256_popcnt_epi32() {
        let test_data = _mm256_set_epi32(0, 1, -1, 2, 7, 0xFF_FE, 0x7F_FF_FF_FF, -100);
        let actual_result = _mm256_popcnt_epi32(test_data);
        let reference_result = _mm256_set_epi32(0, 1, 32, 1, 3, 15, 31, 28);
        assert_eq_m256i(actual_result, reference_result);
    }
    test_mm256_popcnt_epi32();

    #[target_feature(enable = "avx512vpopcntdq,avx512f,avx512vl")]
    unsafe fn test_mm_popcnt_epi32() {
        let test_data = _mm_set_epi32(0, 1, -1, -100);
        let actual_result = _mm_popcnt_epi32(test_data);
        let reference_result = _mm_set_epi32(0, 1, 32, 28);
        assert_eq_m128i(actual_result, reference_result);
    }
    test_mm_popcnt_epi32();

    #[target_feature(enable = "avx512vpopcntdq,avx512f")]
    unsafe fn test_mm512_popcnt_epi64() {
        let test_data = _mm512_set_epi64(0, 1, -1, 2, 7, 0xFF_FE, 0x7F_FF_FF_FF_FF_FF_FF_FF, -100);
        let actual_result = _mm512_popcnt_epi64(test_data);
        let reference_result = _mm512_set_epi64(0, 1, 64, 1, 3, 15, 63, 60);
        assert_eq_m512i(actual_result, reference_result);
    }
    test_mm512_popcnt_epi64();

    #[target_feature(enable = "avx512vpopcntdq,avx512vl")]
    unsafe fn test_mm256_popcnt_epi64() {
        let test_data = _mm256_set_epi64x(0, 1, -1, -100);
        let actual_result = _mm256_popcnt_epi64(test_data);
        let reference_result = _mm256_set_epi64x(0, 1, 64, 60);
        assert_eq_m256i(actual_result, reference_result);
    }
    test_mm256_popcnt_epi64();

    #[target_feature(enable = "avx512vpopcntdq,avx512vl")]
    unsafe fn test_mm_popcnt_epi64() {
        let test_data = _mm_set_epi64x(0, 1);
        let actual_result = _mm_popcnt_epi64(test_data);
        let reference_result = _mm_set_epi64x(0, 1);
        assert_eq_m128i(actual_result, reference_result);
        let test_data = _mm_set_epi64x(-1, -100);
        let actual_result = _mm_popcnt_epi64(test_data);
        let reference_result = _mm_set_epi64x(64, 60);
        assert_eq_m128i(actual_result, reference_result);
    }
    test_mm_popcnt_epi64();
}

#[track_caller]
unsafe fn assert_eq_m512i(a: __m512i, b: __m512i) {
    assert_eq!(transmute::<_, [i32; 16]>(a), transmute::<_, [i32; 16]>(b))
}

#[track_caller]
unsafe fn assert_eq_m256i(a: __m256i, b: __m256i) {
    assert_eq!(transmute::<_, [u64; 4]>(a), transmute::<_, [u64; 4]>(b))
}

#[track_caller]
unsafe fn assert_eq_m128i(a: __m128i, b: __m128i) {
    assert_eq!(transmute::<_, [u64; 2]>(a), transmute::<_, [u64; 2]>(b))
}
