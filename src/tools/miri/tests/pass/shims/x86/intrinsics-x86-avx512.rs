// We're testing x86 target specific features
//@only-target: x86_64 i686
//@compile-flags: -C target-feature=+avx512f,+avx512vl,+avx512bitalg,+avx512vpopcntdq

#![feature(stdarch_x86_avx512)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::transmute;

fn main() {
    assert!(is_x86_feature_detected!("avx512f"));
    assert!(is_x86_feature_detected!("avx512vl"));
    assert!(is_x86_feature_detected!("avx512bitalg"));
    assert!(is_x86_feature_detected!("avx512vpopcntdq"));

    unsafe {
        test_avx512bitalg();
        test_avx512vpopcntdq();
    }
}

// Some of the constants in the tests below are just bit patterns. They should not
// be interpreted as integers; signedness does not make sense for them, but
// __mXXXi happens to be defined in terms of signed integers.
#[allow(overflowing_literals)]
#[target_feature(enable = "avx512bitalg,avx512f,avx512vl")]
unsafe fn test_avx512bitalg() {
    // Mostly copied from library/stdarch/crates/core_arch/src/x86/avx512bitalg.rs

    #[target_feature(enable = "avx512bitalg,avx512f")]
    unsafe fn test_mm512_popcnt_epi16() {
        let test_data = _mm512_set_epi16(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, 0x1_FF, 0x3_FF, 0x7_FF, 0xF_FF, 0x1F_FF,
            0x3F_FF, 0x7F_FF, 0xFF_FF, -1, -100, 255, 256, 2, 4, 8, 16, 32, 64, 128, 256, 512,
            1024, 2048,
        );
        let actual_result = _mm512_popcnt_epi16(test_data);
        let reference_result = _mm512_set_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 12, 8, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
        );
        assert_eq_m512i(actual_result, reference_result);
    }
    test_mm512_popcnt_epi16();

    #[target_feature(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm256_popcnt_epi16() {
        let test_data = _mm256_set_epi16(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, 0x1_FF, 0x3_FF, 0x7_FF, 0xF_FF, 0x1F_FF,
            0x3F_FF, 0x7F_FF,
        );
        let actual_result = _mm256_popcnt_epi16(test_data);
        let reference_result =
            _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m256i(actual_result, reference_result);
    }
    test_mm256_popcnt_epi16();

    #[target_feature(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm_popcnt_epi16() {
        let test_data = _mm_set_epi16(0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F);
        let actual_result = _mm_popcnt_epi16(test_data);
        let reference_result = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq_m128i(actual_result, reference_result);
    }
    test_mm_popcnt_epi16();

    #[target_feature(enable = "avx512bitalg,avx512f")]
    unsafe fn test_mm512_popcnt_epi8() {
        let test_data = _mm512_set_epi8(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, -1, 2, 4, 8, 16, 32, 64, 128, 171, 206, 100,
            217, 109, 253, 190, 177, 254, 179, 215, 230, 68, 201, 172, 183, 154, 84, 56, 227, 189,
            140, 35, 117, 219, 169, 226, 170, 13, 22, 159, 251, 73, 121, 143, 145, 85, 91, 137, 90,
            225, 21, 249, 211, 155, 228, 70,
        );
        let actual_result = _mm512_popcnt_epi8(test_data);
        let reference_result = _mm512_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 1, 1, 1, 1, 1, 1, 1, 5, 5, 3, 5, 5, 7, 6, 4, 7, 5, 6, 5,
            2, 4, 4, 6, 4, 3, 3, 5, 6, 3, 3, 5, 6, 4, 4, 4, 3, 3, 6, 7, 3, 5, 5, 3, 4, 5, 3, 4, 4,
            3, 6, 5, 5, 4, 3,
        );
        assert_eq_m512i(actual_result, reference_result);
    }
    test_mm512_popcnt_epi8();

    #[target_feature(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm256_popcnt_epi8() {
        let test_data = _mm256_set_epi8(
            0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, -1, 2, 4, 8, 16, 32, 64, 128, 171, 206, 100,
            217, 109, 253, 190, 177, 254, 179, 215, 230, 68, 201, 172,
        );
        let actual_result = _mm256_popcnt_epi8(test_data);
        let reference_result = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 1, 1, 1, 1, 1, 1, 1, 5, 5, 3, 5, 5, 7, 6, 4, 7, 5, 6, 5,
            2, 4, 4,
        );
        assert_eq_m256i(actual_result, reference_result);
    }
    test_mm256_popcnt_epi8();

    #[target_feature(enable = "avx512bitalg,avx512f,avx512vl")]
    unsafe fn test_mm_popcnt_epi8() {
        let test_data =
            _mm_set_epi8(0, 1, 3, 7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, -1, 2, 4, 8, 16, 32, 64);
        let actual_result = _mm_popcnt_epi8(test_data);
        let reference_result = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 1, 1, 1, 1, 1, 1);
        assert_eq_m128i(actual_result, reference_result);
    }
    test_mm_popcnt_epi8();
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
