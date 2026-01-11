// We're testing x86 target specific features
//@only-target: x86_64 i686
//@compile-flags: -C target-feature=+avx512f,+avx512vl,+avx512bitalg,+avx512vpopcntdq,+avx512vnni

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
    assert!(is_x86_feature_detected!("avx512vnni"));

    unsafe {
        test_avx512();
        test_avx512bitalg();
        test_avx512vpopcntdq();
        test_avx512ternarylogic();
        test_avx512vnni();
    }
}

#[target_feature(enable = "avx512bw")]
unsafe fn test_avx512() {
    #[target_feature(enable = "avx512bw")]
    unsafe fn test_mm512_sad_epu8() {
        let a = _mm512_set_epi8(
            71, 70, 69, 68, 67, 66, 65, 64, //
            55, 54, 53, 52, 51, 50, 49, 48, //
            47, 46, 45, 44, 43, 42, 41, 40, //
            39, 38, 37, 36, 35, 34, 33, 32, //
            31, 30, 29, 28, 27, 26, 25, 24, //
            23, 22, 21, 20, 19, 18, 17, 16, //
            15, 14, 13, 12, 11, 10, 9, 8, //
            7, 6, 5, 4, 3, 2, 1, 0, //
        );

        //  `d` is the absolute difference with the corresponding row in `a`.
        let b = _mm512_set_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, // lane 7 (d = 8)
            62, 61, 60, 59, 58, 57, 56, 55, // lane 6 (d = 7)
            53, 52, 51, 50, 49, 48, 47, 46, // lane 5 (d = 6)
            44, 43, 42, 41, 40, 39, 38, 37, // lane 4 (d = 5)
            35, 34, 33, 32, 31, 30, 29, 28, // lane 3 (d = 4)
            26, 25, 24, 23, 22, 21, 20, 19, // lane 2 (d = 3)
            17, 16, 15, 14, 13, 12, 11, 10, // lane 1 (d = 2)
            8, 7, 6, 5, 4, 3, 2, 1, // lane 0 (d = 1)
        );

        let r = _mm512_sad_epu8(a, b);
        let e = _mm512_set_epi64(64, 56, 48, 40, 32, 24, 16, 8);

        assert_eq_m512i(r, e);
    }
    test_mm512_sad_epu8();

    #[target_feature(enable = "avx512bw")]
    unsafe fn test_mm512_maddubs_epi16() {
        // `a` is interpreted as `u8x16`, but `_mm512_set_epi8` expects `i8`, so we have to cast.
        #[rustfmt::skip]
        let a = _mm512_set_epi8(
            255u8 as i8, 255u8 as i8,  60,  50, 100, 100, 255u8 as i8, 200u8 as i8,
            255u8 as i8, 200u8 as i8, 200u8 as i8, 100,  60,  50,  20,  10,

            255u8 as i8, 255u8 as i8,  60,  50, 100, 100, 255u8 as i8, 200u8 as i8,
            255u8 as i8, 200u8 as i8, 200u8 as i8, 100,  60,  50,  20,  10,

            255u8 as i8, 255u8 as i8,  60,  50, 100, 100, 255u8 as i8, 200u8 as i8,
            255u8 as i8, 200u8 as i8, 200u8 as i8, 100,  60,  50,  20,  10,

            255u8 as i8, 255u8 as i8,  60,  50, 100, 100, 255u8 as i8, 200u8 as i8,
            255u8 as i8, 200u8 as i8, 200u8 as i8, 100,  60,  50,  20,  10,
        );

        let b = _mm512_set_epi8(
            64, 64, -2, 1, 100, 100, -128, -128, //
            127, 127, -1, 1, 2, 2, 1, 1, //
            64, 64, -2, 1, 100, 100, -128, -128, //
            127, 127, -1, 1, 2, 2, 1, 1, //
            64, 64, -2, 1, 100, 100, -128, -128, //
            127, 127, -1, 1, 2, 2, 1, 1, //
            64, 64, -2, 1, 100, 100, -128, -128, //
            127, 127, -1, 1, 2, 2, 1, 1, //
        );

        let r = _mm512_maddubs_epi16(a, b);

        let e = _mm512_set_epi16(
            32640, -70, 20000, -32768, 32767, -100, 220, 30, //
            32640, -70, 20000, -32768, 32767, -100, 220, 30, //
            32640, -70, 20000, -32768, 32767, -100, 220, 30, //
            32640, -70, 20000, -32768, 32767, -100, 220, 30, //
        );

        assert_eq_m512i(r, e);
    }
    test_mm512_maddubs_epi16();

    #[target_feature(enable = "avx512f")]
    unsafe fn test_mm512_permutexvar_epi32() {
        let a = _mm512_set_epi32(
            15, 14, 13, 12, //
            11, 10, 9, 8, //
            7, 6, 5, 4, //
            3, 2, 1, 0, //
        );

        let idx_identity = _mm512_set_epi32(
            15, 14, 13, 12, //
            11, 10, 9, 8, //
            7, 6, 5, 4, //
            3, 2, 1, 0, //
        );
        let r_id = _mm512_permutexvar_epi32(idx_identity, a);
        assert_eq_m512i(r_id, a);

        // Test some out-of-bounds indices.
        let edge_cases = _mm512_set_epi32(
            0,
            -1,
            -128,
            i32::MIN,
            15,
            16,
            128,
            i32::MAX,
            0,
            -1,
            -128,
            i32::MIN,
            15,
            16,
            128,
            i32::MAX,
        );

        let r = _mm512_permutexvar_epi32(edge_cases, a);

        let e = _mm512_set_epi32(0, 15, 0, 0, 15, 0, 0, 15, 0, 15, 0, 0, 15, 0, 0, 15);

        assert_eq_m512i(r, e);
    }
    test_mm512_permutexvar_epi32();

    #[target_feature(enable = "avx512bw")]
    unsafe fn test_mm512_shuffle_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        #[rustfmt::skip]
        let b = _mm512_set_epi8(-1, 127, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                -1, 127, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                -1, 127, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                -1, 127, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let r = _mm512_shuffle_epi8(a, b);
        // `_mm512_set_epi8` sets the bytes in inverse order (?!?), so the indices in `b` seem to
        // index from the *back* of the corresponding 16-byte block in `a`.
        #[rustfmt::skip]
        let e = _mm512_set_epi8(0, 0, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                                0, 16, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
                                0, 32, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
                                0, 48, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62);
        assert_eq_m512i(r, e);
    }
    test_mm512_shuffle_epi8();
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

#[target_feature(enable = "avx512f,avx512vl")]
unsafe fn test_avx512ternarylogic() {
    #[target_feature(enable = "avx512f")]
    unsafe fn test_mm512_ternarylogic_epi32() {
        let a = _mm512_set4_epi32(0b100, 0b110, 0b001, 0b101);
        let b = _mm512_set4_epi32(0b010, 0b011, 0b001, 0b110);
        let c = _mm512_set4_epi32(0b001, 0b000, 0b001, 0b111);

        // Identity of A.
        let r = _mm512_ternarylogic_epi32::<0b1111_0000>(a, b, c);
        assert_eq_m512i(r, a);

        // Bitwise xor.
        let r = _mm512_ternarylogic_epi32::<0b10010110>(a, b, c);
        let e = _mm512_set4_epi32(0b111, 0b101, 0b001, 0b100);
        assert_eq_m512i(r, e);

        // Majority (2 or more bits set).
        let r = _mm512_ternarylogic_epi32::<0b1110_1000>(a, b, c);
        let e = _mm512_set4_epi32(0b000, 0b010, 0b001, 0b111);
        assert_eq_m512i(r, e);
    }
    test_mm512_ternarylogic_epi32();

    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_ternarylogic_epi32() {
        let _mm256_set4_epi32 = |a, b, c, d| _mm256_setr_epi32(a, b, c, d, a, b, c, d);

        let a = _mm256_set4_epi32(0b100, 0b110, 0b001, 0b101);
        let b = _mm256_set4_epi32(0b010, 0b011, 0b001, 0b110);
        let c = _mm256_set4_epi32(0b001, 0b000, 0b001, 0b111);

        // Identity of A.
        let r = _mm256_ternarylogic_epi32::<0b1111_0000>(a, b, c);
        assert_eq_m256i(r, a);

        // Bitwise xor.
        let r = _mm256_ternarylogic_epi32::<0b10010110>(a, b, c);
        let e = _mm256_set4_epi32(0b111, 0b101, 0b001, 0b100);
        assert_eq_m256i(r, e);

        // Majority (2 or more bits set).
        let r = _mm256_ternarylogic_epi32::<0b1110_1000>(a, b, c);
        let e = _mm256_set4_epi32(0b000, 0b010, 0b001, 0b111);
        assert_eq_m256i(r, e);
    }
    test_mm256_ternarylogic_epi32();

    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn test_mm_ternarylogic_epi32() {
        let a = _mm_setr_epi32(0b100, 0b110, 0b001, 0b101);
        let b = _mm_setr_epi32(0b010, 0b011, 0b001, 0b110);
        let c = _mm_setr_epi32(0b001, 0b000, 0b001, 0b111);

        // Identity of A.
        let r = _mm_ternarylogic_epi32::<0b1111_0000>(a, b, c);
        assert_eq_m128i(r, a);

        // Bitwise xor.
        let r = _mm_ternarylogic_epi32::<0b10010110>(a, b, c);
        let e = _mm_setr_epi32(0b111, 0b101, 0b001, 0b100);
        assert_eq_m128i(r, e);

        // Majority (2 or more bits set).
        let r = _mm_ternarylogic_epi32::<0b1110_1000>(a, b, c);
        let e = _mm_setr_epi32(0b000, 0b010, 0b001, 0b111);
        assert_eq_m128i(r, e);
    }
    test_mm_ternarylogic_epi32();
}

#[target_feature(enable = "avx512vnni")]
unsafe fn test_avx512vnni() {
    #[target_feature(enable = "avx512vnni")]
    unsafe fn test_mm512_dpbusd_epi32() {
        const SRC: [i32; 16] = [
            1,
            // Test that addition with the `src` element uses wrapping arithmetic.
            i32::MAX,
            i32::MIN,
            0,
            0,
            7,
            12345,
            -9876,
            0x01020304,
            -1,
            42,
            0,
            1_000_000_000,
            -1_000_000_000,
            17,
            -17,
        ];

        // The `A` array must be interpreted as a sequence of unsigned 8-bit integers. Setting
        // the high bit of a byte tests that this is implemented correctly.
        const A: [i32; 16] = [
            0x01010101,
            i32::from_le_bytes([1; 4]),
            i32::from_le_bytes([1; 4]),
            i32::from_le_bytes([u8::MAX; 4]),
            i32::from_le_bytes([u8::MAX; 4]),
            0x02_80_01_FF,
            0x00_FF_00_FF,
            0x7F_80_FF_01,
            0x10_20_30_40,
            0xDE_AD_BE_EFu32 as i32,
            0x00_00_00_FF,
            0x12_34_56_78,
            0xFF_00_FF_00u32 as i32,
            0x01_02_03_04,
            0xAA_55_AA_55u32 as i32,
            0x11_22_33_44,
        ];

        // The `B` array must be interpreted as a sequence of signed 8-bit integers. Setting
        // the high bit of a byte tests that this is implemented correctly.
        const B: [i32; 16] = [
            0x01010101,
            i32::from_le_bytes([1; 4]),
            i32::from_le_bytes([(-1i8).cast_unsigned(); 4]),
            i32::from_le_bytes([i8::MAX.cast_unsigned(); 4]),
            i32::from_le_bytes([i8::MIN.cast_unsigned(); 4]),
            0xFF_01_80_7Fu32 as i32,
            0x01_FF_01_FF,
            0x80_7F_00_FFu32 as i32,
            0x7F_01_FF_80u32 as i32,
            0x01_02_03_04,
            0xFF_FF_FF_FFu32 as i32,
            0x80_00_7F_FFu32 as i32,
            0x7F_80_7F_80u32 as i32,
            0x40_C0_20_E0u32 as i32,
            0x00_01_02_03,
            0x7F_7E_80_81u32 as i32,
        ];

        const DST: [i32; 16] = [
            5,
            i32::MAX.wrapping_add(4),
            i32::MIN.wrapping_add(-4),
            129540,
            -130560,
            32390,
            11835,
            -9877,
            16902884,
            2093,
            -213,
            8498,
            1000064770,
            -1000000096,
            697,
            -8738,
        ];

        let src = _mm512_loadu_si512(SRC.as_ptr().cast::<__m512i>());
        let a = _mm512_loadu_si512(A.as_ptr().cast::<__m512i>());
        let b = _mm512_loadu_si512(B.as_ptr().cast::<__m512i>());
        let dst = _mm512_loadu_si512(DST.as_ptr().cast::<__m512i>());

        assert_eq_m512i(_mm512_dpbusd_epi32(src, a, b), dst);
    }
    test_mm512_dpbusd_epi32();
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
