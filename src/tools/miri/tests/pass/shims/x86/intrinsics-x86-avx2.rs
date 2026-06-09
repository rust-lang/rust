// We're testing x86 target specific features
//@only-target: x86_64 i686
//@compile-flags: -C target-feature=+avx2

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::transmute;

fn main() {
    assert!(is_x86_feature_detected!("avx2"));

    unsafe {
        test_avx2();
    }
}

#[target_feature(enable = "avx2")]
unsafe fn test_avx2() {
    // Mostly copied from library/stdarch/crates/core_arch/src/x86/avx2.rs

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_abs_epi32() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi32(
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
        );
        let r = _mm256_abs_epi32(a);
        #[rustfmt::skip]
        let e = _mm256_setr_epi32(
            0, 1, 1, i32::MAX,
            i32::MAX.wrapping_add(1), 100, 100, 32,
        );
        assert_eq_m256i(r, e);
    }
    test_mm256_abs_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_abs_epi16() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi16(
            0,  1, -1, 2, -2, 3, -3, 4,
            -4, 5, -5, i16::MAX, i16::MIN, 100, -100, -32,
        );
        let r = _mm256_abs_epi16(a);
        #[rustfmt::skip]
        let e = _mm256_setr_epi16(
            0, 1, 1, 2, 2, 3, 3, 4,
            4, 5, 5, i16::MAX, i16::MAX.wrapping_add(1), 100, 100, 32,
        );
        assert_eq_m256i(r, e);
    }
    test_mm256_abs_epi16();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_abs_epi8() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            0, 1, -1, 2, -2, 3, -3, 4,
            -4, 5, -5, i8::MAX, i8::MIN, 100, -100, -32,
            0, 1, -1, 2, -2, 3, -3, 4,
            -4, 5, -5, i8::MAX, i8::MIN, 100, -100, -32,
        );
        let r = _mm256_abs_epi8(a);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            0, 1, 1, 2, 2, 3, 3, 4,
            4, 5, 5, i8::MAX, i8::MAX.wrapping_add(1), 100, 100, 32,
            0, 1, 1, 2, 2, 3, 3, 4,
            4, 5, 5, i8::MAX, i8::MAX.wrapping_add(1), 100, 100, 32,
        );
        assert_eq_m256i(r, e);
    }
    test_mm256_abs_epi8();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_hadd_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_hadd_epi16(a, b);
        let e = _mm256_setr_epi16(4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8);
        assert_eq_m256i(r, e);

        // Test wrapping on overflow
        let a = _mm256_setr_epi16(
            i16::MAX,
            1,
            i16::MAX,
            2,
            i16::MAX,
            3,
            i16::MAX,
            4,
            i16::MAX,
            5,
            i16::MAX,
            6,
            i16::MAX,
            7,
            i16::MAX,
            8,
        );
        let b = _mm256_setr_epi16(
            i16::MIN,
            -1,
            i16::MIN,
            -2,
            i16::MIN,
            -3,
            i16::MIN,
            -4,
            i16::MIN,
            -5,
            i16::MIN,
            -6,
            i16::MIN,
            -7,
            i16::MIN,
            -8,
        );
        let expected = _mm256_setr_epi16(
            i16::MIN,
            i16::MIN + 1,
            i16::MIN + 2,
            i16::MIN + 3,
            i16::MAX,
            i16::MAX - 1,
            i16::MAX - 2,
            i16::MAX - 3,
            i16::MIN + 4,
            i16::MIN + 5,
            i16::MIN + 6,
            i16::MIN + 7,
            i16::MAX - 4,
            i16::MAX - 5,
            i16::MAX - 6,
            i16::MAX - 7,
        );
        let r = _mm256_hadd_epi16(a, b);
        assert_eq_m256i(r, expected);
    }
    test_mm256_hadd_epi16();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_hadd_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(4);
        let r = _mm256_hadd_epi32(a, b);
        let e = _mm256_setr_epi32(4, 4, 8, 8, 4, 4, 8, 8);
        assert_eq_m256i(r, e);

        // Test wrapping on overflow
        let a = _mm256_setr_epi32(i32::MAX, 1, i32::MAX, 2, i32::MAX, 3, i32::MAX, 4);
        let b = _mm256_setr_epi32(i32::MIN, -1, i32::MIN, -2, i32::MIN, -3, i32::MIN, -4);
        let expected = _mm256_setr_epi32(
            i32::MIN,
            i32::MIN + 1,
            i32::MAX,
            i32::MAX - 1,
            i32::MIN + 2,
            i32::MIN + 3,
            i32::MAX - 2,
            i32::MAX - 3,
        );
        let r = _mm256_hadd_epi32(a, b);
        assert_eq_m256i(r, expected);
    }
    test_mm256_hadd_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_hadds_epi16() {
        let a = _mm256_set1_epi16(2);
        let a = _mm256_insert_epi16::<0>(a, 0x7fff);
        let a = _mm256_insert_epi16::<1>(a, 1);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_hadds_epi16(a, b);
        let e = _mm256_setr_epi16(0x7FFF, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8);
        assert_eq_m256i(r, e);

        // Test saturating on overflow
        let a = _mm256_setr_epi16(
            i16::MAX,
            1,
            i16::MAX,
            2,
            i16::MAX,
            3,
            i16::MAX,
            4,
            i16::MAX,
            5,
            i16::MAX,
            6,
            i16::MAX,
            7,
            i16::MAX,
            8,
        );
        let b = _mm256_setr_epi16(
            i16::MIN,
            -1,
            i16::MIN,
            -2,
            i16::MIN,
            -3,
            i16::MIN,
            -4,
            i16::MIN,
            -5,
            i16::MIN,
            -6,
            i16::MIN,
            -7,
            i16::MIN,
            -8,
        );
        let expected = _mm256_setr_epi16(
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MIN,
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MIN,
            i16::MIN,
        );
        let r = _mm256_hadds_epi16(a, b);
        assert_eq_m256i(r, expected);
    }
    test_mm256_hadds_epi16();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_hsub_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_hsub_epi16(a, b);
        let e = _mm256_set1_epi16(0);
        assert_eq_m256i(r, e);

        // Test wrapping on overflow
        let a = _mm256_setr_epi16(
            i16::MAX,
            -1,
            i16::MAX,
            -2,
            i16::MAX,
            -3,
            i16::MAX,
            -4,
            i16::MAX,
            -5,
            i16::MAX,
            -6,
            i16::MAX,
            -7,
            i16::MAX,
            -8,
        );
        let b = _mm256_setr_epi16(
            i16::MIN,
            1,
            i16::MIN,
            2,
            i16::MIN,
            3,
            i16::MIN,
            4,
            i16::MIN,
            5,
            i16::MIN,
            6,
            i16::MIN,
            7,
            i16::MIN,
            8,
        );
        let expected = _mm256_setr_epi16(
            i16::MIN,
            i16::MIN + 1,
            i16::MIN + 2,
            i16::MIN + 3,
            i16::MAX,
            i16::MAX - 1,
            i16::MAX - 2,
            i16::MAX - 3,
            i16::MIN + 4,
            i16::MIN + 5,
            i16::MIN + 6,
            i16::MIN + 7,
            i16::MAX - 4,
            i16::MAX - 5,
            i16::MAX - 6,
            i16::MAX - 7,
        );
        let r = _mm256_hsub_epi16(a, b);
        assert_eq_m256i(r, expected);
    }
    test_mm256_hsub_epi16();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_hsub_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(4);
        let r = _mm256_hsub_epi32(a, b);
        let e = _mm256_set1_epi32(0);
        assert_eq_m256i(r, e);

        // Test wrapping on overflow
        let a = _mm256_setr_epi32(i32::MAX, -1, i32::MAX, -2, i32::MAX, -3, i32::MAX, -4);
        let b = _mm256_setr_epi32(i32::MIN, 1, i32::MIN, 2, i32::MIN, 3, i32::MIN, 4);
        let expected = _mm256_setr_epi32(
            i32::MIN,
            i32::MIN + 1,
            i32::MAX,
            i32::MAX - 1,
            i32::MIN + 2,
            i32::MIN + 3,
            i32::MAX - 2,
            i32::MAX - 3,
        );
        let r = _mm256_hsub_epi32(a, b);
        assert_eq_m256i(r, expected);
    }
    test_mm256_hsub_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_hsubs_epi16() {
        let a = _mm256_set1_epi16(2);
        let a = _mm256_insert_epi16::<0>(a, 0x7fff);
        let a = _mm256_insert_epi16::<1>(a, -1);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_hsubs_epi16(a, b);
        let e = _mm256_insert_epi16::<0>(_mm256_set1_epi16(0), 0x7FFF);
        assert_eq_m256i(r, e);

        // Test saturating on overflow
        let a = _mm256_setr_epi16(
            i16::MAX,
            -1,
            i16::MAX,
            -2,
            i16::MAX,
            -3,
            i16::MAX,
            -4,
            i16::MAX,
            -5,
            i16::MAX,
            -6,
            i16::MAX,
            -7,
            i16::MAX,
            -8,
        );
        let b = _mm256_setr_epi16(
            i16::MIN,
            1,
            i16::MIN,
            2,
            i16::MIN,
            3,
            i16::MIN,
            4,
            i16::MIN,
            5,
            i16::MIN,
            6,
            i16::MIN,
            7,
            i16::MIN,
            8,
        );
        let expected = _mm256_setr_epi16(
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MIN,
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MIN,
            i16::MIN,
        );
        let r = _mm256_hsubs_epi16(a, b);
        assert_eq_m256i(r, expected);
    }
    test_mm256_hsubs_epi16();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_i32gather_epi32() {
        let arr: [i32; 128] = core::array::from_fn(|i| i as i32);
        // A multiplier of 4 is word-addressing
        let r = _mm_i32gather_epi32::<4>(arr.as_ptr(), _mm_setr_epi32(0, 16, 32, 48));
        assert_eq_m128i(r, _mm_setr_epi32(0, 16, 32, 48));
    }
    test_mm_i32gather_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_mask_i32gather_epi32() {
        let arr: [i32; 128] = core::array::from_fn(|i| i as i32);
        // A multiplier of 4 is word-addressing
        let r = _mm_mask_i32gather_epi32::<4>(
            _mm_set1_epi32(256),
            arr.as_ptr(),
            _mm_setr_epi32(0, 16, 64, 96),
            _mm_setr_epi32(-1, -1, -1, 0),
        );
        assert_eq_m128i(r, _mm_setr_epi32(0, 16, 64, 256));
    }
    test_mm_mask_i32gather_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_i32gather_epi32() {
        let arr: [i32; 128] = core::array::from_fn(|i| i as i32);
        // A multiplier of 4 is word-addressing
        let r =
            _mm256_i32gather_epi32::<4>(arr.as_ptr(), _mm256_setr_epi32(0, 16, 32, 48, 1, 2, 3, 4));
        assert_eq_m256i(r, _mm256_setr_epi32(0, 16, 32, 48, 1, 2, 3, 4));
    }
    test_mm256_i32gather_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_mask_i32gather_epi32() {
        let arr: [i32; 128] = core::array::from_fn(|i| i as i32);
        // A multiplier of 4 is word-addressing
        let r = _mm256_mask_i32gather_epi32::<4>(
            _mm256_set1_epi32(256),
            arr.as_ptr(),
            _mm256_setr_epi32(0, 16, 64, 96, 0, 0, 0, 0),
            _mm256_setr_epi32(-1, -1, -1, 0, 0, 0, 0, 0),
        );
        assert_eq_m256i(r, _mm256_setr_epi32(0, 16, 64, 256, 256, 256, 256, 256));
    }
    test_mm256_mask_i32gather_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_i32gather_ps() {
        let arr: [f32; 128] = core::array::from_fn(|i| i as f32);
        // A multiplier of 4 is word-addressing for f32s
        let r = _mm_i32gather_ps::<4>(arr.as_ptr(), _mm_setr_epi32(0, 16, 32, 48));
        assert_eq_m128(r, _mm_setr_ps(0.0, 16.0, 32.0, 48.0));
    }
    test_mm_i32gather_ps();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_mask_i32gather_ps() {
        let arr: [f32; 128] = core::array::from_fn(|i| i as f32);
        // A multiplier of 4 is word-addressing for f32s
        let r = _mm_mask_i32gather_ps::<4>(
            _mm_set1_ps(256.0),
            arr.as_ptr(),
            _mm_setr_epi32(0, 16, 64, 96),
            _mm_setr_ps(-1.0, -1.0, -1.0, 0.0),
        );
        assert_eq_m128(r, _mm_setr_ps(0.0, 16.0, 64.0, 256.0));
    }
    test_mm_mask_i32gather_ps();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_i32gather_ps() {
        let arr: [f32; 128] = core::array::from_fn(|i| i as f32);
        // A multiplier of 4 is word-addressing for f32s
        let r =
            _mm256_i32gather_ps::<4>(arr.as_ptr(), _mm256_setr_epi32(0, 16, 32, 48, 1, 2, 3, 4));
        assert_eq_m256(r, _mm256_setr_ps(0.0, 16.0, 32.0, 48.0, 1.0, 2.0, 3.0, 4.0));
    }
    test_mm256_i32gather_ps();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_mask_i32gather_ps() {
        let arr: [f32; 128] = core::array::from_fn(|i| i as f32);
        // A multiplier of 4 is word-addressing for f32s
        let r = _mm256_mask_i32gather_ps::<4>(
            _mm256_set1_ps(256.0),
            arr.as_ptr(),
            _mm256_setr_epi32(0, 16, 64, 96, 0, 0, 0, 0),
            _mm256_setr_ps(-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        );
        assert_eq_m256(r, _mm256_setr_ps(0.0, 16.0, 64.0, 256.0, 256.0, 256.0, 256.0, 256.0));
    }
    test_mm256_mask_i32gather_ps();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_i32gather_epi64() {
        let arr: [i64; 128] = core::array::from_fn(|i| i as i64);
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm_i32gather_epi64::<8>(arr.as_ptr(), _mm_setr_epi32(0, 16, 0, 0));
        assert_eq_m128i(r, _mm_setr_epi64x(0, 16));
    }
    test_mm_i32gather_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_mask_i32gather_epi64() {
        let arr: [i64; 128] = core::array::from_fn(|i| i as i64);
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm_mask_i32gather_epi64::<8>(
            _mm_set1_epi64x(256),
            arr.as_ptr(),
            _mm_setr_epi32(16, 16, 16, 16),
            _mm_setr_epi64x(-1, 0),
        );
        assert_eq_m128i(r, _mm_setr_epi64x(16, 256));
    }
    test_mm_mask_i32gather_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_i32gather_epi64() {
        let arr: [i64; 128] = core::array::from_fn(|i| i as i64);
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm256_i32gather_epi64::<8>(arr.as_ptr(), _mm_setr_epi32(0, 16, 32, 48));
        assert_eq_m256i(r, _mm256_setr_epi64x(0, 16, 32, 48));
    }
    test_mm256_i32gather_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_mask_i32gather_epi64() {
        let arr: [i64; 128] = core::array::from_fn(|i| i as i64);
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm256_mask_i32gather_epi64::<8>(
            _mm256_set1_epi64x(256),
            arr.as_ptr(),
            _mm_setr_epi32(0, 16, 64, 96),
            _mm256_setr_epi64x(-1, -1, -1, 0),
        );
        assert_eq_m256i(r, _mm256_setr_epi64x(0, 16, 64, 256));
    }
    test_mm256_mask_i32gather_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_i32gather_pd() {
        let arr: [f64; 128] = core::array::from_fn(|i| i as f64);
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm_i32gather_pd::<8>(arr.as_ptr(), _mm_setr_epi32(0, 16, 0, 0));
        assert_eq_m128d(r, _mm_setr_pd(0.0, 16.0));
    }
    test_mm_i32gather_pd();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_mask_i32gather_pd() {
        let arr: [f64; 128] = core::array::from_fn(|i| i as f64);
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm_mask_i32gather_pd::<8>(
            _mm_set1_pd(256.0),
            arr.as_ptr(),
            _mm_setr_epi32(16, 16, 16, 16),
            _mm_setr_pd(-1.0, 0.0),
        );
        assert_eq_m128d(r, _mm_setr_pd(16.0, 256.0));
    }
    test_mm_mask_i32gather_pd();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_i32gather_pd() {
        let arr: [f64; 128] = core::array::from_fn(|i| i as f64);
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm256_i32gather_pd::<8>(arr.as_ptr(), _mm_setr_epi32(0, 16, 32, 48));
        assert_eq_m256d(r, _mm256_setr_pd(0.0, 16.0, 32.0, 48.0));
    }
    test_mm256_i32gather_pd();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_mask_i32gather_pd() {
        let arr: [f64; 128] = core::array::from_fn(|i| i as f64);
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm256_mask_i32gather_pd::<8>(
            _mm256_set1_pd(256.0),
            arr.as_ptr(),
            _mm_setr_epi32(0, 16, 64, 96),
            _mm256_setr_pd(-1.0, -1.0, -1.0, 0.0),
        );
        assert_eq_m256d(r, _mm256_setr_pd(0.0, 16.0, 64.0, 256.0));
    }
    test_mm256_mask_i32gather_pd();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_i64gather_epi32() {
        let arr: [i32; 128] = core::array::from_fn(|i| i as i32);
        // A multiplier of 4 is word-addressing
        let r = _mm_i64gather_epi32::<4>(arr.as_ptr(), _mm_setr_epi64x(0, 16));
        assert_eq_m128i(r, _mm_setr_epi32(0, 16, 0, 0));
    }
    test_mm_i64gather_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_mask_i64gather_epi32() {
        let arr: [i32; 128] = core::array::from_fn(|i| i as i32);
        // A multiplier of 4 is word-addressing
        let r = _mm_mask_i64gather_epi32::<4>(
            _mm_set1_epi32(256),
            arr.as_ptr(),
            _mm_setr_epi64x(0, 16),
            _mm_setr_epi32(-1, 0, -1, 0),
        );
        assert_eq_m128i(r, _mm_setr_epi32(0, 256, 0, 0));
    }
    test_mm_mask_i64gather_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_i64gather_epi32() {
        let arr: [i32; 128] = core::array::from_fn(|i| i as i32);
        // A multiplier of 4 is word-addressing
        let r = _mm256_i64gather_epi32::<4>(arr.as_ptr(), _mm256_setr_epi64x(0, 16, 32, 48));
        assert_eq_m128i(r, _mm_setr_epi32(0, 16, 32, 48));
    }
    test_mm256_i64gather_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_mask_i64gather_epi32() {
        let arr: [i32; 128] = core::array::from_fn(|i| i as i32);
        // A multiplier of 4 is word-addressing
        let r = _mm256_mask_i64gather_epi32::<4>(
            _mm_set1_epi32(256),
            arr.as_ptr(),
            _mm256_setr_epi64x(0, 16, 64, 96),
            _mm_setr_epi32(-1, -1, -1, 0),
        );
        assert_eq_m128i(r, _mm_setr_epi32(0, 16, 64, 256));
    }
    test_mm256_mask_i64gather_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_i64gather_ps() {
        let arr: [f32; 128] = core::array::from_fn(|i| i as f32);
        // A multiplier of 4 is word-addressing for f32s
        let r = _mm_i64gather_ps::<4>(arr.as_ptr(), _mm_setr_epi64x(0, 16));
        assert_eq_m128(r, _mm_setr_ps(0.0, 16.0, 0.0, 0.0));
    }
    test_mm_i64gather_ps();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_mask_i64gather_ps() {
        let arr: [f32; 128] = core::array::from_fn(|i| i as f32);
        // A multiplier of 4 is word-addressing for f32s
        let r = _mm_mask_i64gather_ps::<4>(
            _mm_set1_ps(256.0),
            arr.as_ptr(),
            _mm_setr_epi64x(0, 16),
            _mm_setr_ps(-1.0, 0.0, -1.0, 0.0),
        );
        assert_eq_m128(r, _mm_setr_ps(0.0, 256.0, 0.0, 0.0));
    }
    test_mm_mask_i64gather_ps();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_i64gather_ps() {
        let arr: [f32; 128] = core::array::from_fn(|i| i as f32);
        // A multiplier of 4 is word-addressing for f32s
        let r = _mm256_i64gather_ps::<4>(arr.as_ptr(), _mm256_setr_epi64x(0, 16, 32, 48));
        assert_eq_m128(r, _mm_setr_ps(0.0, 16.0, 32.0, 48.0));
    }
    test_mm256_i64gather_ps();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_mask_i64gather_ps() {
        let arr: [f32; 128] = core::array::from_fn(|i| i as f32);
        // A multiplier of 4 is word-addressing for f32s
        let r = _mm256_mask_i64gather_ps::<4>(
            _mm_set1_ps(256.0),
            arr.as_ptr(),
            _mm256_setr_epi64x(0, 16, 64, 96),
            _mm_setr_ps(-1.0, -1.0, -1.0, 0.0),
        );
        assert_eq_m128(r, _mm_setr_ps(0.0, 16.0, 64.0, 256.0));
    }
    test_mm256_mask_i64gather_ps();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_i64gather_epi64() {
        let arr: [i64; 128] = core::array::from_fn(|i| i as i64);
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm_i64gather_epi64::<8>(arr.as_ptr(), _mm_setr_epi64x(0, 16));
        assert_eq_m128i(r, _mm_setr_epi64x(0, 16));
    }
    test_mm_i64gather_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_mask_i64gather_epi64() {
        let arr: [i64; 128] = core::array::from_fn(|i| i as i64);
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm_mask_i64gather_epi64::<8>(
            _mm_set1_epi64x(256),
            arr.as_ptr(),
            _mm_setr_epi64x(16, 16),
            _mm_setr_epi64x(-1, 0),
        );
        assert_eq_m128i(r, _mm_setr_epi64x(16, 256));
    }
    test_mm_mask_i64gather_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_i64gather_epi64() {
        let arr: [i64; 128] = core::array::from_fn(|i| i as i64);
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm256_i64gather_epi64::<8>(arr.as_ptr(), _mm256_setr_epi64x(0, 16, 32, 48));
        assert_eq_m256i(r, _mm256_setr_epi64x(0, 16, 32, 48));
    }
    test_mm256_i64gather_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_mask_i64gather_epi64() {
        let arr: [i64; 128] = core::array::from_fn(|i| i as i64);
        // A multiplier of 8 is word-addressing for i64s
        let r = _mm256_mask_i64gather_epi64::<8>(
            _mm256_set1_epi64x(256),
            arr.as_ptr(),
            _mm256_setr_epi64x(0, 16, 64, 96),
            _mm256_setr_epi64x(-1, -1, -1, 0),
        );
        assert_eq_m256i(r, _mm256_setr_epi64x(0, 16, 64, 256));
    }
    test_mm256_mask_i64gather_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_i64gather_pd() {
        let arr: [f64; 128] = core::array::from_fn(|i| i as f64);
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm_i64gather_pd::<8>(arr.as_ptr(), _mm_setr_epi64x(0, 16));
        assert_eq_m128d(r, _mm_setr_pd(0.0, 16.0));
    }
    test_mm_i64gather_pd();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_mask_i64gather_pd() {
        let arr: [f64; 128] = core::array::from_fn(|i| i as f64);
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm_mask_i64gather_pd::<8>(
            _mm_set1_pd(256.0),
            arr.as_ptr(),
            _mm_setr_epi64x(16, 16),
            _mm_setr_pd(-1.0, 0.0),
        );
        assert_eq_m128d(r, _mm_setr_pd(16.0, 256.0));
    }
    test_mm_mask_i64gather_pd();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_i64gather_pd() {
        let arr: [f64; 128] = core::array::from_fn(|i| i as f64);
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm256_i64gather_pd::<8>(arr.as_ptr(), _mm256_setr_epi64x(0, 16, 32, 48));
        assert_eq_m256d(r, _mm256_setr_pd(0.0, 16.0, 32.0, 48.0));
    }
    test_mm256_i64gather_pd();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_mask_i64gather_pd() {
        let arr: [f64; 128] = core::array::from_fn(|i| i as f64);
        // A multiplier of 8 is word-addressing for f64s
        let r = _mm256_mask_i64gather_pd::<8>(
            _mm256_set1_pd(256.0),
            arr.as_ptr(),
            _mm256_setr_epi64x(0, 16, 64, 96),
            _mm256_setr_pd(-1.0, -1.0, -1.0, 0.0),
        );
        assert_eq_m256d(r, _mm256_setr_pd(0.0, 16.0, 64.0, 256.0));
    }
    test_mm256_mask_i64gather_pd();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_madd_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_madd_epi16(a, b);
        let e = _mm256_set1_epi32(16);
        assert_eq_m256i(r, e);
    }
    test_mm256_madd_epi16();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_maddubs_epi16() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_maddubs_epi16(a, b);
        let e = _mm256_set1_epi16(16);
        assert_eq_m256i(r, e);
    }
    test_mm256_maddubs_epi16();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_maskload_epi32() {
        let nums = [1, 2, 3, 4];
        let a = &nums as *const i32;
        let mask = _mm_setr_epi32(-1, 0, 0, -1);
        let r = _mm_maskload_epi32(a, mask);
        let e = _mm_setr_epi32(1, 0, 0, 4);
        assert_eq_m128i(r, e);

        // Unaligned pointer
        let a = Unaligned::new([1i32, 2, 3, 4]);
        let mask = _mm_setr_epi32(0, !0, 0, !0);
        let r = _mm_maskload_epi32(a.as_ptr().cast(), mask);
        let e = _mm_setr_epi32(0, 2, 0, 4);
        assert_eq_m128i(r, e);

        // Only loading first element, so slice can be short.
        let a = &[2i32];
        let mask = _mm_setr_epi32(!0, 0, 0, 0);
        let r = _mm_maskload_epi32(a.as_ptr(), mask);
        let e = _mm_setr_epi32(2, 0, 0, 0);
        assert_eq_m128i(r, e);

        // Only loading last element, so slice can be short.
        let a = &[2i32];
        let mask = _mm_setr_epi32(0, 0, 0, !0);
        let r = _mm_maskload_epi32(a.as_ptr().wrapping_sub(3), mask);
        let e = _mm_setr_epi32(0, 0, 0, 2);
        assert_eq_m128i(r, e);
    }
    test_mm_maskload_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_maskload_epi32() {
        let nums = [1, 2, 3, 4, 5, 6, 7, 8];
        let a = &nums as *const i32;
        let mask = _mm256_setr_epi32(-1, 0, 0, -1, 0, -1, -1, 0);
        let r = _mm256_maskload_epi32(a, mask);
        let e = _mm256_setr_epi32(1, 0, 0, 4, 0, 6, 7, 0);
        assert_eq_m256i(r, e);

        // Unaligned pointer
        let a = Unaligned::new([1i32, 2, 3, 4, 5, 6, 7, 8]);
        let mask = _mm256_setr_epi32(0, !0, 0, !0, 0, !0, 0, !0);
        let r = _mm256_maskload_epi32(a.as_ptr().cast(), mask);
        let e = _mm256_setr_epi32(0, 2, 0, 4, 0, 6, 0, 8);
        assert_eq_m256i(r, e);

        // Only loading first element, so slice can be short.
        let a = &[2i32];
        let mask = _mm256_setr_epi32(!0, 0, 0, 0, 0, 0, 0, 0);
        let r = _mm256_maskload_epi32(a.as_ptr(), mask);
        let e = _mm256_setr_epi32(2, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m256i(r, e);

        // Only loading last element, so slice can be short.
        let a = &[2i32];
        let mask = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, !0);
        let r = _mm256_maskload_epi32(a.as_ptr().wrapping_sub(7), mask);
        let e = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 2);
        assert_eq_m256i(r, e);
    }
    test_mm256_maskload_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_maskload_epi64() {
        let nums = [1_i64, 2_i64];
        let a = &nums as *const i64;
        let mask = _mm_setr_epi64x(0, -1);
        let r = _mm_maskload_epi64(a, mask);
        let e = _mm_setr_epi64x(0, 2);
        assert_eq_m128i(r, e);

        // Unaligned pointer
        let a = Unaligned::new([1i64, 2]);
        let mask = _mm_setr_epi64x(0, !0);
        let r = _mm_maskload_epi64(a.as_ptr().cast(), mask);
        let e = _mm_setr_epi64x(0, 2);
        assert_eq_m128i(r, e);

        // Only loading first element, so slice can be short.
        let a = &[2i64];
        let mask = _mm_setr_epi64x(!0, 0);
        let r = _mm_maskload_epi64(a.as_ptr(), mask);
        let e = _mm_setr_epi64x(2, 0);
        assert_eq_m128i(r, e);

        // Only loading last element, so slice can be short.
        let a = &[2i64];
        let mask = _mm_setr_epi64x(0, !0);
        let r = _mm_maskload_epi64(a.as_ptr().wrapping_sub(1), mask);
        let e = _mm_setr_epi64x(0, 2);
        assert_eq_m128i(r, e);
    }
    test_mm_maskload_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_maskload_epi64() {
        let nums = [1_i64, 2_i64, 3_i64, 4_i64];
        let a = &nums as *const i64;
        let mask = _mm256_setr_epi64x(0, -1, -1, 0);
        let r = _mm256_maskload_epi64(a, mask);
        let e = _mm256_setr_epi64x(0, 2, 3, 0);
        assert_eq_m256i(r, e);

        // Unaligned pointer
        let a = Unaligned::new([1i64, 2, 3, 4]);
        let mask = _mm256_setr_epi64x(0, !0, 0, !0);
        let r = _mm256_maskload_epi64(a.as_ptr().cast(), mask);
        let e = _mm256_setr_epi64x(0, 2, 0, 4);
        assert_eq_m256i(r, e);

        // Only loading first element, so slice can be short.
        let a = &[2i64];
        let mask = _mm256_setr_epi64x(!0, 0, 0, 0);
        let r = _mm256_maskload_epi64(a.as_ptr(), mask);
        let e = _mm256_setr_epi64x(2, 0, 0, 0);
        assert_eq_m256i(r, e);

        // Only loading last element, so slice can be short.
        let a = &[2i64];
        let mask = _mm256_setr_epi64x(0, 0, 0, !0);
        let r = _mm256_maskload_epi64(a.as_ptr().wrapping_sub(3), mask);
        let e = _mm256_setr_epi64x(0, 0, 0, 2);
        assert_eq_m256i(r, e);
    }
    test_mm256_maskload_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_maskstore_epi32() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let mut arr = [-1, -1, -1, -1];
        let mask = _mm_setr_epi32(-1, 0, 0, -1);
        _mm_maskstore_epi32(arr.as_mut_ptr(), mask, a);
        let e = [1, -1, -1, 4];
        assert_eq!(arr, e);

        // Unaligned pointer
        let mut r = Unaligned::new([0i32; 4]);
        let mask = _mm_setr_epi32(0, !0, 0, !0);
        let a = _mm_setr_epi32(1, 2, 3, 4);
        _mm_maskstore_epi32(r.as_mut_ptr().cast(), mask, a);
        let e = [0i32, 2, 0, 4];
        assert_eq!(r.read(), e);

        // Only storing first element, so slice can be short.
        let mut r = [0i32];
        let mask = _mm_setr_epi32(!0, 0, 0, 0);
        let a = _mm_setr_epi32(1, 2, 3, 4);
        _mm_maskstore_epi32(r.as_mut_ptr(), mask, a);
        let e = [1i32];
        assert_eq!(r, e);

        // Only storing last element, so slice can be short.
        let mut r = [0i32];
        let mask = _mm_setr_epi32(0, 0, 0, !0);
        let a = _mm_setr_epi32(1, 2, 3, 4);
        _mm_maskstore_epi32(r.as_mut_ptr().wrapping_sub(3), mask, a);
        let e = [4i32];
        assert_eq!(r, e);
    }
    test_mm_maskstore_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_maskstore_epi32() {
        let a = _mm256_setr_epi32(1, 0x6d726f, 3, 42, 0x777161, 6, 7, 8);
        let mut arr = [-1, -1, -1, 0x776173, -1, 0x68657265, -1, -1];
        let mask = _mm256_setr_epi32(-1, 0, 0, -1, 0, -1, -1, 0);
        _mm256_maskstore_epi32(arr.as_mut_ptr(), mask, a);
        let e = [1, -1, -1, 42, -1, 6, 7, -1];
        assert_eq!(arr, e);

        // Unaligned pointer
        let mut r = Unaligned::new([0i32; 8]);
        let mask = _mm256_setr_epi32(0, !0, 0, !0, 0, !0, 0, !0);
        let a = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        _mm256_maskstore_epi32(r.as_mut_ptr().cast(), mask, a);
        let e = [0i32, 2, 0, 4, 0, 6, 0, 8];
        assert_eq!(r.read(), e);

        // Only storing first element, so slice can be short.
        let mut r = [0i32];
        let mask = _mm256_setr_epi32(!0, 0, 0, 0, 0, 0, 0, 0);
        let a = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        _mm256_maskstore_epi32(r.as_mut_ptr(), mask, a);
        let e = [1i32];
        assert_eq!(r, e);

        // Only storing last element, so slice can be short.
        let mut r = [0i32];
        let mask = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, !0);
        let a = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        _mm256_maskstore_epi32(r.as_mut_ptr().wrapping_sub(7), mask, a);
        let e = [8i32];
        assert_eq!(r, e);
    }
    test_mm256_maskstore_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_maskstore_epi64() {
        let a = _mm_setr_epi64x(1_i64, 2_i64);
        let mut arr = [-1_i64, -1_i64];
        let mask = _mm_setr_epi64x(0, -1);
        _mm_maskstore_epi64(arr.as_mut_ptr(), mask, a);
        let e = [-1, 2];
        assert_eq!(arr, e);

        // Unaligned pointer
        let mut r = Unaligned::new([0i64; 2]);
        let mask = _mm_setr_epi64x(0, !0);
        let a = _mm_setr_epi64x(1, 2);
        _mm_maskstore_epi64(r.as_mut_ptr().cast(), mask, a);
        let e = [0i64, 2];
        assert_eq!(r.read(), e);

        // Only storing first element, so slice can be short.
        let mut r = [0i64];
        let mask = _mm_setr_epi64x(!0, 0);
        let a = _mm_setr_epi64x(1, 2);
        _mm_maskstore_epi64(r.as_mut_ptr(), mask, a);
        let e = [1i64];
        assert_eq!(r, e);

        // Only storing last element, so slice can be short.
        let mut r = [0i64];
        let mask = _mm_setr_epi64x(0, !0);
        let a = _mm_setr_epi64x(1, 2);
        _mm_maskstore_epi64(r.as_mut_ptr().wrapping_sub(1), mask, a);
        let e = [2i64];
        assert_eq!(r, e);
    }
    test_mm_maskstore_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_maskstore_epi64() {
        let a = _mm256_setr_epi64x(1_i64, 2_i64, 3_i64, 4_i64);
        let mut arr = [-1_i64, -1_i64, -1_i64, -1_i64];
        let mask = _mm256_setr_epi64x(0, -1, -1, 0);
        _mm256_maskstore_epi64(arr.as_mut_ptr(), mask, a);
        let e = [-1, 2, 3, -1];
        assert_eq!(arr, e);

        // Unaligned pointer
        let mut r = Unaligned::new([0i64; 4]);
        let mask = _mm256_setr_epi64x(0, !0, 0, !0);
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        _mm256_maskstore_epi64(r.as_mut_ptr().cast(), mask, a);
        let e = [0i64, 2, 0, 4];
        assert_eq!(r.read(), e);

        // Only storing first element, so slice can be short.
        let mut r = [0i64];
        let mask = _mm256_setr_epi64x(!0, 0, 0, 0);
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        _mm256_maskstore_epi64(r.as_mut_ptr(), mask, a);
        let e = [1i64];
        assert_eq!(r, e);

        // Only storing last element, so slice can be short.
        let mut r = [0i64];
        let mask = _mm256_setr_epi64x(0, 0, 0, !0);
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        _mm256_maskstore_epi64(r.as_mut_ptr().wrapping_sub(3), mask, a);
        let e = [4i64];
        assert_eq!(r, e);
    }
    test_mm256_maskstore_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_mpsadbw_epu8() {
        let a = _mm256_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 2, 4, 6, 8, 10, 12, 14, 16,
            18, 20, 22, 24, 26, 28, 30,
        );

        let r = _mm256_mpsadbw_epu8::<0b000>(a, a);
        let e = _mm256_setr_epi16(0, 4, 8, 12, 16, 20, 24, 28, 0, 8, 16, 24, 32, 40, 48, 56);
        assert_eq_m256i(r, e);

        let r = _mm256_mpsadbw_epu8::<0b001>(a, a);
        let e = _mm256_setr_epi16(16, 12, 8, 4, 0, 4, 8, 12, 32, 24, 16, 8, 0, 8, 16, 24);
        assert_eq_m256i(r, e);

        let r = _mm256_mpsadbw_epu8::<0b100>(a, a);
        let e = _mm256_setr_epi16(16, 20, 24, 28, 32, 36, 40, 44, 32, 40, 48, 56, 64, 72, 80, 88);
        assert_eq_m256i(r, e);

        let r = _mm256_mpsadbw_epu8::<0b101>(a, a);
        let e = _mm256_setr_epi16(0, 4, 8, 12, 16, 20, 24, 28, 0, 8, 16, 24, 32, 40, 48, 56);
        assert_eq_m256i(r, e);

        let r = _mm256_mpsadbw_epu8::<0b111>(a, a);
        let e = _mm256_setr_epi16(32, 28, 24, 20, 16, 12, 8, 4, 64, 56, 48, 40, 32, 24, 16, 8);
        assert_eq_m256i(r, e);
    }
    test_mm256_mpsadbw_epu8();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_mulhrs_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_mullo_epi16(a, b);
        let e = _mm256_set1_epi16(8);
        assert_eq_m256i(r, e);
    }
    test_mm256_mulhrs_epi16();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_packs_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_packs_epi16(a, b);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4,
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4,
        );

        assert_eq_m256i(r, e);
    }
    test_mm256_packs_epi16();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_packs_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(4);
        let r = _mm256_packs_epi32(a, b);
        let e = _mm256_setr_epi16(2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4);

        assert_eq_m256i(r, e);
    }
    test_mm256_packs_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_packus_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(4);
        let r = _mm256_packus_epi16(a, b);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4,
            2, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4,
        );

        assert_eq_m256i(r, e);
    }
    test_mm256_packus_epi16();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_packus_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(4);
        let r = _mm256_packus_epi32(a, b);
        let e = _mm256_setr_epi16(2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4);

        assert_eq_m256i(r, e);
    }
    test_mm256_packus_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_permutevar8x32_epi32() {
        let a = _mm256_setr_epi32(100, 200, 300, 400, 500, 600, 700, 800);
        let b = _mm256_setr_epi32(5, 0, 5, 1, 7, 6, 3, 4);
        let expected = _mm256_setr_epi32(600, 100, 600, 200, 800, 700, 400, 500);
        let r = _mm256_permutevar8x32_epi32(a, b);
        assert_eq_m256i(r, expected);
    }
    test_mm256_permutevar8x32_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_permute2x128_si256() {
        let a = _mm256_setr_epi64x(100, 200, 500, 600);
        let b = _mm256_setr_epi64x(300, 400, 700, 800);
        let r = _mm256_permute2x128_si256::<0b00_01_00_11>(a, b);
        let e = _mm256_setr_epi64x(700, 800, 500, 600);
        assert_eq_m256i(r, e);
    }
    test_mm256_permute2x128_si256();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_permutevar8x32_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let b = _mm256_setr_epi32(5, 0, 5, 1, 7, 6, 3, 4);
        let r = _mm256_permutevar8x32_ps(a, b);
        let e = _mm256_setr_ps(6., 1., 6., 2., 8., 7., 4., 5.);
        assert_eq_m256(r, e);
    }
    test_mm256_permutevar8x32_ps();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_sad_epu8() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(4);
        let r = _mm256_sad_epu8(a, b);
        let e = _mm256_set1_epi64x(16);
        assert_eq_m256i(r, e);
    }
    test_mm256_sad_epu8();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_shuffle_epi8() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        #[rustfmt::skip]
        let b = _mm256_setr_epi8(
            4, 128u8 as i8, 4, 3, 24, 12, 6, 19,
            12, 5, 5, 10, 4, 1, 8, 0,
            4, 128u8 as i8, 4, 3, 24, 12, 6, 19,
            12, 5, 5, 10, 4, 1, 8, 0,
        );
        #[rustfmt::skip]
        let expected = _mm256_setr_epi8(
            5, 0, 5, 4, 9, 13, 7, 4,
            13, 6, 6, 11, 5, 2, 9, 1,
            21, 0, 21, 20, 25, 29, 23, 20,
            29, 22, 22, 27, 21, 18, 25, 17,
        );
        let r = _mm256_shuffle_epi8(a, b);
        assert_eq_m256i(r, expected);
    }
    test_mm256_shuffle_epi8();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_sign_epi16() {
        let a = _mm256_set1_epi16(2);
        let b = _mm256_set1_epi16(-1);
        let r = _mm256_sign_epi16(a, b);
        let e = _mm256_set1_epi16(-2);
        assert_eq_m256i(r, e);
    }
    test_mm256_sign_epi16();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_sign_epi32() {
        let a = _mm256_set1_epi32(2);
        let b = _mm256_set1_epi32(-1);
        let r = _mm256_sign_epi32(a, b);
        let e = _mm256_set1_epi32(-2);
        assert_eq_m256i(r, e);
    }
    test_mm256_sign_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_sign_epi8() {
        let a = _mm256_set1_epi8(2);
        let b = _mm256_set1_epi8(-1);
        let r = _mm256_sign_epi8(a, b);
        let e = _mm256_set1_epi8(-2);
        assert_eq_m256i(r, e);
    }
    test_mm256_sign_epi8();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_sll_epi16() {
        let a = _mm256_setr_epi16(
            0x88, -0x88, 0x99, -0x99, 0xAA, -0xAA, 0xBB, -0xBB, 0xCC, -0xCC, 0xDD, -0xDD, 0xEE,
            -0xEE, 0xFF, -0xFF,
        );
        let r = _mm256_sll_epi16(a, _mm_set_epi64x(0, 4));
        assert_eq_m256i(
            r,
            _mm256_setr_epi16(
                0x880, -0x880, 0x990, -0x990, 0xAA0, -0xAA0, 0xBB0, -0xBB0, 0xCC0, -0xCC0, 0xDD0,
                -0xDD0, 0xEE0, -0xEE0, 0xFF0, -0xFF0,
            ),
        );
        let r = _mm256_sll_epi16(a, _mm_set_epi64x(4, 0));
        assert_eq_m256i(r, a);
        let r = _mm256_sll_epi16(a, _mm_set_epi64x(0, 16));
        assert_eq_m256i(r, _mm256_set1_epi16(0));
        let r = _mm256_sll_epi16(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m256i(r, _mm256_set1_epi16(0));
    }
    test_mm256_sll_epi16();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_sll_epi32() {
        let a =
            _mm256_setr_epi32(0xCCCC, -0xCCCC, 0xDDDD, -0xDDDD, 0xEEEE, -0xEEEE, 0xFFFF, -0xFFFF);
        let r = _mm256_sll_epi32(a, _mm_set_epi64x(0, 4));
        assert_eq_m256i(
            r,
            _mm256_setr_epi32(
                0xCCCC0, -0xCCCC0, 0xDDDD0, -0xDDDD0, 0xEEEE0, -0xEEEE0, 0xFFFF0, -0xFFFF0,
            ),
        );
        let r = _mm256_sll_epi32(a, _mm_set_epi64x(4, 0));
        assert_eq_m256i(r, a);
        let r = _mm256_sll_epi32(a, _mm_set_epi64x(0, 32));
        assert_eq_m256i(r, _mm256_set1_epi32(0));
        let r = _mm256_sll_epi32(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m256i(r, _mm256_set1_epi32(0));
    }
    test_mm256_sll_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_sll_epi64() {
        let a = _mm256_set_epi64x(0xEEEEEEEE, -0xEEEEEEEE, 0xFFFFFFFF, -0xFFFFFFFF);
        let r = _mm256_sll_epi64(a, _mm_set_epi64x(0, 4));
        assert_eq_m256i(r, _mm256_set_epi64x(0xEEEEEEEE0, -0xEEEEEEEE0, 0xFFFFFFFF0, -0xFFFFFFFF0));
        let r = _mm256_sll_epi64(a, _mm_set_epi64x(4, 0));
        assert_eq_m256i(r, a);
        let r = _mm256_sll_epi64(a, _mm_set_epi64x(0, 64));
        assert_eq_m256i(r, _mm256_set1_epi64x(0));
        let r = _mm256_sll_epi64(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m256i(r, _mm256_set1_epi64x(0));
    }
    test_mm256_sll_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_sra_epi16() {
        let a = _mm256_setr_epi16(
            0x88, -0x88, 0x99, -0x99, 0xAA, -0xAA, 0xBB, -0xBB, 0xCC, -0xCC, 0xDD, -0xDD, 0xEE,
            -0xEE, 0xFF, -0xFF,
        );
        let r = _mm256_sra_epi16(a, _mm_set_epi64x(0, 4));
        assert_eq_m256i(
            r,
            _mm256_setr_epi16(
                0x8, -0x9, 0x9, -0xA, 0xA, -0xB, 0xB, -0xC, 0xC, -0xD, 0xD, -0xE, 0xE, -0xF, 0xF,
                -0x10,
            ),
        );
        let r = _mm256_sra_epi16(a, _mm_set_epi64x(4, 0));
        assert_eq_m256i(r, a);
        let r = _mm256_sra_epi16(a, _mm_set_epi64x(0, 16));
        assert_eq_m256i(
            r,
            _mm256_setr_epi16(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1),
        );
        let r = _mm256_sra_epi16(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m256i(
            r,
            _mm256_setr_epi16(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1),
        );
    }
    test_mm256_sra_epi16();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_sra_epi32() {
        let a =
            _mm256_setr_epi32(0xCCCC, -0xCCCC, 0xDDDD, -0xDDDD, 0xEEEE, -0xEEEE, 0xFFFF, -0xFFFF);
        let r = _mm256_sra_epi32(a, _mm_set_epi64x(0, 4));
        assert_eq_m256i(
            r,
            _mm256_setr_epi32(0xCCC, -0xCCD, 0xDDD, -0xDDE, 0xEEE, -0xEEF, 0xFFF, -0x1000),
        );
        let r = _mm256_sra_epi32(a, _mm_set_epi64x(4, 0));
        assert_eq_m256i(r, a);
        let r = _mm256_sra_epi32(a, _mm_set_epi64x(0, 32));
        assert_eq_m256i(r, _mm256_setr_epi32(0, -1, 0, -1, 0, -1, 0, -1));
        let r = _mm256_sra_epi32(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m256i(r, _mm256_setr_epi32(0, -1, 0, -1, 0, -1, 0, -1));
    }
    test_mm256_sra_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_srl_epi16() {
        let a = _mm256_setr_epi16(
            0x88, -0x88, 0x99, -0x99, 0xAA, -0xAA, 0xBB, -0xBB, 0xCC, -0xCC, 0xDD, -0xDD, 0xEE,
            -0xEE, 0xFF, -0xFF,
        );
        let r = _mm256_srl_epi16(a, _mm_set_epi64x(0, 4));
        assert_eq_m256i(
            r,
            _mm256_setr_epi16(
                0x8, 0xFF7, 0x9, 0xFF6, 0xA, 0xFF5, 0xB, 0xFF4, 0xC, 0xFF3, 0xD, 0xFF2, 0xE, 0xFF1,
                0xF, 0xFF0,
            ),
        );
        let r = _mm256_srl_epi16(a, _mm_set_epi64x(4, 0));
        assert_eq_m256i(r, a);
        let r = _mm256_srl_epi16(a, _mm_set_epi64x(0, 16));
        assert_eq_m256i(r, _mm256_set1_epi16(0));
        let r = _mm256_srl_epi16(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m256i(r, _mm256_set1_epi16(0));
    }
    test_mm256_srl_epi16();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_srl_epi32() {
        let a =
            _mm256_setr_epi32(0xCCCC, -0xCCCC, 0xDDDD, -0xDDDD, 0xEEEE, -0xEEEE, 0xFFFF, -0xFFFF);
        let r = _mm256_srl_epi32(a, _mm_set_epi64x(0, 4));
        assert_eq_m256i(
            r,
            _mm256_setr_epi32(
                0xCCC, 0xFFFF333, 0xDDD, 0xFFFF222, 0xEEE, 0xFFFF111, 0xFFF, 0xFFFF000,
            ),
        );
        let r = _mm256_srl_epi32(a, _mm_set_epi64x(4, 0));
        assert_eq_m256i(r, a);
        let r = _mm256_srl_epi32(a, _mm_set_epi64x(0, 32));
        assert_eq_m256i(r, _mm256_set1_epi32(0));
        let r = _mm256_srl_epi32(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m256i(r, _mm256_set1_epi32(0));
    }
    test_mm256_srl_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_srl_epi64() {
        let a = _mm256_set_epi64x(0xEEEEEEEE, -0xEEEEEEEE, 0xFFFFFFFF, -0xFFFFFFFF);
        let r = _mm256_srl_epi64(a, _mm_set_epi64x(0, 4));
        assert_eq_m256i(
            r,
            _mm256_set_epi64x(0xEEEEEEE, 0xFFFFFFFF1111111, 0xFFFFFFF, 0xFFFFFFFF0000000),
        );
        let r = _mm256_srl_epi64(a, _mm_set_epi64x(4, 0));
        assert_eq_m256i(r, a);
        let r = _mm256_srl_epi64(a, _mm_set_epi64x(0, 64));
        assert_eq_m256i(r, _mm256_set1_epi64x(0));
        let r = _mm256_srl_epi64(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m256i(r, _mm256_set1_epi64x(0));
    }
    test_mm256_srl_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_sllv_epi32() {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let b = _mm_set_epi32(4, 3, 2, 1);
        let r = _mm_sllv_epi32(a, b);
        let e = _mm_set_epi32(16, 16, 12, 8);
        assert_eq_m128i(r, e);
    }
    test_mm_sllv_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_sllv_epi32() {
        let a = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
        let r = _mm256_sllv_epi32(a, b);
        let e = _mm256_set_epi32(256, 256, 192, 128, 80, 48, 28, 16);
        assert_eq_m256i(r, e);
    }
    test_mm256_sllv_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_sllv_epi64() {
        let a = _mm_set_epi64x(2, 3);
        let b = _mm_set_epi64x(1, 2);
        let r = _mm_sllv_epi64(a, b);
        let e = _mm_set_epi64x(4, 12);
        assert_eq_m128i(r, e);
    }
    test_mm_sllv_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_sllv_epi64() {
        let a = _mm256_set_epi64x(1, 2, 3, 4);
        let b = _mm256_set_epi64x(4, 3, 2, 1);
        let r = _mm256_sllv_epi64(a, b);
        let e = _mm256_set_epi64x(16, 16, 12, 8);
        assert_eq_m256i(r, e);
    }
    test_mm256_sllv_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_srav_epi32() {
        let a = _mm_set_epi32(16, -32, 64, -128);
        let b = _mm_set_epi32(4, 3, 2, 1);
        let r = _mm_srav_epi32(a, b);
        let e = _mm_set_epi32(1, -4, 16, -64);
        assert_eq_m128i(r, e);
    }
    test_mm_srav_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_srav_epi32() {
        let a = _mm256_set_epi32(256, -512, 1024, -2048, 4096, -8192, 16384, -32768);
        let b = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
        let r = _mm256_srav_epi32(a, b);
        let e = _mm256_set_epi32(1, -4, 16, -64, 256, -1024, 4096, -16384);
        assert_eq_m256i(r, e);
    }
    test_mm256_srav_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_srlv_epi32() {
        let a = _mm_set_epi32(16, 32, 64, 128);
        let b = _mm_set_epi32(4, 3, 2, 1);
        let r = _mm_srlv_epi32(a, b);
        let e = _mm_set_epi32(1, 4, 16, 64);
        assert_eq_m128i(r, e);
    }
    test_mm_srlv_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_srlv_epi32() {
        let a = _mm256_set_epi32(256, 512, 1024, 2048, 4096, 8192, 16384, 32768);
        let b = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
        let r = _mm256_srlv_epi32(a, b);
        let e = _mm256_set_epi32(1, 4, 16, 64, 256, 1024, 4096, 16384);
        assert_eq_m256i(r, e);
    }
    test_mm256_srlv_epi32();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm_srlv_epi64() {
        let a = _mm_set_epi64x(4, 8);
        let b = _mm_set_epi64x(2, 1);
        let r = _mm_srlv_epi64(a, b);
        let e = _mm_set_epi64x(1, 4);
        assert_eq_m128i(r, e);
    }
    test_mm_srlv_epi64();

    #[target_feature(enable = "avx2")]
    unsafe fn test_mm256_srlv_epi64() {
        let a = _mm256_set_epi64x(16, 32, 64, 128);
        let b = _mm256_set_epi64x(4, 3, 2, 1);
        let r = _mm256_srlv_epi64(a, b);
        let e = _mm256_set_epi64x(1, 4, 16, 64);
        assert_eq_m256i(r, e);
    }
    test_mm256_srlv_epi64();
}

#[target_feature(enable = "sse2")]
unsafe fn _mm_setr_epi64x(a: i64, b: i64) -> __m128i {
    _mm_set_epi64x(b, a)
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
unsafe fn assert_eq_m128i(a: __m128i, b: __m128i) {
    assert_eq!(transmute::<_, [u64; 2]>(a), transmute::<_, [u64; 2]>(b))
}

#[track_caller]
#[target_feature(enable = "avx")]
unsafe fn assert_eq_m256(a: __m256, b: __m256) {
    let cmp = _mm256_cmp_ps::<_CMP_EQ_OQ>(a, b);
    if _mm256_movemask_ps(cmp) != 0b11111111 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[track_caller]
#[target_feature(enable = "avx")]
unsafe fn assert_eq_m256d(a: __m256d, b: __m256d) {
    let cmp = _mm256_cmp_pd::<_CMP_EQ_OQ>(a, b);
    if _mm256_movemask_pd(cmp) != 0b1111 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[track_caller]
#[target_feature(enable = "avx")]
unsafe fn assert_eq_m256i(a: __m256i, b: __m256i) {
    assert_eq!(transmute::<_, [u64; 4]>(a), transmute::<_, [u64; 4]>(b))
}

/// Stores `T` in an unaligned address
struct Unaligned<T: Copy> {
    buf: Vec<u8>,
    offset: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Copy> Unaligned<T> {
    fn new(value: T) -> Self {
        // Allocate extra byte for unalignment headroom
        let len = std::mem::size_of::<T>();
        let mut buf = Vec::<u8>::with_capacity(len + 1);
        // Force the address to be a non-multiple of 2, so it is as unaligned as it can get.
        let offset = (buf.as_ptr() as usize % 2) == 0;
        let value_ptr: *const T = &value;
        unsafe {
            buf.as_mut_ptr().add(offset.into()).copy_from_nonoverlapping(value_ptr.cast(), len);
        }
        Self { buf, offset, _marker: std::marker::PhantomData }
    }

    fn as_ptr(&self) -> *const T {
        unsafe { self.buf.as_ptr().add(self.offset.into()).cast() }
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        unsafe { self.buf.as_mut_ptr().add(self.offset.into()).cast() }
    }

    fn read(&self) -> T {
        unsafe { self.as_ptr().read_unaligned() }
    }
}
