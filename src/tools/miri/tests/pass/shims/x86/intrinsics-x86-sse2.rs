// We're testing x86 target specific features
//@only-target: x86_64 i686
#![allow(unnecessary_transmutes)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::f64::NAN;
use std::mem::transmute;

fn main() {
    assert!(is_x86_feature_detected!("sse2"));

    unsafe {
        test_sse2();
    }
}

#[target_feature(enable = "sse2")]
unsafe fn _mm_setr_epi64x(a: i64, b: i64) -> __m128i {
    _mm_set_epi64x(b, a)
}

#[target_feature(enable = "sse2")]
unsafe fn test_sse2() {
    // Mostly copied from library/stdarch/crates/core_arch/src/x86{,_64}/sse2.rs

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
    unsafe fn assert_eq_m128i(a: __m128i, b: __m128i) {
        assert_eq!(transmute::<_, [u64; 2]>(a), transmute::<_, [u64; 2]>(b))
    }

    #[track_caller]
    #[target_feature(enable = "sse2")]
    unsafe fn assert_eq_m128d(a: __m128d, b: __m128d) {
        if _mm_movemask_pd(_mm_cmpeq_pd(a, b)) != 0b11 {
            panic!("{:?} != {:?}", a, b);
        }
    }

    fn test_mm_pause() {
        unsafe { _mm_pause() }
    }
    test_mm_pause();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_avg_epu8() {
        let (a, b) = (_mm_set1_epi8(3), _mm_set1_epi8(9));
        let r = _mm_avg_epu8(a, b);
        assert_eq_m128i(r, _mm_set1_epi8(6));
    }
    test_mm_avg_epu8();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_avg_epu16() {
        let (a, b) = (_mm_set1_epi16(3), _mm_set1_epi16(9));
        let r = _mm_avg_epu16(a, b);
        assert_eq_m128i(r, _mm_set1_epi16(6));
    }
    test_mm_avg_epu16();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_madd_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm_setr_epi16(9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm_madd_epi16(a, b);
        let e = _mm_setr_epi32(29, 81, 149, 233);
        assert_eq_m128i(r, e);

        let a = _mm_setr_epi16(i16::MAX, i16::MAX, i16::MIN, i16::MIN, i16::MIN, i16::MAX, 0, 0);
        let b = _mm_setr_epi16(i16::MAX, i16::MAX, i16::MIN, i16::MIN, i16::MAX, i16::MIN, 0, 0);
        let r = _mm_madd_epi16(a, b);
        let e = _mm_setr_epi32(0x7FFE0002, i32::MIN, -0x7FFF0000, 0);
        assert_eq_m128i(r, e);
    }
    test_mm_madd_epi16();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_mulhi_epi16() {
        let (a, b) = (_mm_set1_epi16(1000), _mm_set1_epi16(-1001));
        let r = _mm_mulhi_epi16(a, b);
        assert_eq_m128i(r, _mm_set1_epi16(-16));
    }
    test_mm_mulhi_epi16();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_mulhi_epu16() {
        let (a, b) = (_mm_set1_epi16(1000), _mm_set1_epi16(1001));
        let r = _mm_mulhi_epu16(a, b);
        assert_eq_m128i(r, _mm_set1_epi16(15));
    }
    test_mm_mulhi_epu16();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_mul_epu32() {
        let a = _mm_setr_epi64x(1_000_000_000, 1 << 34);
        let b = _mm_setr_epi64x(1_000_000_000, 1 << 35);
        let r = _mm_mul_epu32(a, b);
        let e = _mm_setr_epi64x(1_000_000_000 * 1_000_000_000, 0);
        assert_eq_m128i(r, e);
    }
    test_mm_mul_epu32();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_sad_epu8() {
        #[rustfmt::skip]
            let a = _mm_setr_epi8(
                255u8 as i8, 254u8 as i8, 253u8 as i8, 252u8 as i8,
                1, 2, 3, 4,
                155u8 as i8, 154u8 as i8, 153u8 as i8, 152u8 as i8,
                1, 2, 3, 4,
            );
        let b = _mm_setr_epi8(0, 0, 0, 0, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2);
        let r = _mm_sad_epu8(a, b);
        let e = _mm_setr_epi64x(1020, 614);
        assert_eq_m128i(r, e);
    }
    test_mm_sad_epu8();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_sll_epi16() {
        let a = _mm_setr_epi16(0xCC, -0xCC, 0xDD, -0xDD, 0xEE, -0xEE, 0xFF, -0xFF);
        let r = _mm_sll_epi16(a, _mm_set_epi64x(0, 4));
        assert_eq_m128i(
            r,
            _mm_setr_epi16(0xCC0, -0xCC0, 0xDD0, -0xDD0, 0xEE0, -0xEE0, 0xFF0, -0xFF0),
        );
        let r = _mm_sll_epi16(a, _mm_set_epi64x(4, 0));
        assert_eq_m128i(r, a);
        let r = _mm_sll_epi16(a, _mm_set_epi64x(0, 16));
        assert_eq_m128i(r, _mm_set1_epi16(0));
        let r = _mm_sll_epi16(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m128i(r, _mm_set1_epi16(0));
    }
    test_mm_sll_epi16();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_srl_epi16() {
        let a = _mm_setr_epi16(0xCC, -0xCC, 0xDD, -0xDD, 0xEE, -0xEE, 0xFF, -0xFF);
        let r = _mm_srl_epi16(a, _mm_set_epi64x(0, 4));
        assert_eq_m128i(r, _mm_setr_epi16(0xC, 0xFF3, 0xD, 0xFF2, 0xE, 0xFF1, 0xF, 0xFF0));
        let r = _mm_srl_epi16(a, _mm_set_epi64x(4, 0));
        assert_eq_m128i(r, a);
        let r = _mm_srl_epi16(a, _mm_set_epi64x(0, 16));
        assert_eq_m128i(r, _mm_set1_epi16(0));
        let r = _mm_srl_epi16(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m128i(r, _mm_set1_epi16(0));
    }
    test_mm_srl_epi16();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_sra_epi16() {
        let a = _mm_setr_epi16(0xCC, -0xCC, 0xDD, -0xDD, 0xEE, -0xEE, 0xFF, -0xFF);
        let r = _mm_sra_epi16(a, _mm_set_epi64x(0, 4));
        assert_eq_m128i(r, _mm_setr_epi16(0xC, -0xD, 0xD, -0xE, 0xE, -0xF, 0xF, -0x10));
        let r = _mm_sra_epi16(a, _mm_set_epi64x(4, 0));
        assert_eq_m128i(r, a);
        let r = _mm_sra_epi16(a, _mm_set_epi64x(0, 16));
        assert_eq_m128i(r, _mm_setr_epi16(0, -1, 0, -1, 0, -1, 0, -1));
        let r = _mm_sra_epi16(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m128i(r, _mm_setr_epi16(0, -1, 0, -1, 0, -1, 0, -1));
    }
    test_mm_sra_epi16();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_sll_epi32() {
        let a = _mm_setr_epi32(0xEEEE, -0xEEEE, 0xFFFF, -0xFFFF);
        let r = _mm_sll_epi32(a, _mm_set_epi64x(0, 4));
        assert_eq_m128i(r, _mm_setr_epi32(0xEEEE0, -0xEEEE0, 0xFFFF0, -0xFFFF0));
        let r = _mm_sll_epi32(a, _mm_set_epi64x(4, 0));
        assert_eq_m128i(r, a);
        let r = _mm_sll_epi32(a, _mm_set_epi64x(0, 32));
        assert_eq_m128i(r, _mm_set1_epi32(0));
        let r = _mm_sll_epi32(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m128i(r, _mm_set1_epi32(0));
    }
    test_mm_sll_epi32();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_srl_epi32() {
        let a = _mm_setr_epi32(0xEEEE, -0xEEEE, 0xFFFF, -0xFFFF);
        let r = _mm_srl_epi32(a, _mm_set_epi64x(0, 4));
        assert_eq_m128i(r, _mm_setr_epi32(0xEEE, 0xFFFF111, 0xFFF, 0xFFFF000));
        let r = _mm_srl_epi32(a, _mm_set_epi64x(4, 0));
        assert_eq_m128i(r, a);
        let r = _mm_srl_epi32(a, _mm_set_epi64x(0, 32));
        assert_eq_m128i(r, _mm_set1_epi32(0));
        let r = _mm_srl_epi32(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m128i(r, _mm_set1_epi32(0));
    }
    test_mm_srl_epi32();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_sra_epi32() {
        let a = _mm_setr_epi32(0xEEEE, -0xEEEE, 0xFFFF, -0xFFFF);
        let r = _mm_sra_epi32(a, _mm_set_epi64x(0, 4));
        assert_eq_m128i(r, _mm_setr_epi32(0xEEE, -0xEEF, 0xFFF, -0x1000));
        let r = _mm_sra_epi32(a, _mm_set_epi64x(4, 0));
        assert_eq_m128i(r, a);
        let r = _mm_sra_epi32(a, _mm_set_epi64x(0, 32));
        assert_eq_m128i(r, _mm_setr_epi32(0, -1, 0, -1));
        let r = _mm_sra_epi32(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m128i(r, _mm_setr_epi32(0, -1, 0, -1));
    }
    test_mm_sra_epi32();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_sll_epi64() {
        let a = _mm_set_epi64x(0xFFFFFFFF, -0xFFFFFFFF);
        let r = _mm_sll_epi64(a, _mm_set_epi64x(0, 4));
        assert_eq_m128i(r, _mm_set_epi64x(0xFFFFFFFF0, -0xFFFFFFFF0));
        let r = _mm_sll_epi64(a, _mm_set_epi64x(4, 0));
        assert_eq_m128i(r, a);
        let r = _mm_sll_epi64(a, _mm_set_epi64x(0, 64));
        assert_eq_m128i(r, _mm_set1_epi64x(0));
        let r = _mm_sll_epi64(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m128i(r, _mm_set1_epi64x(0));
    }
    test_mm_sll_epi64();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_srl_epi64() {
        let a = _mm_set_epi64x(0xFFFFFFFF, -0xFFFFFFFF);
        let r = _mm_srl_epi64(a, _mm_set_epi64x(0, 4));
        assert_eq_m128i(r, _mm_set_epi64x(0xFFFFFFF, 0xFFFFFFFF0000000));
        let r = _mm_srl_epi64(a, _mm_set_epi64x(4, 0));
        assert_eq_m128i(r, a);
        let r = _mm_srl_epi64(a, _mm_set_epi64x(0, 64));
        assert_eq_m128i(r, _mm_set1_epi64x(0));
        let r = _mm_srl_epi64(a, _mm_set_epi64x(0, i64::MAX));
        assert_eq_m128i(r, _mm_set1_epi64x(0));
    }
    test_mm_srl_epi64();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cvtepi32_ps() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let r = _mm_cvtepi32_ps(a);
        assert_eq_m128(r, _mm_setr_ps(1.0, 2.0, 3.0, 4.0));
    }
    test_mm_cvtepi32_ps();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cvtps_epi32() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let r = _mm_cvtps_epi32(a);
        assert_eq_m128i(r, _mm_setr_epi32(1, 2, 3, 4));
    }
    test_mm_cvtps_epi32();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cvttps_epi32() {
        let a = _mm_setr_ps(-1.1, 2.2, -3.3, 6.6);
        let r = _mm_cvttps_epi32(a);
        assert_eq_m128i(r, _mm_setr_epi32(-1, 2, -3, 6));

        let a = _mm_setr_ps(f32::NEG_INFINITY, f32::INFINITY, f32::MIN, f32::MAX);
        let r = _mm_cvttps_epi32(a);
        assert_eq_m128i(r, _mm_setr_epi32(i32::MIN, i32::MIN, i32::MIN, i32::MIN));
    }
    test_mm_cvttps_epi32();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_packs_epi16() {
        let a = _mm_setr_epi16(0x80, -0x81, 0, 0, 0, 0, 0, 0);
        let b = _mm_setr_epi16(0, 0, 0, 0, 0, 0, -0x81, 0x80);
        let r = _mm_packs_epi16(a, b);
        assert_eq_m128i(
            r,
            _mm_setr_epi8(0x7F, -0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0x80, 0x7F),
        );
    }
    test_mm_packs_epi16();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_packus_epi16() {
        let a = _mm_setr_epi16(0x100, -1, 0, 0, 0, 0, 0, 0);
        let b = _mm_setr_epi16(0, 0, 0, 0, 0, 0, -1, 0x100);
        let r = _mm_packus_epi16(a, b);
        assert_eq_m128i(r, _mm_setr_epi8(!0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, !0));
    }
    test_mm_packus_epi16();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_packs_epi32() {
        let a = _mm_setr_epi32(0x8000, -0x8001, 0, 0);
        let b = _mm_setr_epi32(0, 0, -0x8001, 0x8000);
        let r = _mm_packs_epi32(a, b);
        assert_eq_m128i(r, _mm_setr_epi16(0x7FFF, -0x8000, 0, 0, 0, 0, -0x8000, 0x7FFF));
    }
    test_mm_packs_epi32();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_min_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_min_sd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(1.0, 2.0));
    }
    test_mm_min_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_min_pd() {
        let a = _mm_setr_pd(-1.0, 5.0);
        let b = _mm_setr_pd(-100.0, 20.0);
        let r = _mm_min_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(-100.0, 5.0));

        // `_mm_min_pd` can **not** be implemented using the `simd_min` rust intrinsic because
        // the semantics of `simd_min` are different to those of `_mm_min_pd` regarding handling
        // of `-0.0`.
        let a = _mm_setr_pd(-0.0, 0.0);
        let b = _mm_setr_pd(0.0, 0.0);
        let r1: [u8; 16] = transmute(_mm_min_pd(a, b));
        let r2: [u8; 16] = transmute(_mm_min_pd(b, a));
        let a: [u8; 16] = transmute(a);
        let b: [u8; 16] = transmute(b);
        assert_eq!(r1, b);
        assert_eq!(r2, a);
        assert_ne!(a, b); // sanity check that -0.0 is actually present
    }
    test_mm_min_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_max_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_max_sd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(5.0, 2.0));
    }
    test_mm_max_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_max_pd() {
        let a = _mm_setr_pd(-1.0, 5.0);
        let b = _mm_setr_pd(-100.0, 20.0);
        let r = _mm_max_pd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(-1.0, 20.0));

        // `_mm_max_pd` can **not** be implemented using the `simd_max` rust intrinsic because
        // the semantics of `simd_max` are different to those of `_mm_max_pd` regarding handling
        // of `-0.0`.
        let a = _mm_setr_pd(-0.0, 0.0);
        let b = _mm_setr_pd(0.0, 0.0);
        let r1: [u8; 16] = transmute(_mm_max_pd(a, b));
        let r2: [u8; 16] = transmute(_mm_max_pd(b, a));
        let a: [u8; 16] = transmute(a);
        let b: [u8; 16] = transmute(b);
        assert_eq!(r1, b);
        assert_eq!(r2, a);
        assert_ne!(a, b); // sanity check that -0.0 is actually present
    }
    test_mm_max_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_sqrt_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(5.0, 10.0);
        let r = _mm_sqrt_sd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(5.0f64.sqrt(), 2.0));
    }
    test_mm_sqrt_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_sqrt_pd() {
        let r = _mm_sqrt_pd(_mm_setr_pd(1.0, 2.0));
        assert_eq_m128d(r, _mm_setr_pd(1.0f64.sqrt(), 2.0f64.sqrt()));
    }
    test_mm_sqrt_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpeq_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(!0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpeq_sd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpeq_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmplt_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(!0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmplt_sd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmplt_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmple_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(!0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmple_sd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmple_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpgt_sd() {
        let (a, b) = (_mm_setr_pd(5.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(!0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpgt_sd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpgt_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpge_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(!0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpge_sd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpge_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpord_sd() {
        let (a, b) = (_mm_setr_pd(NAN, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpord_sd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpord_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpunord_sd() {
        let (a, b) = (_mm_setr_pd(NAN, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(!0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpunord_sd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpunord_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpneq_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(!0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpneq_sd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpneq_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpnlt_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpnlt_sd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpnlt_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpnle_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpnle_sd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpnle_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpngt_sd() {
        let (a, b) = (_mm_setr_pd(5.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpngt_sd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpngt_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpnge_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, transmute(2.0f64));
        let r = transmute::<_, __m128i>(_mm_cmpnge_sd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpnge_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpeq_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(!0, 0);
        let r = transmute::<_, __m128i>(_mm_cmpeq_pd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpeq_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmplt_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, !0);
        let r = transmute::<_, __m128i>(_mm_cmplt_pd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmplt_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmple_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(!0, !0);
        let r = transmute::<_, __m128i>(_mm_cmple_pd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmple_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpgt_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, 0);
        let r = transmute::<_, __m128i>(_mm_cmpgt_pd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpgt_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpge_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(!0, 0);
        let r = transmute::<_, __m128i>(_mm_cmpge_pd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpge_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpord_pd() {
        let (a, b) = (_mm_setr_pd(NAN, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(0, !0);
        let r = transmute::<_, __m128i>(_mm_cmpord_pd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpord_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpunord_pd() {
        let (a, b) = (_mm_setr_pd(NAN, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(!0, 0);
        let r = transmute::<_, __m128i>(_mm_cmpunord_pd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpunord_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpneq_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(!0, !0);
        let r = transmute::<_, __m128i>(_mm_cmpneq_pd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpneq_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpnlt_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(5.0, 3.0));
        let e = _mm_setr_epi64x(0, 0);
        let r = transmute::<_, __m128i>(_mm_cmpnlt_pd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpnlt_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpnle_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, 0);
        let r = transmute::<_, __m128i>(_mm_cmpnle_pd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpnle_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpngt_pd() {
        let (a, b) = (_mm_setr_pd(5.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, !0);
        let r = transmute::<_, __m128i>(_mm_cmpngt_pd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpngt_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cmpnge_pd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        let e = _mm_setr_epi64x(0, !0);
        let r = transmute::<_, __m128i>(_mm_cmpnge_pd(a, b));
        assert_eq_m128i(r, e);
    }
    test_mm_cmpnge_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_comieq_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_comieq_sd(a, b) != 0);

        let (a, b) = (_mm_setr_pd(NAN, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_comieq_sd(a, b) == 0);
    }
    test_mm_comieq_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_comilt_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_comilt_sd(a, b) == 0);
    }
    test_mm_comilt_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_comile_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_comile_sd(a, b) != 0);
    }
    test_mm_comile_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_comigt_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_comigt_sd(a, b) == 0);
    }
    test_mm_comigt_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_comige_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_comige_sd(a, b) != 0);
    }
    test_mm_comige_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_comineq_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_comineq_sd(a, b) == 0);
    }
    test_mm_comineq_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_ucomieq_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_ucomieq_sd(a, b) != 0);

        let (a, b) = (_mm_setr_pd(NAN, 2.0), _mm_setr_pd(NAN, 3.0));
        assert!(_mm_ucomieq_sd(a, b) == 0);
    }
    test_mm_ucomieq_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_ucomilt_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_ucomilt_sd(a, b) == 0);
    }
    test_mm_ucomilt_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_ucomile_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_ucomile_sd(a, b) != 0);
    }
    test_mm_ucomile_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_ucomigt_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_ucomigt_sd(a, b) == 0);
    }
    test_mm_ucomigt_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_ucomige_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_ucomige_sd(a, b) != 0);
    }
    test_mm_ucomige_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_ucomineq_sd() {
        let (a, b) = (_mm_setr_pd(1.0, 2.0), _mm_setr_pd(1.0, 3.0));
        assert!(_mm_ucomineq_sd(a, b) == 0);
    }
    test_mm_ucomineq_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cvtpd_ps() {
        let r = _mm_cvtpd_ps(_mm_setr_pd(-1.0, 5.0));
        assert_eq_m128(r, _mm_setr_ps(-1.0, 5.0, 0.0, 0.0));

        let r = _mm_cvtpd_ps(_mm_setr_pd(-1.0, -5.0));
        assert_eq_m128(r, _mm_setr_ps(-1.0, -5.0, 0.0, 0.0));

        let r = _mm_cvtpd_ps(_mm_setr_pd(f64::MAX, f64::MIN));
        assert_eq_m128(r, _mm_setr_ps(f32::INFINITY, f32::NEG_INFINITY, 0.0, 0.0));

        let r = _mm_cvtpd_ps(_mm_setr_pd(f32::MAX as f64, f32::MIN as f64));
        assert_eq_m128(r, _mm_setr_ps(f32::MAX, f32::MIN, 0.0, 0.0));
    }
    test_mm_cvtpd_ps();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cvtps_pd() {
        let r = _mm_cvtps_pd(_mm_setr_ps(-1.0, 2.0, -3.0, 5.0));
        assert_eq_m128d(r, _mm_setr_pd(-1.0, 2.0));

        let r = _mm_cvtps_pd(_mm_setr_ps(f32::MAX, f32::INFINITY, f32::NEG_INFINITY, f32::MIN));
        assert_eq_m128d(r, _mm_setr_pd(f32::MAX as f64, f64::INFINITY));
    }
    test_mm_cvtps_pd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cvtpd_epi32() {
        let r = _mm_cvtpd_epi32(_mm_setr_pd(-1.0, 5.0));
        assert_eq_m128i(r, _mm_setr_epi32(-1, 5, 0, 0));

        let r = _mm_cvtpd_epi32(_mm_setr_pd(-1.0, -5.0));
        assert_eq_m128i(r, _mm_setr_epi32(-1, -5, 0, 0));

        let r = _mm_cvtpd_epi32(_mm_setr_pd(f64::MAX, f64::MIN));
        assert_eq_m128i(r, _mm_setr_epi32(i32::MIN, i32::MIN, 0, 0));

        let r = _mm_cvtpd_epi32(_mm_setr_pd(f64::INFINITY, f64::NEG_INFINITY));
        assert_eq_m128i(r, _mm_setr_epi32(i32::MIN, i32::MIN, 0, 0));

        let r = _mm_cvtpd_epi32(_mm_setr_pd(f64::NAN, f64::NAN));
        assert_eq_m128i(r, _mm_setr_epi32(i32::MIN, i32::MIN, 0, 0));
    }
    test_mm_cvtpd_epi32();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cvttpd_epi32() {
        let a = _mm_setr_pd(-1.1, 2.2);
        let r = _mm_cvttpd_epi32(a);
        assert_eq_m128i(r, _mm_setr_epi32(-1, 2, 0, 0));

        let a = _mm_setr_pd(f64::NEG_INFINITY, f64::NAN);
        let r = _mm_cvttpd_epi32(a);
        assert_eq_m128i(r, _mm_setr_epi32(i32::MIN, i32::MIN, 0, 0));
    }
    test_mm_cvttpd_epi32();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cvtsd_si32() {
        let r = _mm_cvtsd_si32(_mm_setr_pd(-2.0, 5.0));
        assert_eq!(r, -2);

        let r = _mm_cvtsd_si32(_mm_setr_pd(f64::MAX, f64::MIN));
        assert_eq!(r, i32::MIN);

        let r = _mm_cvtsd_si32(_mm_setr_pd(f64::NAN, f64::NAN));
        assert_eq!(r, i32::MIN);
    }
    test_mm_cvtsd_si32();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cvttsd_si32() {
        let a = _mm_setr_pd(-1.1, 2.2);
        let r = _mm_cvttsd_si32(a);
        assert_eq!(r, -1);

        let a = _mm_setr_pd(f64::NEG_INFINITY, f64::NAN);
        let r = _mm_cvttsd_si32(a);
        assert_eq!(r, i32::MIN);
    }
    test_mm_cvttsd_si32();

    // Intrinsic only available on x86_64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cvtsd_si64() {
        let r = _mm_cvtsd_si64(_mm_setr_pd(-2.0, 5.0));
        assert_eq!(r, -2_i64);

        let r = _mm_cvtsd_si64(_mm_setr_pd(f64::MAX, f64::MIN));
        assert_eq!(r, i64::MIN);
    }
    #[cfg(target_arch = "x86_64")]
    test_mm_cvtsd_si64();

    // Intrinsic only available on x86_64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cvttsd_si64() {
        let a = _mm_setr_pd(-1.1, 2.2);
        let r = _mm_cvttsd_si64(a);
        assert_eq!(r, -1_i64);
    }
    #[cfg(target_arch = "x86_64")]
    test_mm_cvttsd_si64();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cvtsd_ss() {
        let a = _mm_setr_ps(-1.1, -2.2, 3.3, 4.4);
        let b = _mm_setr_pd(2.0, -5.0);

        let r = _mm_cvtsd_ss(a, b);

        assert_eq_m128(r, _mm_setr_ps(2.0, -2.2, 3.3, 4.4));

        let a = _mm_setr_ps(-1.1, f32::NEG_INFINITY, f32::MAX, f32::NEG_INFINITY);
        let b = _mm_setr_pd(f64::INFINITY, -5.0);

        let r = _mm_cvtsd_ss(a, b);

        assert_eq_m128(
            r,
            _mm_setr_ps(f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::NEG_INFINITY),
        );
    }
    test_mm_cvtsd_ss();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_cvtss_sd() {
        let a = _mm_setr_pd(-1.1, 2.2);
        let b = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);

        let r = _mm_cvtss_sd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(1.0, 2.2));

        let a = _mm_setr_pd(-1.1, f64::INFINITY);
        let b = _mm_setr_ps(f32::NEG_INFINITY, 2.0, 3.0, 4.0);

        let r = _mm_cvtss_sd(a, b);
        assert_eq_m128d(r, _mm_setr_pd(f64::NEG_INFINITY, f64::INFINITY));
    }
    test_mm_cvtss_sd();

    #[target_feature(enable = "sse2")]
    unsafe fn test_mm_movemask_pd() {
        let r = _mm_movemask_pd(_mm_setr_pd(-1.0, 5.0));
        assert_eq!(r, 0b01);

        let r = _mm_movemask_pd(_mm_setr_pd(-1.0, -5.0));
        assert_eq!(r, 0b11);
    }
    test_mm_movemask_pd();
}
