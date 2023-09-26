// Ignore everything except x86 and x86_64
// Any additional target are added to CI should be ignored here
// (We cannot use `cfg`-based tricks here since the `target-feature` flags below only work on x86.)
//@ignore-target-aarch64
//@ignore-target-arm
//@ignore-target-avr
//@ignore-target-s390x
//@ignore-target-thumbv7em
//@ignore-target-wasm32
//@compile-flags: -C target-feature=+sse3

use core::mem::transmute;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn main() {
    assert!(is_x86_feature_detected!("sse3"));

    unsafe {
        test_sse3();
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
