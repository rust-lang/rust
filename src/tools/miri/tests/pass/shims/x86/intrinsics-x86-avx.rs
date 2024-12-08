// We're testing x86 target specific features
//@only-target: x86_64 i686
//@compile-flags: -C target-feature=+avx

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::transmute;

fn main() {
    assert!(is_x86_feature_detected!("avx"));

    unsafe {
        test_avx();
    }
}

#[target_feature(enable = "avx")]
unsafe fn test_avx() {
    // Mostly copied from library/stdarch/crates/core_arch/src/x86/avx.rs

    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr, $eps:expr) => {{
            let (a, b) = (&$a, &$b);
            assert!(
                (*a - *b).abs() < $eps,
                "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
                *a,
                *b,
                $eps,
                (*a - *b).abs()
            );
        }};
    }

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_max_pd() {
        let a = _mm256_setr_pd(1., 4., 5., 8.);
        let b = _mm256_setr_pd(2., 3., 6., 7.);
        let r = _mm256_max_pd(a, b);
        let e = _mm256_setr_pd(2., 4., 6., 8.);
        assert_eq_m256d(r, e);
        // > If the values being compared are both 0.0s (of either sign), the
        // > value in the second operand (source operand) is returned.
        let w = _mm256_max_pd(_mm256_set1_pd(0.0), _mm256_set1_pd(-0.0));
        let x = _mm256_max_pd(_mm256_set1_pd(-0.0), _mm256_set1_pd(0.0));
        let wu: [u64; 4] = transmute(w);
        let xu: [u64; 4] = transmute(x);
        assert_eq!(wu, [0x8000_0000_0000_0000u64; 4]);
        assert_eq!(xu, [0u64; 4]);
        // > If only one value is a NaN (SNaN or QNaN) for this instruction, the
        // > second operand (source operand), either a NaN or a valid
        // > floating-point value, is written to the result.
        let y = _mm256_max_pd(_mm256_set1_pd(f64::NAN), _mm256_set1_pd(0.0));
        let z = _mm256_max_pd(_mm256_set1_pd(0.0), _mm256_set1_pd(f64::NAN));
        let yf: [f64; 4] = transmute(y);
        let zf: [f64; 4] = transmute(z);
        assert_eq!(yf, [0.0; 4]);
        assert!(zf.iter().all(|f| f.is_nan()), "{:?}", zf);
    }
    test_mm256_max_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_max_ps() {
        let a = _mm256_setr_ps(1., 4., 5., 8., 9., 12., 13., 16.);
        let b = _mm256_setr_ps(2., 3., 6., 7., 10., 11., 14., 15.);
        let r = _mm256_max_ps(a, b);
        let e = _mm256_setr_ps(2., 4., 6., 8., 10., 12., 14., 16.);
        assert_eq_m256(r, e);
        // > If the values being compared are both 0.0s (of either sign), the
        // > value in the second operand (source operand) is returned.
        let w = _mm256_max_ps(_mm256_set1_ps(0.0), _mm256_set1_ps(-0.0));
        let x = _mm256_max_ps(_mm256_set1_ps(-0.0), _mm256_set1_ps(0.0));
        let wu: [u32; 8] = transmute(w);
        let xu: [u32; 8] = transmute(x);
        assert_eq!(wu, [0x8000_0000u32; 8]);
        assert_eq!(xu, [0u32; 8]);
        // > If only one value is a NaN (SNaN or QNaN) for this instruction, the
        // > second operand (source operand), either a NaN or a valid
        // > floating-point value, is written to the result.
        let y = _mm256_max_ps(_mm256_set1_ps(f32::NAN), _mm256_set1_ps(0.0));
        let z = _mm256_max_ps(_mm256_set1_ps(0.0), _mm256_set1_ps(f32::NAN));
        let yf: [f32; 8] = transmute(y);
        let zf: [f32; 8] = transmute(z);
        assert_eq!(yf, [0.0; 8]);
        assert!(zf.iter().all(|f| f.is_nan()), "{:?}", zf);
    }
    test_mm256_max_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_min_pd() {
        let a = _mm256_setr_pd(1., 4., 5., 8.);
        let b = _mm256_setr_pd(2., 3., 6., 7.);
        let r = _mm256_min_pd(a, b);
        let e = _mm256_setr_pd(1., 3., 5., 7.);
        assert_eq_m256d(r, e);
        // > If the values being compared are both 0.0s (of either sign), the
        // > value in the second operand (source operand) is returned.
        let w = _mm256_min_pd(_mm256_set1_pd(0.0), _mm256_set1_pd(-0.0));
        let x = _mm256_min_pd(_mm256_set1_pd(-0.0), _mm256_set1_pd(0.0));
        let wu: [u64; 4] = transmute(w);
        let xu: [u64; 4] = transmute(x);
        assert_eq!(wu, [0x8000_0000_0000_0000u64; 4]);
        assert_eq!(xu, [0u64; 4]);
        // > If only one value is a NaN (SNaN or QNaN) for this instruction, the
        // > second operand (source operand), either a NaN or a valid
        // > floating-point value, is written to the result.
        let y = _mm256_min_pd(_mm256_set1_pd(f64::NAN), _mm256_set1_pd(0.0));
        let z = _mm256_min_pd(_mm256_set1_pd(0.0), _mm256_set1_pd(f64::NAN));
        let yf: [f64; 4] = transmute(y);
        let zf: [f64; 4] = transmute(z);
        assert_eq!(yf, [0.0; 4]);
        assert!(zf.iter().all(|f| f.is_nan()), "{:?}", zf);
    }
    test_mm256_min_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_min_ps() {
        let a = _mm256_setr_ps(1., 4., 5., 8., 9., 12., 13., 16.);
        let b = _mm256_setr_ps(2., 3., 6., 7., 10., 11., 14., 15.);
        let r = _mm256_min_ps(a, b);
        let e = _mm256_setr_ps(1., 3., 5., 7., 9., 11., 13., 15.);
        assert_eq_m256(r, e);
        // > If the values being compared are both 0.0s (of either sign), the
        // > value in the second operand (source operand) is returned.
        let w = _mm256_min_ps(_mm256_set1_ps(0.0), _mm256_set1_ps(-0.0));
        let x = _mm256_min_ps(_mm256_set1_ps(-0.0), _mm256_set1_ps(0.0));
        let wu: [u32; 8] = transmute(w);
        let xu: [u32; 8] = transmute(x);
        assert_eq!(wu, [0x8000_0000u32; 8]);
        assert_eq!(xu, [0u32; 8]);
        // > If only one value is a NaN (SNaN or QNaN) for this instruction, the
        // > second operand (source operand), either a NaN or a valid
        // > floating-point value, is written to the result.
        let y = _mm256_min_ps(_mm256_set1_ps(f32::NAN), _mm256_set1_ps(0.0));
        let z = _mm256_min_ps(_mm256_set1_ps(0.0), _mm256_set1_ps(f32::NAN));
        let yf: [f32; 8] = transmute(y);
        let zf: [f32; 8] = transmute(z);
        assert_eq!(yf, [0.0; 8]);
        assert!(zf.iter().all(|f| f.is_nan()), "{:?}", zf);
    }
    test_mm256_min_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_round_nearest_f32() {
        #[target_feature(enable = "avx")]
        unsafe fn test(x: f32, res: f32) {
            let a = _mm256_set1_ps(x);
            let e = _mm256_set1_ps(res);
            let r = _mm256_round_ps::<_MM_FROUND_TO_NEAREST_INT>(a);
            assert_eq_m256(r, e);
            // Assume round-to-nearest by default
            let r = _mm256_round_ps::<_MM_FROUND_CUR_DIRECTION>(a);
            assert_eq_m256(r, e);
        }

        // Test rounding direction
        test(-2.5, -2.0);
        test(-1.75, -2.0);
        test(-1.5, -2.0);
        test(-1.25, -1.0);
        test(-1.0, -1.0);
        test(0.0, 0.0);
        test(1.0, 1.0);
        test(1.25, 1.0);
        test(1.5, 2.0);
        test(1.75, 2.0);
        test(2.5, 2.0);

        // Test that each element is rounded
        let a = _mm256_setr_ps(1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5);
        let e = _mm256_setr_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
        let r = _mm256_round_ps::<_MM_FROUND_TO_NEAREST_INT>(a);
        assert_eq_m256(r, e);
        // Assume round-to-nearest by default
        let r = _mm256_round_ps::<_MM_FROUND_CUR_DIRECTION>(a);
        assert_eq_m256(r, e);
    }
    test_round_nearest_f32();

    #[target_feature(enable = "avx")]
    unsafe fn test_round_floor_f32() {
        #[target_feature(enable = "avx")]
        unsafe fn test(x: f32, res: f32) {
            let a = _mm256_set1_ps(x);
            let e = _mm256_set1_ps(res);
            let r = _mm256_floor_ps(a);
            assert_eq_m256(r, e);
            let r = _mm256_round_ps::<_MM_FROUND_TO_NEG_INF>(a);
            assert_eq_m256(r, e);
        }

        // Test rounding direction
        test(-2.5, -3.0);
        test(-1.75, -2.0);
        test(-1.5, -2.0);
        test(-1.25, -2.0);
        test(-1.0, -1.0);
        test(0.0, 0.0);
        test(1.0, 1.0);
        test(1.25, 1.0);
        test(1.5, 1.0);
        test(1.75, 1.0);
        test(2.5, 2.0);

        // Test that each element is rounded
        let a = _mm256_setr_ps(1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5);
        let e = _mm256_setr_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);
        let r = _mm256_floor_ps(a);
        assert_eq_m256(r, e);
        let r = _mm256_round_ps::<_MM_FROUND_TO_NEG_INF>(a);
        assert_eq_m256(r, e);
    }
    test_round_floor_f32();

    #[target_feature(enable = "avx")]
    unsafe fn test_round_ceil_f32() {
        #[target_feature(enable = "avx")]
        unsafe fn test(x: f32, res: f32) {
            let a = _mm256_set1_ps(x);
            let e = _mm256_set1_ps(res);
            let r = _mm256_ceil_ps(a);
            assert_eq_m256(r, e);
            let r = _mm256_round_ps::<_MM_FROUND_TO_POS_INF>(a);
            assert_eq_m256(r, e);
        }

        // Test rounding direction
        test(-2.5, -2.0);
        test(-1.75, -1.0);
        test(-1.5, -1.0);
        test(-1.25, -1.0);
        test(-1.0, -1.0);
        test(0.0, 0.0);
        test(1.0, 1.0);
        test(1.25, 2.0);
        test(1.5, 2.0);
        test(1.75, 2.0);
        test(2.5, 3.0);

        // Test that each element is rounded
        let a = _mm256_setr_ps(1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5);
        let e = _mm256_setr_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
        let r = _mm256_ceil_ps(a);
        assert_eq_m256(r, e);
        let r = _mm256_round_ps::<_MM_FROUND_TO_POS_INF>(a);
        assert_eq_m256(r, e);
    }
    test_round_ceil_f32();

    #[target_feature(enable = "avx")]
    unsafe fn test_round_trunc_f32() {
        #[target_feature(enable = "avx")]
        unsafe fn test(x: f32, res: f32) {
            let a = _mm256_set1_ps(x);
            let e = _mm256_set1_ps(res);
            let r = _mm256_round_ps::<_MM_FROUND_TO_ZERO>(a);
            assert_eq_m256(r, e);
        }

        // Test rounding direction
        test(-2.5, -2.0);
        test(-1.75, -1.0);
        test(-1.5, -1.0);
        test(-1.25, -1.0);
        test(-1.0, -1.0);
        test(0.0, 0.0);
        test(1.0, 1.0);
        test(1.25, 1.0);
        test(1.5, 1.0);
        test(1.75, 1.0);
        test(2.5, 2.0);

        // Test that each element is rounded
        let a = _mm256_setr_ps(1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5);
        let e = _mm256_setr_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);
        let r = _mm256_round_ps::<_MM_FROUND_TO_ZERO>(a);
        assert_eq_m256(r, e);
    }
    test_round_trunc_f32();

    #[target_feature(enable = "avx")]
    unsafe fn test_round_nearest_f64() {
        #[target_feature(enable = "avx")]
        unsafe fn test(x: f64, res: f64) {
            let a = _mm256_set1_pd(x);
            let e = _mm256_set1_pd(res);
            let r = _mm256_round_pd::<_MM_FROUND_TO_NEAREST_INT>(a);
            assert_eq_m256d(r, e);
            // Assume round-to-nearest by default
            let r = _mm256_round_pd::<_MM_FROUND_CUR_DIRECTION>(a);
            assert_eq_m256d(r, e);
        }

        // Test rounding direction
        test(-2.5, -2.0);
        test(-1.75, -2.0);
        test(-1.5, -2.0);
        test(-1.25, -1.0);
        test(-1.0, -1.0);
        test(0.0, 0.0);
        test(1.0, 1.0);
        test(1.25, 1.0);
        test(1.5, 2.0);
        test(1.75, 2.0);
        test(2.5, 2.0);

        // Test that each element is rounded
        let a = _mm256_setr_pd(1.5, 3.5, 5.5, 7.5);
        let e = _mm256_setr_pd(2.0, 4.0, 6.0, 8.0);
        let r = _mm256_round_pd::<_MM_FROUND_TO_NEAREST_INT>(a);
        assert_eq_m256d(r, e);
        // Assume round-to-nearest by default
        let r = _mm256_round_pd::<_MM_FROUND_CUR_DIRECTION>(a);
        assert_eq_m256d(r, e);
    }
    test_round_nearest_f64();

    #[target_feature(enable = "avx")]
    unsafe fn test_round_floor_f64() {
        #[target_feature(enable = "avx")]
        unsafe fn test(x: f64, res: f64) {
            let a = _mm256_set1_pd(x);
            let e = _mm256_set1_pd(res);
            let r = _mm256_floor_pd(a);
            assert_eq_m256d(r, e);
            let r = _mm256_round_pd::<_MM_FROUND_TO_NEG_INF>(a);
            assert_eq_m256d(r, e);
        }

        // Test rounding direction
        test(-2.5, -3.0);
        test(-1.75, -2.0);
        test(-1.5, -2.0);
        test(-1.25, -2.0);
        test(-1.0, -1.0);
        test(0.0, 0.0);
        test(1.0, 1.0);
        test(1.25, 1.0);
        test(1.5, 1.0);
        test(1.75, 1.0);
        test(2.5, 2.0);

        // Test that each element is rounded
        let a = _mm256_setr_pd(1.5, 3.5, 5.5, 7.5);
        let e = _mm256_setr_pd(1.0, 3.0, 5.0, 7.0);
        let r = _mm256_floor_pd(a);
        assert_eq_m256d(r, e);
        let r = _mm256_round_pd::<_MM_FROUND_TO_NEG_INF>(a);
        assert_eq_m256d(r, e);
    }
    test_round_floor_f64();

    #[target_feature(enable = "avx")]
    unsafe fn test_round_ceil_f64() {
        #[target_feature(enable = "avx")]
        unsafe fn test(x: f64, res: f64) {
            let a = _mm256_set1_pd(x);
            let e = _mm256_set1_pd(res);
            let r = _mm256_ceil_pd(a);
            assert_eq_m256d(r, e);
            let r = _mm256_round_pd::<_MM_FROUND_TO_POS_INF>(a);
            assert_eq_m256d(r, e);
        }

        // Test rounding direction
        test(-2.5, -2.0);
        test(-1.75, -1.0);
        test(-1.5, -1.0);
        test(-1.25, -1.0);
        test(-1.0, -1.0);
        test(0.0, 0.0);
        test(1.0, 1.0);
        test(1.25, 2.0);
        test(1.5, 2.0);
        test(1.75, 2.0);
        test(2.5, 3.0);

        // Test that each element is rounded
        let a = _mm256_setr_pd(1.5, 3.5, 5.5, 7.5);
        let e = _mm256_setr_pd(2.0, 4.0, 6.0, 8.0);
        let r = _mm256_ceil_pd(a);
        assert_eq_m256d(r, e);
        let r = _mm256_round_pd::<_MM_FROUND_TO_POS_INF>(a);
        assert_eq_m256d(r, e);
    }
    test_round_ceil_f64();

    #[target_feature(enable = "avx")]
    unsafe fn test_round_trunc_f64() {
        #[target_feature(enable = "avx")]
        unsafe fn test(x: f64, res: f64) {
            let a = _mm256_set1_pd(x);
            let e = _mm256_set1_pd(res);
            let r = _mm256_round_pd::<_MM_FROUND_TO_ZERO>(a);
            assert_eq_m256d(r, e);
        }

        // Test rounding direction
        test(-2.5, -2.0);
        test(-1.75, -1.0);
        test(-1.5, -1.0);
        test(-1.25, -1.0);
        test(-1.0, -1.0);
        test(0.0, 0.0);
        test(1.0, 1.0);
        test(1.25, 1.0);
        test(1.5, 1.0);
        test(1.75, 1.0);
        test(2.5, 2.0);

        // Test that each element is rounded
        let a = _mm256_setr_pd(1.5, 3.5, 5.5, 7.5);
        let e = _mm256_setr_pd(1.0, 3.0, 5.0, 7.0);
        let r = _mm256_round_pd::<_MM_FROUND_TO_ZERO>(a);
        assert_eq_m256d(r, e);
    }
    test_round_trunc_f64();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_sqrt_ps() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let r = _mm256_sqrt_ps(a);
        let e = _mm256_setr_ps(2., 3., 4., 5., 2., 3., 4., 5.);
        assert_eq_m256(r, e);
    }
    test_mm256_sqrt_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_rcp_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_rcp_ps(a);
        #[rustfmt::skip]
        let e = _mm256_setr_ps(
            0.99975586, 0.49987793, 0.33325195, 0.24993896,
            0.19995117, 0.16662598, 0.14282227, 0.12496948,
        );
        let rel_err = 0.00048828125;

        let r: [f32; 8] = transmute(r);
        let e: [f32; 8] = transmute(e);
        for i in 0..8 {
            assert_approx_eq!(r[i], e[i], 2. * rel_err);
        }
    }
    test_mm256_rcp_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_rsqrt_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        let r = _mm256_rsqrt_ps(a);
        #[rustfmt::skip]
        let e = _mm256_setr_ps(
            0.99975586, 0.7069092, 0.5772705, 0.49987793,
            0.44714355, 0.40820313, 0.3779297, 0.3534546,
        );
        let rel_err = 0.00048828125;

        let r: [f32; 8] = transmute(r);
        let e: [f32; 8] = transmute(e);
        for i in 0..8 {
            assert_approx_eq!(r[i], e[i], 2. * rel_err);
        }
    }
    test_mm256_rsqrt_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_dp_ps() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = _mm256_dp_ps::<0xFF>(a, b);
        let e = _mm256_setr_ps(200., 200., 200., 200., 2387., 2387., 2387., 2387.);
        assert_eq_m256(r, e);
    }
    test_mm256_dp_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_hadd_pd() {
        let a = _mm256_setr_pd(4., 9., 16., 25.);
        let b = _mm256_setr_pd(4., 3., 2., 5.);
        let r = _mm256_hadd_pd(a, b);
        let e = _mm256_setr_pd(13., 7., 41., 7.);
        assert_eq_m256d(r, e);

        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_hadd_pd(a, b);
        let e = _mm256_setr_pd(3., 11., 7., 15.);
        assert_eq_m256d(r, e);
    }
    test_mm256_hadd_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_hadd_ps() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = _mm256_hadd_ps(a, b);
        let e = _mm256_setr_ps(13., 41., 7., 7., 13., 41., 17., 114.);
        assert_eq_m256(r, e);

        let a = _mm256_setr_ps(1., 2., 3., 4., 1., 2., 3., 4.);
        let b = _mm256_setr_ps(5., 6., 7., 8., 5., 6., 7., 8.);
        let r = _mm256_hadd_ps(a, b);
        let e = _mm256_setr_ps(3., 7., 11., 15., 3., 7., 11., 15.);
        assert_eq_m256(r, e);
    }
    test_mm256_hadd_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_hsub_pd() {
        let a = _mm256_setr_pd(4., 9., 16., 25.);
        let b = _mm256_setr_pd(4., 3., 2., 5.);
        let r = _mm256_hsub_pd(a, b);
        let e = _mm256_setr_pd(-5., 1., -9., -3.);
        assert_eq_m256d(r, e);

        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_hsub_pd(a, b);
        let e = _mm256_setr_pd(-1., -1., -1., -1.);
        assert_eq_m256d(r, e);
    }
    test_mm256_hsub_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_hsub_ps() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let b = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let r = _mm256_hsub_ps(a, b);
        let e = _mm256_setr_ps(-5., -9., 1., -3., -5., -9., -1., 14.);
        assert_eq_m256(r, e);

        let a = _mm256_setr_ps(1., 2., 3., 4., 1., 2., 3., 4.);
        let b = _mm256_setr_ps(5., 6., 7., 8., 5., 6., 7., 8.);
        let r = _mm256_hsub_ps(a, b);
        let e = _mm256_setr_ps(-1., -1., -1., -1., -1., -1., -1., -1.);
        assert_eq_m256(r, e);
    }
    test_mm256_hsub_ps();

    fn expected_cmp<F: PartialOrd>(imm: i32, lhs: F, rhs: F, if_t: F, if_f: F) -> F {
        let res = match imm {
            _CMP_EQ_OQ => lhs == rhs,
            _CMP_LT_OS => lhs < rhs,
            _CMP_LE_OS => lhs <= rhs,
            _CMP_UNORD_Q => lhs.partial_cmp(&rhs).is_none(),
            _CMP_NEQ_UQ => lhs != rhs,
            _CMP_NLT_UQ => !(lhs < rhs),
            _CMP_NLE_UQ => !(lhs <= rhs),
            _CMP_ORD_Q => lhs.partial_cmp(&rhs).is_some(),
            _CMP_EQ_UQ => lhs == rhs || lhs.partial_cmp(&rhs).is_none(),
            _CMP_NGE_US => !(lhs >= rhs),
            _CMP_NGT_US => !(lhs > rhs),
            _CMP_FALSE_OQ => false,
            _CMP_NEQ_OQ => lhs != rhs && lhs.partial_cmp(&rhs).is_some(),
            _CMP_GE_OS => lhs >= rhs,
            _CMP_GT_OS => lhs > rhs,
            _CMP_TRUE_US => true,
            _ => unreachable!(),
        };
        if res { if_t } else { if_f }
    }
    fn expected_cmp_f32(imm: i32, lhs: f32, rhs: f32) -> f32 {
        expected_cmp(imm, lhs, rhs, f32::from_bits(u32::MAX), 0.0)
    }
    fn expected_cmp_f64(imm: i32, lhs: f64, rhs: f64) -> f64 {
        expected_cmp(imm, lhs, rhs, f64::from_bits(u64::MAX), 0.0)
    }

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_cmp_ss<const IMM: i32>() {
        let values = [
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (f32::NAN, 0.0),
            (0.0, f32::NAN),
            (f32::NAN, f32::NAN),
        ];

        for (lhs, rhs) in values {
            let a = _mm_setr_ps(lhs, 2.0, 3.0, 4.0);
            let b = _mm_setr_ps(rhs, 5.0, 6.0, 7.0);
            let r: [u32; 4] = transmute(_mm_cmp_ss::<IMM>(a, b));
            let e: [u32; 4] =
                transmute(_mm_setr_ps(expected_cmp_f32(IMM, lhs, rhs), 2.0, 3.0, 4.0));
            assert_eq!(r, e);
        }
    }

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_cmp_ps<const IMM: i32>() {
        let values = [
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (f32::NAN, 0.0),
            (0.0, f32::NAN),
            (f32::NAN, f32::NAN),
        ];

        for (lhs, rhs) in values {
            let a = _mm_set1_ps(lhs);
            let b = _mm_set1_ps(rhs);
            let r: [u32; 4] = transmute(_mm_cmp_ps::<IMM>(a, b));
            let e: [u32; 4] = transmute(_mm_set1_ps(expected_cmp_f32(IMM, lhs, rhs)));
            assert_eq!(r, e);
        }
    }

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_cmp_sd<const IMM: i32>() {
        let values = [
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (f64::NAN, 0.0),
            (0.0, f64::NAN),
            (f64::NAN, f64::NAN),
        ];

        for (lhs, rhs) in values {
            let a = _mm_setr_pd(lhs, 2.0);
            let b = _mm_setr_pd(rhs, 3.0);
            let r: [u64; 2] = transmute(_mm_cmp_sd::<IMM>(a, b));
            let e: [u64; 2] = transmute(_mm_setr_pd(expected_cmp_f64(IMM, lhs, rhs), 2.0));
            assert_eq!(r, e);
        }
    }

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_cmp_pd<const IMM: i32>() {
        let values = [
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (f64::NAN, 0.0),
            (0.0, f64::NAN),
            (f64::NAN, f64::NAN),
        ];

        for (lhs, rhs) in values {
            let a = _mm_set1_pd(lhs);
            let b = _mm_set1_pd(rhs);
            let r: [u64; 2] = transmute(_mm_cmp_pd::<IMM>(a, b));
            let e: [u64; 2] = transmute(_mm_set1_pd(expected_cmp_f64(IMM, lhs, rhs)));
            assert_eq!(r, e);
        }
    }

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_cmp_ps<const IMM: i32>() {
        let values = [
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (f32::NAN, 0.0),
            (0.0, f32::NAN),
            (f32::NAN, f32::NAN),
        ];

        for (lhs, rhs) in values {
            let a = _mm256_set1_ps(lhs);
            let b = _mm256_set1_ps(rhs);
            let r: [u32; 8] = transmute(_mm256_cmp_ps::<IMM>(a, b));
            let e: [u32; 8] = transmute(_mm256_set1_ps(expected_cmp_f32(IMM, lhs, rhs)));
            assert_eq!(r, e);
        }
    }

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_cmp_pd<const IMM: i32>() {
        let values = [
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (f64::NAN, 0.0),
            (0.0, f64::NAN),
            (f64::NAN, f64::NAN),
        ];

        for (lhs, rhs) in values {
            let a = _mm256_set1_pd(lhs);
            let b = _mm256_set1_pd(rhs);
            let r: [u64; 4] = transmute(_mm256_cmp_pd::<IMM>(a, b));
            let e: [u64; 4] = transmute(_mm256_set1_pd(expected_cmp_f64(IMM, lhs, rhs)));
            assert_eq!(r, e);
        }
    }

    #[target_feature(enable = "avx")]
    unsafe fn test_cmp<const IMM: i32>() {
        test_mm_cmp_ss::<IMM>();
        test_mm_cmp_ps::<IMM>();
        test_mm_cmp_sd::<IMM>();
        test_mm_cmp_pd::<IMM>();
        test_mm256_cmp_ps::<IMM>();
        test_mm256_cmp_pd::<IMM>();
    }

    test_cmp::<_CMP_EQ_OQ>();
    test_cmp::<_CMP_LT_OS>();
    test_cmp::<_CMP_LE_OS>();
    test_cmp::<_CMP_UNORD_Q>();
    test_cmp::<_CMP_NEQ_UQ>();
    test_cmp::<_CMP_NLT_UQ>();
    test_cmp::<_CMP_NLE_UQ>();
    test_cmp::<_CMP_ORD_Q>();
    test_cmp::<_CMP_EQ_UQ>();
    test_cmp::<_CMP_NGE_US>();
    test_cmp::<_CMP_NGT_US>();
    test_cmp::<_CMP_FALSE_OQ>();
    test_cmp::<_CMP_NEQ_OQ>();
    test_cmp::<_CMP_GE_OS>();
    test_cmp::<_CMP_GT_OS>();
    test_cmp::<_CMP_TRUE_US>();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_cvtps_epi32() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let r = _mm256_cvtps_epi32(a);
        let e = _mm256_setr_epi32(4, 9, 16, 25, 4, 9, 16, 25);
        assert_eq_m256i(r, e);

        let a = _mm256_setr_ps(
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::MIN,
            f32::MAX,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        );
        let r = _mm256_cvtps_epi32(a);
        assert_eq_m256i(r, _mm256_set1_epi32(i32::MIN));
    }
    test_mm256_cvtps_epi32();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_cvttps_epi32() {
        let a = _mm256_setr_ps(4., 9., 16., 25., 4., 9., 16., 25.);
        let r = _mm256_cvttps_epi32(a);
        let e = _mm256_setr_epi32(4, 9, 16, 25, 4, 9, 16, 25);
        assert_eq_m256i(r, e);

        let a = _mm256_setr_ps(
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::MIN,
            f32::MAX,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        );
        let r = _mm256_cvttps_epi32(a);
        assert_eq_m256i(r, _mm256_set1_epi32(i32::MIN));
    }
    test_mm256_cvttps_epi32();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_cvtpd_epi32() {
        let a = _mm256_setr_pd(4., 9., 16., 25.);
        let r = _mm256_cvtpd_epi32(a);
        let e = _mm_setr_epi32(4, 9, 16, 25);
        assert_eq_m128i(r, e);

        let a = _mm256_setr_pd(f64::NEG_INFINITY, f64::INFINITY, f64::MIN, f64::MAX);
        let r = _mm256_cvtpd_epi32(a);
        assert_eq_m128i(r, _mm_set1_epi32(i32::MIN));
    }
    test_mm256_cvtpd_epi32();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_cvttpd_epi32() {
        let a = _mm256_setr_pd(4., 9., 16., 25.);
        let r = _mm256_cvttpd_epi32(a);
        let e = _mm_setr_epi32(4, 9, 16, 25);
        assert_eq_m128i(r, e);

        let a = _mm256_setr_pd(f64::NEG_INFINITY, f64::INFINITY, f64::MIN, f64::MAX);
        let r = _mm256_cvttpd_epi32(a);
        assert_eq_m128i(r, _mm_set1_epi32(i32::MIN));
    }
    test_mm256_cvttpd_epi32();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_permutevar_ps() {
        let a = _mm_setr_ps(4., 3., 2., 5.);
        let b = _mm_setr_epi32(1, 2, 3, 4);
        let r = _mm_permutevar_ps(a, b);
        let e = _mm_setr_ps(3., 2., 5., 4.);
        assert_eq_m128(r, e);
    }
    test_mm_permutevar_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_permutevar_ps() {
        let a = _mm256_setr_ps(4., 3., 2., 5., 8., 9., 64., 50.);
        let b = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm256_permutevar_ps(a, b);
        let e = _mm256_setr_ps(3., 2., 5., 4., 9., 64., 50., 8.);
        assert_eq_m256(r, e);
    }
    test_mm256_permutevar_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_permutevar_pd() {
        let a = _mm_setr_pd(4., 3.);
        let b = _mm_setr_epi64x(3, 0);
        let r = _mm_permutevar_pd(a, b);
        let e = _mm_setr_pd(3., 4.);
        assert_eq_m128d(r, e);
    }
    test_mm_permutevar_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_permutevar_pd() {
        let a = _mm256_setr_pd(4., 3., 2., 5.);
        let b = _mm256_setr_epi64x(1, 2, 3, 4);
        let r = _mm256_permutevar_pd(a, b);
        let e = _mm256_setr_pd(4., 3., 5., 2.);
        assert_eq_m256d(r, e);
    }
    test_mm256_permutevar_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_permute2f128_ps() {
        let a = _mm256_setr_ps(1., 2., 3., 4., 1., 2., 3., 4.);
        let b = _mm256_setr_ps(5., 6., 7., 8., 5., 6., 7., 8.);
        let r = _mm256_permute2f128_ps::<0x13>(a, b);
        let e = _mm256_setr_ps(5., 6., 7., 8., 1., 2., 3., 4.);
        assert_eq_m256(r, e);

        let r = _mm256_permute2f128_ps::<0x44>(a, b);
        let e = _mm256_setr_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq_m256(r, e);
    }
    test_mm256_permute2f128_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_permute2f128_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_permute2f128_pd::<0x31>(a, b);
        let e = _mm256_setr_pd(3., 4., 7., 8.);
        assert_eq_m256d(r, e);

        let r = _mm256_permute2f128_pd::<0x44>(a, b);
        let e = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
        assert_eq_m256d(r, e);
    }
    test_mm256_permute2f128_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_permute2f128_si256() {
        let a = _mm256_setr_epi32(1, 2, 3, 4, 1, 2, 3, 4);
        let b = _mm256_setr_epi32(5, 6, 7, 8, 5, 6, 7, 8);
        let r = _mm256_permute2f128_si256::<0x20>(a, b);
        let e = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq_m256i(r, e);

        let r = _mm256_permute2f128_si256::<0x44>(a, b);
        let e = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m256i(r, e);
    }
    test_mm256_permute2f128_si256();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_maskload_ps() {
        let a = &[1.0f32, 2., 3., 4.];
        let mask = _mm_setr_epi32(0, !0, 0, !0);
        let r = _mm_maskload_ps(a.as_ptr(), mask);
        let e = _mm_setr_ps(0., 2., 0., 4.);
        assert_eq_m128(r, e);

        // Unaligned pointer
        let a = Unaligned::new([1.0f32, 2., 3., 4.]);
        let mask = _mm_setr_epi32(0, !0, 0, !0);
        let r = _mm_maskload_ps(a.as_ptr().cast(), mask);
        let e = _mm_setr_ps(0., 2., 0., 4.);
        assert_eq_m128(r, e);

        // Only loading first element, so slice can be short.
        let a = &[2.0f32];
        let mask = _mm_setr_epi32(!0, 0, 0, 0);
        let r = _mm_maskload_ps(a.as_ptr(), mask);
        let e = _mm_setr_ps(2.0, 0.0, 0.0, 0.0);
        assert_eq_m128(r, e);

        // Only loading last element, so slice can be short.
        let a = &[2.0f32];
        let mask = _mm_setr_epi32(0, 0, 0, !0);
        let r = _mm_maskload_ps(a.as_ptr().wrapping_sub(3), mask);
        let e = _mm_setr_ps(0.0, 0.0, 0.0, 2.0);
        assert_eq_m128(r, e);
    }
    test_mm_maskload_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_maskload_pd() {
        let a = &[1.0f64, 2.];
        let mask = _mm_setr_epi64x(0, !0);
        let r = _mm_maskload_pd(a.as_ptr(), mask);
        let e = _mm_setr_pd(0., 2.);
        assert_eq_m128d(r, e);

        // Unaligned pointer
        let a = Unaligned::new([1.0f64, 2.]);
        let mask = _mm_setr_epi64x(0, !0);
        let r = _mm_maskload_pd(a.as_ptr().cast(), mask);
        let e = _mm_setr_pd(0., 2.);
        assert_eq_m128d(r, e);

        // Only loading first element, so slice can be short.
        let a = &[2.0f64];
        let mask = _mm_setr_epi64x(!0, 0);
        let r = _mm_maskload_pd(a.as_ptr(), mask);
        let e = _mm_setr_pd(2.0, 0.0);
        assert_eq_m128d(r, e);

        // Only loading last element, so slice can be short.
        let a = &[2.0f64];
        let mask = _mm_setr_epi64x(0, !0);
        let r = _mm_maskload_pd(a.as_ptr().wrapping_sub(1), mask);
        let e = _mm_setr_pd(0.0, 2.0);
        assert_eq_m128d(r, e);
    }
    test_mm_maskload_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_maskload_ps() {
        let a = &[1.0f32, 2., 3., 4., 5., 6., 7., 8.];
        let mask = _mm256_setr_epi32(0, !0, 0, !0, 0, !0, 0, !0);
        let r = _mm256_maskload_ps(a.as_ptr(), mask);
        let e = _mm256_setr_ps(0., 2., 0., 4., 0., 6., 0., 8.);
        assert_eq_m256(r, e);

        // Unaligned pointer
        let a = Unaligned::new([1.0f32, 2., 3., 4., 5., 6., 7., 8.]);
        let mask = _mm256_setr_epi32(0, !0, 0, !0, 0, !0, 0, !0);
        let r = _mm256_maskload_ps(a.as_ptr().cast(), mask);
        let e = _mm256_setr_ps(0., 2., 0., 4., 0., 6., 0., 8.);
        assert_eq_m256(r, e);

        // Only loading first element, so slice can be short.
        let a = &[2.0f32];
        let mask = _mm256_setr_epi32(!0, 0, 0, 0, 0, 0, 0, 0);
        let r = _mm256_maskload_ps(a.as_ptr(), mask);
        let e = _mm256_setr_ps(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq_m256(r, e);

        // Only loading last element, so slice can be short.
        let a = &[2.0f32];
        let mask = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, !0);
        let r = _mm256_maskload_ps(a.as_ptr().wrapping_sub(7), mask);
        let e = _mm256_setr_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0);
        assert_eq_m256(r, e);
    }
    test_mm256_maskload_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_maskload_pd() {
        let a = &[1.0f64, 2., 3., 4.];
        let mask = _mm256_setr_epi64x(0, !0, 0, !0);
        let r = _mm256_maskload_pd(a.as_ptr(), mask);
        let e = _mm256_setr_pd(0., 2., 0., 4.);
        assert_eq_m256d(r, e);

        // Unaligned pointer
        let a = Unaligned::new([1.0f64, 2., 3., 4.]);
        let mask = _mm256_setr_epi64x(0, !0, 0, !0);
        let r = _mm256_maskload_pd(a.as_ptr().cast(), mask);
        let e = _mm256_setr_pd(0., 2., 0., 4.);
        assert_eq_m256d(r, e);

        // Only loading first element, so slice can be short.
        let a = &[2.0f64];
        let mask = _mm256_setr_epi64x(!0, 0, 0, 0);
        let r = _mm256_maskload_pd(a.as_ptr(), mask);
        let e = _mm256_setr_pd(2.0, 0.0, 0.0, 0.0);
        assert_eq_m256d(r, e);

        // Only loading last element, so slice can be short.
        let a = &[2.0f64];
        let mask = _mm256_setr_epi64x(0, 0, 0, !0);
        let r = _mm256_maskload_pd(a.as_ptr().wrapping_sub(3), mask);
        let e = _mm256_setr_pd(0.0, 0.0, 0.0, 2.0);
        assert_eq_m256d(r, e);
    }
    test_mm256_maskload_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_maskstore_ps() {
        let mut r = _mm_set1_ps(0.);
        let mask = _mm_setr_epi32(0, !0, 0, !0);
        let a = _mm_setr_ps(1., 2., 3., 4.);
        _mm_maskstore_ps(&mut r as *mut _ as *mut f32, mask, a);
        let e = _mm_setr_ps(0., 2., 0., 4.);
        assert_eq_m128(r, e);

        // Unaligned pointer
        let mut r = Unaligned::new([0.0f32; 4]);
        let mask = _mm_setr_epi32(0, !0, 0, !0);
        let a = _mm_setr_ps(1., 2., 3., 4.);
        _mm_maskstore_ps(r.as_mut_ptr().cast(), mask, a);
        let e = [0., 2., 0., 4.];
        assert_eq!(r.read(), e);

        // Only storing first element, so slice can be short.
        let mut r = [0.0f32];
        let mask = _mm_setr_epi32(!0, 0, 0, 0);
        let a = _mm_setr_ps(1., 2., 3., 4.);
        _mm_maskstore_ps(r.as_mut_ptr(), mask, a);
        let e = [1.0f32];
        assert_eq!(r, e);

        // Only storing last element, so slice can be short.
        let mut r = [0.0f32];
        let mask = _mm_setr_epi32(0, 0, 0, !0);
        let a = _mm_setr_ps(1., 2., 3., 4.);
        _mm_maskstore_ps(r.as_mut_ptr().wrapping_sub(3), mask, a);
        let e = [4.0f32];
        assert_eq!(r, e);
    }
    test_mm_maskstore_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_maskstore_pd() {
        let mut r = _mm_set1_pd(0.);
        let mask = _mm_setr_epi64x(0, !0);
        let a = _mm_setr_pd(1., 2.);
        _mm_maskstore_pd(&mut r as *mut _ as *mut f64, mask, a);
        let e = _mm_setr_pd(0., 2.);
        assert_eq_m128d(r, e);

        // Unaligned pointer
        let mut r = Unaligned::new([0.0f64; 2]);
        let mask = _mm_setr_epi64x(0, !0);
        let a = _mm_setr_pd(1., 2.);
        _mm_maskstore_pd(r.as_mut_ptr().cast(), mask, a);
        let e = [0., 2.];
        assert_eq!(r.read(), e);

        // Only storing first element, so slice can be short.
        let mut r = [0.0f64];
        let mask = _mm_setr_epi64x(!0, 0);
        let a = _mm_setr_pd(1., 2.);
        _mm_maskstore_pd(r.as_mut_ptr(), mask, a);
        let e = [1.0f64];
        assert_eq!(r, e);

        // Only storing last element, so slice can be short.
        let mut r = [0.0f64];
        let mask = _mm_setr_epi64x(0, !0);
        let a = _mm_setr_pd(1., 2.);
        _mm_maskstore_pd(r.as_mut_ptr().wrapping_sub(1), mask, a);
        let e = [2.0f64];
        assert_eq!(r, e);
    }
    test_mm_maskstore_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_maskstore_ps() {
        let mut r = _mm256_set1_ps(0.);
        let mask = _mm256_setr_epi32(0, !0, 0, !0, 0, !0, 0, !0);
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        _mm256_maskstore_ps(&mut r as *mut _ as *mut f32, mask, a);
        let e = _mm256_setr_ps(0., 2., 0., 4., 0., 6., 0., 8.);
        assert_eq_m256(r, e);

        // Unaligned pointer
        let mut r = Unaligned::new([0.0f32; 8]);
        let mask = _mm256_setr_epi32(0, !0, 0, !0, 0, !0, 0, !0);
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        _mm256_maskstore_ps(r.as_mut_ptr().cast(), mask, a);
        let e = [0., 2., 0., 4., 0., 6., 0., 8.];
        assert_eq!(r.read(), e);

        // Only storing first element, so slice can be short.
        let mut r = [0.0f32];
        let mask = _mm256_setr_epi32(!0, 0, 0, 0, 0, 0, 0, 0);
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        _mm256_maskstore_ps(r.as_mut_ptr(), mask, a);
        let e = [1.0f32];
        assert_eq!(r, e);

        // Only storing last element, so slice can be short.
        let mut r = [0.0f32];
        let mask = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, !0);
        let a = _mm256_setr_ps(1., 2., 3., 4., 5., 6., 7., 8.);
        _mm256_maskstore_ps(r.as_mut_ptr().wrapping_sub(7), mask, a);
        let e = [8.0f32];
        assert_eq!(r, e);
    }
    test_mm256_maskstore_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_maskstore_pd() {
        let mut r = _mm256_set1_pd(0.);
        let mask = _mm256_setr_epi64x(0, !0, 0, !0);
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        _mm256_maskstore_pd(&mut r as *mut _ as *mut f64, mask, a);
        let e = _mm256_setr_pd(0., 2., 0., 4.);
        assert_eq_m256d(r, e);

        // Unaligned pointer
        let mut r = Unaligned::new([0.0f64; 4]);
        let mask = _mm256_setr_epi64x(0, !0, 0, !0);
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        _mm256_maskstore_pd(r.as_mut_ptr().cast(), mask, a);
        let e = [0., 2., 0., 4.];
        assert_eq!(r.read(), e);

        // Only storing first element, so slice can be short.
        let mut r = [0.0f64];
        let mask = _mm256_setr_epi64x(!0, 0, 0, 0);
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        _mm256_maskstore_pd(r.as_mut_ptr(), mask, a);
        let e = [1.0f64];
        assert_eq!(r, e);

        // Only storing last element, so slice can be short.
        let mut r = [0.0f64];
        let mask = _mm256_setr_epi64x(0, 0, 0, !0);
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        _mm256_maskstore_pd(r.as_mut_ptr().wrapping_sub(3), mask, a);
        let e = [4.0f64];
        assert_eq!(r, e);
    }
    test_mm256_maskstore_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_lddqu_si256() {
        #[rustfmt::skip]
        let a = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let p = &a as *const _;
        let r = _mm256_lddqu_si256(p);
        #[rustfmt::skip]
        let e = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        assert_eq_m256i(r, e);
    }
    test_mm256_lddqu_si256();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_testz_si256() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let b = _mm256_setr_epi64x(5, 6, 7, 8);
        let r = _mm256_testz_si256(a, b);
        assert_eq!(r, 0);
        let b = _mm256_set1_epi64x(0);
        let r = _mm256_testz_si256(a, b);
        assert_eq!(r, 1);
    }
    test_mm256_testz_si256();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_testc_si256() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let b = _mm256_setr_epi64x(5, 6, 7, 8);
        let r = _mm256_testc_si256(a, b);
        assert_eq!(r, 0);
        let b = _mm256_set1_epi64x(0);
        let r = _mm256_testc_si256(a, b);
        assert_eq!(r, 1);
    }
    test_mm256_testc_si256();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_testnzc_si256() {
        let a = _mm256_setr_epi64x(1, 2, 3, 4);
        let b = _mm256_setr_epi64x(5, 6, 7, 8);
        let r = _mm256_testnzc_si256(a, b);
        assert_eq!(r, 1);
        let a = _mm256_setr_epi64x(0, 0, 0, 0);
        let b = _mm256_setr_epi64x(0, 0, 0, 0);
        let r = _mm256_testnzc_si256(a, b);
        assert_eq!(r, 0);
    }
    test_mm256_testnzc_si256();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_testz_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_testz_pd(a, b);
        assert_eq!(r, 1);
        let a = _mm256_set1_pd(-1.);
        let r = _mm256_testz_pd(a, a);
        assert_eq!(r, 0);
    }
    test_mm256_testz_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_testc_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_testc_pd(a, b);
        assert_eq!(r, 1);
        let a = _mm256_set1_pd(1.);
        let b = _mm256_set1_pd(-1.);
        let r = _mm256_testc_pd(a, b);
        assert_eq!(r, 0);
    }
    test_mm256_testc_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_testnzc_pd() {
        let a = _mm256_setr_pd(1., 2., 3., 4.);
        let b = _mm256_setr_pd(5., 6., 7., 8.);
        let r = _mm256_testnzc_pd(a, b);
        assert_eq!(r, 0);
        let a = _mm256_setr_pd(1., -1., -1., -1.);
        let b = _mm256_setr_pd(-1., -1., 1., 1.);
        let r = _mm256_testnzc_pd(a, b);
        assert_eq!(r, 1);
    }
    test_mm256_testnzc_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_testz_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 6.);
        let r = _mm_testz_pd(a, b);
        assert_eq!(r, 1);
        let a = _mm_set1_pd(-1.);
        let r = _mm_testz_pd(a, a);
        assert_eq!(r, 0);
    }
    test_mm_testz_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_testc_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 6.);
        let r = _mm_testc_pd(a, b);
        assert_eq!(r, 1);
        let a = _mm_set1_pd(1.);
        let b = _mm_set1_pd(-1.);
        let r = _mm_testc_pd(a, b);
        assert_eq!(r, 0);
    }
    test_mm_testc_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_testnzc_pd() {
        let a = _mm_setr_pd(1., 2.);
        let b = _mm_setr_pd(5., 6.);
        let r = _mm_testnzc_pd(a, b);
        assert_eq!(r, 0);
        let a = _mm_setr_pd(1., -1.);
        let b = _mm_setr_pd(-1., -1.);
        let r = _mm_testnzc_pd(a, b);
        assert_eq!(r, 1);
    }
    test_mm_testnzc_pd();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_testz_ps() {
        let a = _mm256_set1_ps(1.);
        let r = _mm256_testz_ps(a, a);
        assert_eq!(r, 1);
        let a = _mm256_set1_ps(-1.);
        let r = _mm256_testz_ps(a, a);
        assert_eq!(r, 0);
    }
    test_mm256_testz_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_testc_ps() {
        let a = _mm256_set1_ps(1.);
        let r = _mm256_testc_ps(a, a);
        assert_eq!(r, 1);
        let b = _mm256_set1_ps(-1.);
        let r = _mm256_testc_ps(a, b);
        assert_eq!(r, 0);
    }
    test_mm256_testc_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm256_testnzc_ps() {
        let a = _mm256_set1_ps(1.);
        let r = _mm256_testnzc_ps(a, a);
        assert_eq!(r, 0);
        let a = _mm256_setr_ps(1., -1., -1., -1., -1., -1., -1., -1.);
        let b = _mm256_setr_ps(-1., -1., 1., 1., 1., 1., 1., 1.);
        let r = _mm256_testnzc_ps(a, b);
        assert_eq!(r, 1);
    }
    test_mm256_testnzc_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_testz_ps() {
        let a = _mm_set1_ps(1.);
        let r = _mm_testz_ps(a, a);
        assert_eq!(r, 1);
        let a = _mm_set1_ps(-1.);
        let r = _mm_testz_ps(a, a);
        assert_eq!(r, 0);
    }
    test_mm_testz_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_testc_ps() {
        let a = _mm_set1_ps(1.);
        let r = _mm_testc_ps(a, a);
        assert_eq!(r, 1);
        let b = _mm_set1_ps(-1.);
        let r = _mm_testc_ps(a, b);
        assert_eq!(r, 0);
    }
    test_mm_testc_ps();

    #[target_feature(enable = "avx")]
    unsafe fn test_mm_testnzc_ps() {
        let a = _mm_set1_ps(1.);
        let r = _mm_testnzc_ps(a, a);
        assert_eq!(r, 0);
        let a = _mm_setr_ps(1., -1., -1., -1.);
        let b = _mm_setr_ps(-1., -1., 1., 1.);
        let r = _mm_testnzc_ps(a, b);
        assert_eq!(r, 1);
    }
    test_mm_testnzc_ps();

    // These intrinsics are functionally no-ops. The only thing
    // that needs to be tested is that they can be executed.
    _mm256_zeroupper();
    _mm256_zeroall();
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
