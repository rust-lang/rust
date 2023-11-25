// Ignore everything except x86 and x86_64
// Any new targets that are added to CI should be ignored here.
// (We cannot use `cfg`-based tricks here since the `target-feature` flags below only work on x86.)
//@ignore-target-aarch64
//@ignore-target-arm
//@ignore-target-avr
//@ignore-target-s390x
//@ignore-target-thumbv7em
//@ignore-target-wasm32
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
    unsafe fn test_cmp<const IMM: i32>() {
        test_mm_cmp_ss::<IMM>();
        test_mm_cmp_ps::<IMM>();
        test_mm_cmp_sd::<IMM>();
        test_mm_cmp_pd::<IMM>();
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
}
