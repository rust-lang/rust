use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::unpack;
use rustc_apfloat::{Category, ExpInt, IEK_INF, IEK_NAN, IEK_ZERO};
use rustc_apfloat::{Float, Status};

mod common;

use common::DoubleExt;

#[test]
fn add() {
    // Test Special Cases against each other and normal values.

    // FIXMES/NOTES:
    // 1. Since we perform only default exception handling all operations with
    // signaling NaNs should have a result that is a quiet NaN. Currently they
    // return sNaN.

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let qnan = Single::NAN;
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    let overflow_status = Status::OVERFLOW | Status::INEXACT;

    let special_cases = [
        (p_inf, p_inf, "inf", Status::OK, Category::Infinity),
        (p_inf, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, p_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, m_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_inf, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_inf, p_normal_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_normal_value, "inf", Status::OK, Category::Infinity),
        (p_inf, p_largest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_largest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (m_inf, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_inf, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_inf, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_inf, p_normal_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_normal_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_largest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_largest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (p_zero, p_inf, "inf", Status::OK, Category::Infinity),
        (p_zero, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_zero, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_zero, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_zero, p_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (p_zero, m_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (p_zero, p_largest_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_zero, m_largest_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (p_zero, p_smallest_value, "0x1p-149", Status::OK, Category::Normal),
        (p_zero, m_smallest_value, "-0x1p-149", Status::OK, Category::Normal),
        (p_zero, p_smallest_normalized, "0x1p-126", Status::OK, Category::Normal),
        (p_zero, m_smallest_normalized, "-0x1p-126", Status::OK, Category::Normal),
        (m_zero, p_inf, "inf", Status::OK, Category::Infinity),
        (m_zero, m_inf, "-inf", Status::OK, Category::Infinity),
        (m_zero, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_zero, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_zero, p_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (m_zero, m_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (m_zero, p_largest_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (m_zero, m_largest_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_zero, p_smallest_value, "0x1p-149", Status::OK, Category::Normal),
        (m_zero, m_smallest_value, "-0x1p-149", Status::OK, Category::Normal),
        (m_zero, p_smallest_normalized, "0x1p-126", Status::OK, Category::Normal),
        (m_zero, m_smallest_normalized, "-0x1p-126", Status::OK, Category::Normal),
        (qnan, p_inf, "nan", Status::OK, Category::NaN),
        (qnan, m_inf, "nan", Status::OK, Category::NaN),
        (qnan, p_zero, "nan", Status::OK, Category::NaN),
        (qnan, m_zero, "nan", Status::OK, Category::NaN),
        (qnan, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (qnan, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (qnan, p_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, m_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, p_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_normalized, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_normalized, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (snan, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (snan, qnan, "nan", Status::INVALID_OP, Category::NaN),
        (snan, snan, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_normal_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_normal_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_largest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_largest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_normal_value, p_inf, "inf", Status::OK, Category::Infinity),
        (p_normal_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_normal_value, p_zero, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, m_zero, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_normal_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_normal_value, p_normal_value, "0x1p+1", Status::OK, Category::Normal),
        (p_normal_value, m_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, p_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_normal_value, m_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_normal_value, p_smallest_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_normal_value, m_smallest_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_normal_value, p_smallest_normalized, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_normal_value, m_smallest_normalized, "0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, p_inf, "inf", Status::OK, Category::Infinity),
        (m_normal_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (m_normal_value, p_zero, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, m_zero, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_normal_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_normal_value, p_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, m_normal_value, "-0x1p+1", Status::OK, Category::Normal),
        (m_normal_value, p_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_normal_value, m_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_normal_value, p_smallest_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, m_smallest_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, p_smallest_normalized, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, m_smallest_normalized, "-0x1p+0", Status::INEXACT, Category::Normal),
        (p_largest_value, p_inf, "inf", Status::OK, Category::Infinity),
        (p_largest_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_largest_value, p_zero, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, m_zero, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_largest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_largest_value, p_normal_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, m_normal_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, p_largest_value, "inf", overflow_status, Category::Infinity),
        (p_largest_value, m_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, p_smallest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, m_smallest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (
            p_largest_value,
            p_smallest_normalized,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_smallest_normalized,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (m_largest_value, p_inf, "inf", Status::OK, Category::Infinity),
        (m_largest_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (m_largest_value, p_zero, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, m_zero, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_largest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_largest_value, p_normal_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, m_normal_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, p_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, m_largest_value, "-inf", overflow_status, Category::Infinity),
        (m_largest_value, p_smallest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, m_smallest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (
            m_largest_value,
            p_smallest_normalized,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_smallest_normalized,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (p_smallest_value, p_inf, "inf", Status::OK, Category::Infinity),
        (p_smallest_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_smallest_value, p_zero, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, m_zero, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_smallest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_smallest_value, p_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_smallest_value, m_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (p_smallest_value, p_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_smallest_value, m_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_smallest_value, p_smallest_value, "0x1p-148", Status::OK, Category::Normal),
        (p_smallest_value, m_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, p_smallest_normalized, "0x1.000002p-126", Status::OK, Category::Normal),
        (p_smallest_value, m_smallest_normalized, "-0x1.fffffcp-127", Status::OK, Category::Normal),
        (m_smallest_value, p_inf, "inf", Status::OK, Category::Infinity),
        (m_smallest_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (m_smallest_value, p_zero, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, m_zero, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_smallest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_smallest_value, p_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (m_smallest_value, m_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_smallest_value, p_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_smallest_value, m_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_smallest_value, p_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_value, m_smallest_value, "-0x1p-148", Status::OK, Category::Normal),
        (m_smallest_value, p_smallest_normalized, "0x1.fffffcp-127", Status::OK, Category::Normal),
        (m_smallest_value, m_smallest_normalized, "-0x1.000002p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, p_inf, "inf", Status::OK, Category::Infinity),
        (p_smallest_normalized, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_smallest_normalized, p_zero, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, m_zero, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_smallest_normalized, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_smallest_normalized, p_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_smallest_normalized, m_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (
            p_smallest_normalized,
            p_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (p_smallest_normalized, p_smallest_value, "0x1.000002p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, m_smallest_value, "0x1.fffffcp-127", Status::OK, Category::Normal),
        (p_smallest_normalized, p_smallest_normalized, "0x1p-125", Status::OK, Category::Normal),
        (p_smallest_normalized, m_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, p_inf, "inf", Status::OK, Category::Infinity),
        (m_smallest_normalized, m_inf, "-inf", Status::OK, Category::Infinity),
        (m_smallest_normalized, p_zero, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, m_zero, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_smallest_normalized, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_smallest_normalized, p_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (m_smallest_normalized, m_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (
            m_smallest_normalized,
            p_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (m_smallest_normalized, p_smallest_value, "-0x1.fffffcp-127", Status::OK, Category::Normal),
        (m_smallest_normalized, m_smallest_value, "-0x1.000002p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, p_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, m_smallest_normalized, "-0x1p-125", Status::OK, Category::Normal),
    ];

    for (x, y, e_result, e_status, e_category) in special_cases {
        let status;
        let result = unpack!(status=, x + y);
        assert_eq!(status, e_status);
        assert_eq!(result.category(), e_category);
        assert!(result.bitwise_eq(e_result.parse::<Single>().unwrap()));
    }
}

#[test]
fn subtract() {
    // Test Special Cases against each other and normal values.

    // FIXMES/NOTES:
    // 1. Since we perform only default exception handling all operations with
    // signaling NaNs should have a result that is a quiet NaN. Currently they
    // return sNaN.

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let qnan = Single::NAN;
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    let overflow_status = Status::OVERFLOW | Status::INEXACT;

    let special_cases = [
        (p_inf, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_inf, "inf", Status::OK, Category::Infinity),
        (p_inf, p_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, m_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, qnan, "-nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_inf, snan, "-nan", Status::INVALID_OP, Category::NaN),
                */
        (p_inf, p_normal_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_normal_value, "inf", Status::OK, Category::Infinity),
        (p_inf, p_largest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_largest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (m_inf, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, p_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, qnan, "-nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_inf, snan, "-nan", Status::INVALID_OP, Category::NaN),
                */
        (m_inf, p_normal_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_normal_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_largest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_largest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (p_zero, p_inf, "-inf", Status::OK, Category::Infinity),
        (p_zero, m_inf, "inf", Status::OK, Category::Infinity),
        (p_zero, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, qnan, "-nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_zero, snan, "-nan", Status::INVALID_OP, Category::NaN),
                */
        (p_zero, p_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (p_zero, m_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (p_zero, p_largest_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (p_zero, m_largest_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_zero, p_smallest_value, "-0x1p-149", Status::OK, Category::Normal),
        (p_zero, m_smallest_value, "0x1p-149", Status::OK, Category::Normal),
        (p_zero, p_smallest_normalized, "-0x1p-126", Status::OK, Category::Normal),
        (p_zero, m_smallest_normalized, "0x1p-126", Status::OK, Category::Normal),
        (m_zero, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_zero, m_inf, "inf", Status::OK, Category::Infinity),
        (m_zero, p_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, qnan, "-nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_zero, snan, "-nan", Status::INVALID_OP, Category::NaN),
                */
        (m_zero, p_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (m_zero, m_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (m_zero, p_largest_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_zero, m_largest_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (m_zero, p_smallest_value, "-0x1p-149", Status::OK, Category::Normal),
        (m_zero, m_smallest_value, "0x1p-149", Status::OK, Category::Normal),
        (m_zero, p_smallest_normalized, "-0x1p-126", Status::OK, Category::Normal),
        (m_zero, m_smallest_normalized, "0x1p-126", Status::OK, Category::Normal),
        (qnan, p_inf, "nan", Status::OK, Category::NaN),
        (qnan, m_inf, "nan", Status::OK, Category::NaN),
        (qnan, p_zero, "nan", Status::OK, Category::NaN),
        (qnan, m_zero, "nan", Status::OK, Category::NaN),
        (qnan, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (qnan, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (qnan, p_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, m_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, p_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_normalized, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_normalized, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (snan, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (snan, qnan, "nan", Status::INVALID_OP, Category::NaN),
        (snan, snan, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_normal_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_normal_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_largest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_largest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_normal_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (p_normal_value, m_inf, "inf", Status::OK, Category::Infinity),
        (p_normal_value, p_zero, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, m_zero, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, qnan, "-nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_normal_value, snan, "-nan", Status::INVALID_OP, Category::NaN),
                */
        (p_normal_value, p_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, m_normal_value, "0x1p+1", Status::OK, Category::Normal),
        (p_normal_value, p_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_normal_value, m_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_normal_value, p_smallest_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_normal_value, m_smallest_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_normal_value, p_smallest_normalized, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_normal_value, m_smallest_normalized, "0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_normal_value, m_inf, "inf", Status::OK, Category::Infinity),
        (m_normal_value, p_zero, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, m_zero, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, qnan, "-nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_normal_value, snan, "-nan", Status::INVALID_OP, Category::NaN),
                */
        (m_normal_value, p_normal_value, "-0x1p+1", Status::OK, Category::Normal),
        (m_normal_value, m_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, p_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_normal_value, m_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_normal_value, p_smallest_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, m_smallest_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, p_smallest_normalized, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, m_smallest_normalized, "-0x1p+0", Status::INEXACT, Category::Normal),
        (p_largest_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (p_largest_value, m_inf, "inf", Status::OK, Category::Infinity),
        (p_largest_value, p_zero, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, m_zero, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, qnan, "-nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_largest_value, snan, "-nan", Status::INVALID_OP, Category::NaN),
                */
        (p_largest_value, p_normal_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, m_normal_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, p_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, m_largest_value, "inf", overflow_status, Category::Infinity),
        (p_largest_value, p_smallest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, m_smallest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (
            p_largest_value,
            p_smallest_normalized,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_smallest_normalized,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (m_largest_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_largest_value, m_inf, "inf", Status::OK, Category::Infinity),
        (m_largest_value, p_zero, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, m_zero, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, qnan, "-nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_largest_value, snan, "-nan", Status::INVALID_OP, Category::NaN),
                */
        (m_largest_value, p_normal_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, m_normal_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, p_largest_value, "-inf", overflow_status, Category::Infinity),
        (m_largest_value, m_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, p_smallest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, m_smallest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (
            m_largest_value,
            p_smallest_normalized,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_smallest_normalized,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (p_smallest_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (p_smallest_value, m_inf, "inf", Status::OK, Category::Infinity),
        (p_smallest_value, p_zero, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, m_zero, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, qnan, "-nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_smallest_value, snan, "-nan", Status::INVALID_OP, Category::NaN),
                */
        (p_smallest_value, p_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (p_smallest_value, m_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_smallest_value, p_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_smallest_value, m_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_smallest_value, p_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, m_smallest_value, "0x1p-148", Status::OK, Category::Normal),
        (p_smallest_value, p_smallest_normalized, "-0x1.fffffcp-127", Status::OK, Category::Normal),
        (p_smallest_value, m_smallest_normalized, "0x1.000002p-126", Status::OK, Category::Normal),
        (m_smallest_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_smallest_value, m_inf, "inf", Status::OK, Category::Infinity),
        (m_smallest_value, p_zero, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, m_zero, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, qnan, "-nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_smallest_value, snan, "-nan", Status::INVALID_OP, Category::NaN),
                */
        (m_smallest_value, p_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_smallest_value, m_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (m_smallest_value, p_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_smallest_value, m_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_smallest_value, p_smallest_value, "-0x1p-148", Status::OK, Category::Normal),
        (m_smallest_value, m_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_value, p_smallest_normalized, "-0x1.000002p-126", Status::OK, Category::Normal),
        (m_smallest_value, m_smallest_normalized, "0x1.fffffcp-127", Status::OK, Category::Normal),
        (p_smallest_normalized, p_inf, "-inf", Status::OK, Category::Infinity),
        (p_smallest_normalized, m_inf, "inf", Status::OK, Category::Infinity),
        (p_smallest_normalized, p_zero, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, m_zero, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, qnan, "-nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_smallest_normalized, snan, "-nan", Status::INVALID_OP, Category::NaN),
                */
        (p_smallest_normalized, p_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (p_smallest_normalized, m_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (
            p_smallest_normalized,
            p_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (p_smallest_normalized, p_smallest_value, "0x1.fffffcp-127", Status::OK, Category::Normal),
        (p_smallest_normalized, m_smallest_value, "0x1.000002p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, p_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_normalized, m_smallest_normalized, "0x1p-125", Status::OK, Category::Normal),
        (m_smallest_normalized, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_smallest_normalized, m_inf, "inf", Status::OK, Category::Infinity),
        (m_smallest_normalized, p_zero, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, m_zero, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, qnan, "-nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_smallest_normalized, snan, "-nan", Status::INVALID_OP, Category::NaN),
                */
        (m_smallest_normalized, p_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_smallest_normalized, m_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (
            m_smallest_normalized,
            p_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (m_smallest_normalized, p_smallest_value, "-0x1.000002p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, m_smallest_value, "-0x1.fffffcp-127", Status::OK, Category::Normal),
        (m_smallest_normalized, p_smallest_normalized, "-0x1p-125", Status::OK, Category::Normal),
        (m_smallest_normalized, m_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
    ];

    for (x, y, e_result, e_status, e_category) in special_cases {
        let status;
        let result = unpack!(status=, x - y);
        assert_eq!(status, e_status);
        assert_eq!(result.category(), e_category);
        assert!(result.bitwise_eq(e_result.parse::<Single>().unwrap()));
    }
}

#[test]
fn multiply() {
    // Test Special Cases against each other and normal values.

    // FIXMES/NOTES:
    // 1. Since we perform only default exception handling all operations with
    // signaling NaNs should have a result that is a quiet NaN. Currently they
    // return sNaN.

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let qnan = Single::NAN;
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    let overflow_status = Status::OVERFLOW | Status::INEXACT;
    let underflow_status = Status::UNDERFLOW | Status::INEXACT;

    let special_cases = [
        (p_inf, p_inf, "inf", Status::OK, Category::Infinity),
        (p_inf, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_inf, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_inf, p_normal_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_normal_value, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_largest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_largest_value, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_value, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_inf, "inf", Status::OK, Category::Infinity),
        (m_inf, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_inf, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_inf, p_normal_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_normal_value, "inf", Status::OK, Category::Infinity),
        (m_inf, p_largest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_largest_value, "inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_value, "inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (p_zero, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_zero, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_zero, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_zero, p_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_normal_value, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_largest_value, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_zero, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_zero, p_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_zero, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_zero, p_normal_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_largest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (qnan, p_inf, "nan", Status::OK, Category::NaN),
        (qnan, m_inf, "nan", Status::OK, Category::NaN),
        (qnan, p_zero, "nan", Status::OK, Category::NaN),
        (qnan, m_zero, "nan", Status::OK, Category::NaN),
        (qnan, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (qnan, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (qnan, p_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, m_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, p_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_normalized, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_normalized, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (snan, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (snan, qnan, "nan", Status::INVALID_OP, Category::NaN),
        (snan, snan, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_normal_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_normal_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_largest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_largest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_normal_value, p_inf, "inf", Status::OK, Category::Infinity),
        (p_normal_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_normal_value, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, m_zero, "-0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_normal_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_normal_value, p_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, m_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, p_largest_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_normal_value, m_largest_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (p_normal_value, p_smallest_value, "0x1p-149", Status::OK, Category::Normal),
        (p_normal_value, m_smallest_value, "-0x1p-149", Status::OK, Category::Normal),
        (p_normal_value, p_smallest_normalized, "0x1p-126", Status::OK, Category::Normal),
        (p_normal_value, m_smallest_normalized, "-0x1p-126", Status::OK, Category::Normal),
        (m_normal_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_normal_value, m_inf, "inf", Status::OK, Category::Infinity),
        (m_normal_value, p_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_normal_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_normal_value, p_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, m_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, p_largest_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_normal_value, m_largest_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (m_normal_value, p_smallest_value, "-0x1p-149", Status::OK, Category::Normal),
        (m_normal_value, m_smallest_value, "0x1p-149", Status::OK, Category::Normal),
        (m_normal_value, p_smallest_normalized, "-0x1p-126", Status::OK, Category::Normal),
        (m_normal_value, m_smallest_normalized, "0x1p-126", Status::OK, Category::Normal),
        (p_largest_value, p_inf, "inf", Status::OK, Category::Infinity),
        (p_largest_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_largest_value, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, m_zero, "-0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_largest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_largest_value, p_normal_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, m_normal_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, p_largest_value, "inf", overflow_status, Category::Infinity),
        (p_largest_value, m_largest_value, "-inf", overflow_status, Category::Infinity),
        (p_largest_value, p_smallest_value, "0x1.fffffep-22", Status::OK, Category::Normal),
        (p_largest_value, m_smallest_value, "-0x1.fffffep-22", Status::OK, Category::Normal),
        (p_largest_value, p_smallest_normalized, "0x1.fffffep+1", Status::OK, Category::Normal),
        (p_largest_value, m_smallest_normalized, "-0x1.fffffep+1", Status::OK, Category::Normal),
        (m_largest_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_largest_value, m_inf, "inf", Status::OK, Category::Infinity),
        (m_largest_value, p_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_largest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_largest_value, p_normal_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, m_normal_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, p_largest_value, "-inf", overflow_status, Category::Infinity),
        (m_largest_value, m_largest_value, "inf", overflow_status, Category::Infinity),
        (m_largest_value, p_smallest_value, "-0x1.fffffep-22", Status::OK, Category::Normal),
        (m_largest_value, m_smallest_value, "0x1.fffffep-22", Status::OK, Category::Normal),
        (m_largest_value, p_smallest_normalized, "-0x1.fffffep+1", Status::OK, Category::Normal),
        (m_largest_value, m_smallest_normalized, "0x1.fffffep+1", Status::OK, Category::Normal),
        (p_smallest_value, p_inf, "inf", Status::OK, Category::Infinity),
        (p_smallest_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_smallest_value, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, m_zero, "-0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_smallest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_smallest_value, p_normal_value, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, m_normal_value, "-0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, p_largest_value, "0x1.fffffep-22", Status::OK, Category::Normal),
        (p_smallest_value, m_largest_value, "-0x1.fffffep-22", Status::OK, Category::Normal),
        (p_smallest_value, p_smallest_value, "0x0p+0", underflow_status, Category::Zero),
        (p_smallest_value, m_smallest_value, "-0x0p+0", underflow_status, Category::Zero),
        (p_smallest_value, p_smallest_normalized, "0x0p+0", underflow_status, Category::Zero),
        (p_smallest_value, m_smallest_normalized, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_smallest_value, m_inf, "inf", Status::OK, Category::Infinity),
        (m_smallest_value, p_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_smallest_value, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_smallest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_smallest_value, p_normal_value, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, m_normal_value, "0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, p_largest_value, "-0x1.fffffep-22", Status::OK, Category::Normal),
        (m_smallest_value, m_largest_value, "0x1.fffffep-22", Status::OK, Category::Normal),
        (m_smallest_value, p_smallest_value, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_value, m_smallest_value, "0x0p+0", underflow_status, Category::Zero),
        (m_smallest_value, p_smallest_normalized, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_value, m_smallest_normalized, "0x0p+0", underflow_status, Category::Zero),
        (p_smallest_normalized, p_inf, "inf", Status::OK, Category::Infinity),
        (p_smallest_normalized, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_smallest_normalized, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_normalized, m_zero, "-0x0p+0", Status::OK, Category::Zero),
        (p_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_smallest_normalized, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_smallest_normalized, p_normal_value, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, m_normal_value, "-0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, p_largest_value, "0x1.fffffep+1", Status::OK, Category::Normal),
        (p_smallest_normalized, m_largest_value, "-0x1.fffffep+1", Status::OK, Category::Normal),
        (p_smallest_normalized, p_smallest_value, "0x0p+0", underflow_status, Category::Zero),
        (p_smallest_normalized, m_smallest_value, "-0x0p+0", underflow_status, Category::Zero),
        (p_smallest_normalized, p_smallest_normalized, "0x0p+0", underflow_status, Category::Zero),
        (p_smallest_normalized, m_smallest_normalized, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_normalized, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_smallest_normalized, m_inf, "inf", Status::OK, Category::Infinity),
        (m_smallest_normalized, p_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_smallest_normalized, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_smallest_normalized, p_normal_value, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, m_normal_value, "0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, p_largest_value, "-0x1.fffffep+1", Status::OK, Category::Normal),
        (m_smallest_normalized, m_largest_value, "0x1.fffffep+1", Status::OK, Category::Normal),
        (m_smallest_normalized, p_smallest_value, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_normalized, m_smallest_value, "0x0p+0", underflow_status, Category::Zero),
        (m_smallest_normalized, p_smallest_normalized, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_normalized, m_smallest_normalized, "0x0p+0", underflow_status, Category::Zero),
    ];

    for (x, y, e_result, e_status, e_category) in special_cases {
        let status;
        let result = unpack!(status=, x * y);
        assert_eq!(status, e_status);
        assert_eq!(result.category(), e_category);
        assert!(result.bitwise_eq(e_result.parse::<Single>().unwrap()));
    }
}

#[test]
fn divide() {
    // Test Special Cases against each other and normal values.

    // FIXMES/NOTES:
    // 1. Since we perform only default exception handling all operations with
    // signaling NaNs should have a result that is a quiet NaN. Currently they
    // return sNaN.

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let qnan = Single::NAN;
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    let overflow_status = Status::OVERFLOW | Status::INEXACT;
    let underflow_status = Status::UNDERFLOW | Status::INEXACT;

    let special_cases = [
        (p_inf, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, p_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, m_zero, "-inf", Status::OK, Category::Infinity),
        (p_inf, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_inf, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_inf, p_normal_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_normal_value, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_largest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_largest_value, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_value, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, p_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_zero, "inf", Status::OK, Category::Infinity),
        (m_inf, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_inf, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_inf, p_normal_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_normal_value, "inf", Status::OK, Category::Infinity),
        (m_inf, p_largest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_largest_value, "inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_value, "inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (p_zero, p_inf, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_inf, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_zero, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_zero, p_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_normal_value, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_largest_value, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_inf, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_inf, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_zero, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_zero, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_zero, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_zero, p_normal_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_largest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (qnan, p_inf, "nan", Status::OK, Category::NaN),
        (qnan, m_inf, "nan", Status::OK, Category::NaN),
        (qnan, p_zero, "nan", Status::OK, Category::NaN),
        (qnan, m_zero, "nan", Status::OK, Category::NaN),
        (qnan, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (qnan, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (qnan, p_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, m_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, p_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_normalized, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_normalized, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (snan, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (snan, qnan, "nan", Status::INVALID_OP, Category::NaN),
        (snan, snan, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_normal_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_normal_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_largest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_largest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_normal_value, p_inf, "0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, m_inf, "-0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, p_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_normal_value, m_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_normal_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_normal_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_normal_value, p_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, m_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, p_largest_value, "0x1p-128", underflow_status, Category::Normal),
        (p_normal_value, m_largest_value, "-0x1p-128", underflow_status, Category::Normal),
        (p_normal_value, p_smallest_value, "inf", overflow_status, Category::Infinity),
        (p_normal_value, m_smallest_value, "-inf", overflow_status, Category::Infinity),
        (p_normal_value, p_smallest_normalized, "0x1p+126", Status::OK, Category::Normal),
        (p_normal_value, m_smallest_normalized, "-0x1p+126", Status::OK, Category::Normal),
        (m_normal_value, p_inf, "-0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, m_inf, "0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, p_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_normal_value, m_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_normal_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_normal_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_normal_value, p_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, m_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, p_largest_value, "-0x1p-128", underflow_status, Category::Normal),
        (m_normal_value, m_largest_value, "0x1p-128", underflow_status, Category::Normal),
        (m_normal_value, p_smallest_value, "-inf", overflow_status, Category::Infinity),
        (m_normal_value, m_smallest_value, "inf", overflow_status, Category::Infinity),
        (m_normal_value, p_smallest_normalized, "-0x1p+126", Status::OK, Category::Normal),
        (m_normal_value, m_smallest_normalized, "0x1p+126", Status::OK, Category::Normal),
        (p_largest_value, p_inf, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, m_inf, "-0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, p_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_largest_value, m_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_largest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_largest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_largest_value, p_normal_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, m_normal_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, p_largest_value, "0x1p+0", Status::OK, Category::Normal),
        (p_largest_value, m_largest_value, "-0x1p+0", Status::OK, Category::Normal),
        (p_largest_value, p_smallest_value, "inf", overflow_status, Category::Infinity),
        (p_largest_value, m_smallest_value, "-inf", overflow_status, Category::Infinity),
        (p_largest_value, p_smallest_normalized, "inf", overflow_status, Category::Infinity),
        (p_largest_value, m_smallest_normalized, "-inf", overflow_status, Category::Infinity),
        (m_largest_value, p_inf, "-0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, m_inf, "0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, p_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_largest_value, m_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_largest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_largest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_largest_value, p_normal_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, m_normal_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, p_largest_value, "-0x1p+0", Status::OK, Category::Normal),
        (m_largest_value, m_largest_value, "0x1p+0", Status::OK, Category::Normal),
        (m_largest_value, p_smallest_value, "-inf", overflow_status, Category::Infinity),
        (m_largest_value, m_smallest_value, "inf", overflow_status, Category::Infinity),
        (m_largest_value, p_smallest_normalized, "-inf", overflow_status, Category::Infinity),
        (m_largest_value, m_smallest_normalized, "inf", overflow_status, Category::Infinity),
        (p_smallest_value, p_inf, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, m_inf, "-0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, p_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_smallest_value, m_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_smallest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_smallest_value, p_normal_value, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, m_normal_value, "-0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, p_largest_value, "0x0p+0", underflow_status, Category::Zero),
        (p_smallest_value, m_largest_value, "-0x0p+0", underflow_status, Category::Zero),
        (p_smallest_value, p_smallest_value, "0x1p+0", Status::OK, Category::Normal),
        (p_smallest_value, m_smallest_value, "-0x1p+0", Status::OK, Category::Normal),
        (p_smallest_value, p_smallest_normalized, "0x1p-23", Status::OK, Category::Normal),
        (p_smallest_value, m_smallest_normalized, "-0x1p-23", Status::OK, Category::Normal),
        (m_smallest_value, p_inf, "-0x0p+0", Status::OK, Category::Zero),
        (m_smallest_value, m_inf, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_value, p_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_smallest_value, m_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_smallest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_smallest_value, p_normal_value, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, m_normal_value, "0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, p_largest_value, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_value, m_largest_value, "0x0p+0", underflow_status, Category::Zero),
        (m_smallest_value, p_smallest_value, "-0x1p+0", Status::OK, Category::Normal),
        (m_smallest_value, m_smallest_value, "0x1p+0", Status::OK, Category::Normal),
        (m_smallest_value, p_smallest_normalized, "-0x1p-23", Status::OK, Category::Normal),
        (m_smallest_value, m_smallest_normalized, "0x1p-23", Status::OK, Category::Normal),
        (p_smallest_normalized, p_inf, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_normalized, m_inf, "-0x0p+0", Status::OK, Category::Zero),
        (p_smallest_normalized, p_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_smallest_normalized, m_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (p_smallest_normalized, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (p_smallest_normalized, p_normal_value, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, m_normal_value, "-0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, p_largest_value, "0x0p+0", underflow_status, Category::Zero),
        (p_smallest_normalized, m_largest_value, "-0x0p+0", underflow_status, Category::Zero),
        (p_smallest_normalized, p_smallest_value, "0x1p+23", Status::OK, Category::Normal),
        (p_smallest_normalized, m_smallest_value, "-0x1p+23", Status::OK, Category::Normal),
        (p_smallest_normalized, p_smallest_normalized, "0x1p+0", Status::OK, Category::Normal),
        (p_smallest_normalized, m_smallest_normalized, "-0x1p+0", Status::OK, Category::Normal),
        (m_smallest_normalized, p_inf, "-0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, m_inf, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, p_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_smallest_normalized, m_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        /*
        // See Note 1.
        (m_smallest_normalized, snan, "nan", Status::INVALID_OP, Category::NaN),
                */
        (m_smallest_normalized, p_normal_value, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, m_normal_value, "0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, p_largest_value, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_normalized, m_largest_value, "0x0p+0", underflow_status, Category::Zero),
        (m_smallest_normalized, p_smallest_value, "-0x1p+23", Status::OK, Category::Normal),
        (m_smallest_normalized, m_smallest_value, "0x1p+23", Status::OK, Category::Normal),
        (m_smallest_normalized, p_smallest_normalized, "-0x1p+0", Status::OK, Category::Normal),
        (m_smallest_normalized, m_smallest_normalized, "0x1p+0", Status::OK, Category::Normal),
    ];

    for (x, y, e_result, e_status, e_category) in special_cases {
        let status;
        let result = unpack!(status=, x / y);
        assert_eq!(status, e_status);
        assert_eq!(result.category(), e_category);
        assert!(result.bitwise_eq(e_result.parse::<Single>().unwrap()));
    }
}

#[test]
fn operator_overloads() {
    // This is mostly testing that these operator overloads compile.
    let one = "0x1p+0".parse::<Single>().unwrap();
    let two = "0x2p+0".parse::<Single>().unwrap();
    assert!(two.bitwise_eq((one + one).value));
    assert!(one.bitwise_eq((two - one).value));
    assert!(two.bitwise_eq((one * two).value));
    assert!(one.bitwise_eq((two / two).value));
}

#[test]
fn abs() {
    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let p_qnan = Single::NAN;
    let m_qnan = -Single::NAN;
    let p_snan = Single::snan(None);
    let m_snan = -Single::snan(None);
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    assert!(p_inf.bitwise_eq(p_inf.abs()));
    assert!(p_inf.bitwise_eq(m_inf.abs()));
    assert!(p_zero.bitwise_eq(p_zero.abs()));
    assert!(p_zero.bitwise_eq(m_zero.abs()));
    assert!(p_qnan.bitwise_eq(p_qnan.abs()));
    assert!(p_qnan.bitwise_eq(m_qnan.abs()));
    assert!(p_snan.bitwise_eq(p_snan.abs()));
    assert!(p_snan.bitwise_eq(m_snan.abs()));
    assert!(p_normal_value.bitwise_eq(p_normal_value.abs()));
    assert!(p_normal_value.bitwise_eq(m_normal_value.abs()));
    assert!(p_largest_value.bitwise_eq(p_largest_value.abs()));
    assert!(p_largest_value.bitwise_eq(m_largest_value.abs()));
    assert!(p_smallest_value.bitwise_eq(p_smallest_value.abs()));
    assert!(p_smallest_value.bitwise_eq(m_smallest_value.abs()));
    assert!(p_smallest_normalized.bitwise_eq(p_smallest_normalized.abs(),));
    assert!(p_smallest_normalized.bitwise_eq(m_smallest_normalized.abs(),));
}

#[test]
fn neg() {
    let one = "1.0".parse::<Single>().unwrap();
    let neg_one = "-1.0".parse::<Single>().unwrap();
    let zero = Single::ZERO;
    let neg_zero = -Single::ZERO;
    let inf = Single::INFINITY;
    let neg_inf = -Single::INFINITY;
    let qnan = Single::NAN;
    let neg_qnan = -Single::NAN;

    assert!(neg_one.bitwise_eq(-one));
    assert!(one.bitwise_eq(-neg_one));
    assert!(neg_zero.bitwise_eq(-zero));
    assert!(zero.bitwise_eq(-neg_zero));
    assert!(neg_inf.bitwise_eq(-inf));
    assert!(inf.bitwise_eq(-neg_inf));
    assert!(neg_inf.bitwise_eq(-inf));
    assert!(inf.bitwise_eq(-neg_inf));
    assert!(neg_qnan.bitwise_eq(-qnan));
    assert!(qnan.bitwise_eq(-neg_qnan));
}

#[test]
fn ilogb() {
    assert_eq!(-1074, Double::SMALLEST.ilogb());
    assert_eq!(-1074, (-Double::SMALLEST).ilogb());
    assert_eq!(-1023, "0x1.ffffffffffffep-1024".parse::<Double>().unwrap().ilogb());
    assert_eq!(-1023, "0x1.ffffffffffffep-1023".parse::<Double>().unwrap().ilogb());
    assert_eq!(-1023, "-0x1.ffffffffffffep-1023".parse::<Double>().unwrap().ilogb());
    assert_eq!(-51, "0x1p-51".parse::<Double>().unwrap().ilogb());
    assert_eq!(-1023, "0x1.c60f120d9f87cp-1023".parse::<Double>().unwrap().ilogb());
    assert_eq!(-2, "0x0.ffffp-1".parse::<Double>().unwrap().ilogb());
    assert_eq!(-1023, "0x1.fffep-1023".parse::<Double>().unwrap().ilogb());
    assert_eq!(1023, Double::largest().ilogb());
    assert_eq!(1023, (-Double::largest()).ilogb());

    assert_eq!(0, "0x1p+0".parse::<Single>().unwrap().ilogb());
    assert_eq!(0, "-0x1p+0".parse::<Single>().unwrap().ilogb());
    assert_eq!(42, "0x1p+42".parse::<Single>().unwrap().ilogb());
    assert_eq!(-42, "0x1p-42".parse::<Single>().unwrap().ilogb());

    assert_eq!(IEK_INF, Single::INFINITY.ilogb());
    assert_eq!(IEK_INF, (-Single::INFINITY).ilogb());
    assert_eq!(IEK_ZERO, Single::ZERO.ilogb());
    assert_eq!(IEK_ZERO, (-Single::ZERO).ilogb());
    assert_eq!(IEK_NAN, Single::NAN.ilogb());
    assert_eq!(IEK_NAN, Single::snan(None).ilogb());

    assert_eq!(127, Single::largest().ilogb());
    assert_eq!(127, (-Single::largest()).ilogb());

    assert_eq!(-149, Single::SMALLEST.ilogb());
    assert_eq!(-149, (-Single::SMALLEST).ilogb());
    assert_eq!(-126, Single::smallest_normalized().ilogb());
    assert_eq!(-126, (-Single::smallest_normalized()).ilogb());
}

#[test]
fn scalbn() {
    assert!(
        "0x1p+0"
            .parse::<Single>()
            .unwrap()
            .bitwise_eq("0x1p+0".parse::<Single>().unwrap().scalbn(0),)
    );
    assert!(
        "0x1p+42"
            .parse::<Single>()
            .unwrap()
            .bitwise_eq("0x1p+0".parse::<Single>().unwrap().scalbn(42),)
    );
    assert!(
        "0x1p-42"
            .parse::<Single>()
            .unwrap()
            .bitwise_eq("0x1p+0".parse::<Single>().unwrap().scalbn(-42),)
    );

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let p_qnan = Single::NAN;
    let m_qnan = -Single::NAN;
    let snan = Single::snan(None);

    assert!(p_inf.bitwise_eq(p_inf.scalbn(0)));
    assert!(m_inf.bitwise_eq(m_inf.scalbn(0)));
    assert!(p_zero.bitwise_eq(p_zero.scalbn(0)));
    assert!(m_zero.bitwise_eq(m_zero.scalbn(0)));
    assert!(p_qnan.bitwise_eq(p_qnan.scalbn(0)));
    assert!(m_qnan.bitwise_eq(m_qnan.scalbn(0)));
    assert!(!snan.scalbn(0).is_signaling());

    let scalbn_snan = snan.scalbn(1);
    assert!(scalbn_snan.is_nan() && !scalbn_snan.is_signaling());

    // Make sure highest bit of payload is preserved.
    let payload = (1 << 50) | (1 << 49) | (1234 << 32) | 1;

    let snan_with_payload = Double::snan(Some(payload));
    let quiet_payload = snan_with_payload.scalbn(1);
    assert!(quiet_payload.is_nan() && !quiet_payload.is_signaling());
    assert_eq!(payload, quiet_payload.to_bits() & ((1 << 51) - 1));

    assert!(p_inf.bitwise_eq("0x1p+0".parse::<Single>().unwrap().scalbn(128),));
    assert!(m_inf.bitwise_eq("-0x1p+0".parse::<Single>().unwrap().scalbn(128),));
    assert!(p_inf.bitwise_eq("0x1p+127".parse::<Single>().unwrap().scalbn(1),));
    assert!(p_zero.bitwise_eq("0x1p-127".parse::<Single>().unwrap().scalbn(-127),));
    assert!(m_zero.bitwise_eq("-0x1p-127".parse::<Single>().unwrap().scalbn(-127),));
    assert!(
        "-0x1p-149"
            .parse::<Single>()
            .unwrap()
            .bitwise_eq("-0x1p-127".parse::<Single>().unwrap().scalbn(-22),)
    );
    assert!(p_zero.bitwise_eq("0x1p-126".parse::<Single>().unwrap().scalbn(-24),));

    let smallest_f64 = Double::SMALLEST;
    let neg_smallest_f64 = -Double::SMALLEST;

    let largest_f64 = Double::largest();
    let neg_largest_f64 = -Double::largest();

    let largest_denormal_f64 = "0x1.ffffffffffffep-1023".parse::<Double>().unwrap();
    let neg_largest_denormal_f64 = "-0x1.ffffffffffffep-1023".parse::<Double>().unwrap();

    assert!(smallest_f64.bitwise_eq("0x1p-1074".parse::<Double>().unwrap().scalbn(0),));
    assert!(neg_smallest_f64.bitwise_eq("-0x1p-1074".parse::<Double>().unwrap().scalbn(0),));

    assert!("0x1p+1023".parse::<Double>().unwrap().bitwise_eq(smallest_f64.scalbn(2097,),));

    assert!(smallest_f64.scalbn(-2097).is_pos_zero());
    assert!(smallest_f64.scalbn(-2098).is_pos_zero());
    assert!(smallest_f64.scalbn(-2099).is_pos_zero());
    assert!("0x1p+1022".parse::<Double>().unwrap().bitwise_eq(smallest_f64.scalbn(2096,),));
    assert!("0x1p+1023".parse::<Double>().unwrap().bitwise_eq(smallest_f64.scalbn(2097,),));
    assert!(smallest_f64.scalbn(2098).is_infinite());
    assert!(smallest_f64.scalbn(2099).is_infinite());

    // Test for integer overflows when adding to exponent.
    assert!(smallest_f64.scalbn(-ExpInt::MAX).is_pos_zero());
    assert!(largest_f64.scalbn(ExpInt::MAX).is_infinite());

    assert!(largest_denormal_f64.bitwise_eq(largest_denormal_f64.scalbn(0),));
    assert!(neg_largest_denormal_f64.bitwise_eq(neg_largest_denormal_f64.scalbn(0),));

    assert!(
        "0x1.ffffffffffffep-1022"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1))
    );
    assert!(
        "-0x1.ffffffffffffep-1021"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(neg_largest_denormal_f64.scalbn(2))
    );

    assert!(
        "0x1.ffffffffffffep+1"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1024))
    );
    assert!(largest_denormal_f64.scalbn(-1023).is_pos_zero());
    assert!(largest_denormal_f64.scalbn(-1024).is_pos_zero());
    assert!(largest_denormal_f64.scalbn(-2048).is_pos_zero());
    assert!(largest_denormal_f64.scalbn(2047).is_infinite());
    assert!(largest_denormal_f64.scalbn(2098).is_infinite());
    assert!(largest_denormal_f64.scalbn(2099).is_infinite());

    assert!(
        "0x1.ffffffffffffep-2"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1021))
    );
    assert!(
        "0x1.ffffffffffffep-1"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1022))
    );
    assert!(
        "0x1.ffffffffffffep+0"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1023))
    );
    assert!(
        "0x1.ffffffffffffep+1023"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(2046))
    );
    assert!("0x1p+974".parse::<Double>().unwrap().bitwise_eq(smallest_f64.scalbn(2048,),));

    let random_denormal_f64 = "0x1.c60f120d9f87cp+51".parse::<Double>().unwrap();
    assert!(
        "0x1.c60f120d9f87cp-972"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(random_denormal_f64.scalbn(-1023))
    );
    assert!(
        "0x1.c60f120d9f87cp-1"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(random_denormal_f64.scalbn(-52))
    );
    assert!(
        "0x1.c60f120d9f87cp-2"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(random_denormal_f64.scalbn(-53))
    );
    assert!(
        "0x1.c60f120d9f87cp+0"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(random_denormal_f64.scalbn(-51))
    );

    assert!(random_denormal_f64.scalbn(-2097).is_pos_zero());
    assert!(random_denormal_f64.scalbn(-2090).is_pos_zero());

    assert!("-0x1p-1073".parse::<Double>().unwrap().bitwise_eq(neg_largest_f64.scalbn(-2097),));

    assert!("-0x1p-1024".parse::<Double>().unwrap().bitwise_eq(neg_largest_f64.scalbn(-2048),));

    assert!("0x1p-1073".parse::<Double>().unwrap().bitwise_eq(largest_f64.scalbn(-2097,),));

    assert!("0x1p-1074".parse::<Double>().unwrap().bitwise_eq(largest_f64.scalbn(-2098,),));
    assert!("-0x1p-1074".parse::<Double>().unwrap().bitwise_eq(neg_largest_f64.scalbn(-2098),));
    assert!(neg_largest_f64.scalbn(-2099).is_neg_zero());
    assert!(largest_f64.scalbn(1).is_infinite());

    assert!(
        "0x1p+0"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq("0x1p+52".parse::<Double>().unwrap().scalbn(-52),)
    );

    assert!(
        "0x1p-103"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq("0x1p-51".parse::<Double>().unwrap().scalbn(-52),)
    );
}

#[test]
fn frexp() {
    let p_zero = Double::ZERO;
    let m_zero = -Double::ZERO;
    let one = Double::from_f64(1.0);
    let m_one = Double::from_f64(-1.0);

    let largest_denormal = "0x1.ffffffffffffep-1023".parse::<Double>().unwrap();
    let neg_largest_denormal = "-0x1.ffffffffffffep-1023".parse::<Double>().unwrap();

    let smallest = Double::SMALLEST;
    let neg_smallest = -Double::SMALLEST;

    let largest = Double::largest();
    let neg_largest = -Double::largest();

    let p_inf = Double::INFINITY;
    let m_inf = -Double::INFINITY;

    let p_qnan = Double::NAN;
    let m_qnan = -Double::NAN;
    let snan = Double::snan(None);

    // Make sure highest bit of payload is preserved.
    let payload = (1 << 50) | (1 << 49) | (1234 << 32) | 1;

    let snan_with_payload = Double::snan(Some(payload));

    let mut exp = 0;

    let frac = p_zero.frexp(&mut exp);
    assert_eq!(0, exp);
    assert!(frac.is_pos_zero());

    let frac = m_zero.frexp(&mut exp);
    assert_eq!(0, exp);
    assert!(frac.is_neg_zero());

    let frac = one.frexp(&mut exp);
    assert_eq!(1, exp);
    assert!("0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = m_one.frexp(&mut exp);
    assert_eq!(1, exp);
    assert!("-0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = largest_denormal.frexp(&mut exp);
    assert_eq!(-1022, exp);
    assert!("0x1.ffffffffffffep-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = neg_largest_denormal.frexp(&mut exp);
    assert_eq!(-1022, exp);
    assert!("-0x1.ffffffffffffep-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = smallest.frexp(&mut exp);
    assert_eq!(-1073, exp);
    assert!("0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = neg_smallest.frexp(&mut exp);
    assert_eq!(-1073, exp);
    assert!("-0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = largest.frexp(&mut exp);
    assert_eq!(1024, exp);
    assert!("0x1.fffffffffffffp-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = neg_largest.frexp(&mut exp);
    assert_eq!(1024, exp);
    assert!("-0x1.fffffffffffffp-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = p_inf.frexp(&mut exp);
    assert_eq!(IEK_INF, exp);
    assert!(frac.is_infinite() && !frac.is_negative());

    let frac = m_inf.frexp(&mut exp);
    assert_eq!(IEK_INF, exp);
    assert!(frac.is_infinite() && frac.is_negative());

    let frac = p_qnan.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan());

    let frac = m_qnan.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan());

    let frac = snan.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan() && !frac.is_signaling());

    let frac = snan_with_payload.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan() && !frac.is_signaling());
    assert_eq!(payload, frac.to_bits() & ((1 << 51) - 1));

    let frac = "0x0.ffffp-1".parse::<Double>().unwrap().frexp(&mut exp);
    assert_eq!(-1, exp);
    assert!("0x1.fffep-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = "0x1p-51".parse::<Double>().unwrap().frexp(&mut exp);
    assert_eq!(-50, exp);
    assert!("0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = "0x1.c60f120d9f87cp+51".parse::<Double>().unwrap().frexp(&mut exp);
    assert_eq!(52, exp);
    assert!("0x1.c60f120d9f87cp-1".parse::<Double>().unwrap().bitwise_eq(frac));
}

#[test]
fn modulo() {
    let mut status;
    {
        let f1 = "1.5".parse::<Double>().unwrap();
        let f2 = "1.0".parse::<Double>().unwrap();
        let expected = "0.5".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0.5".parse::<Double>().unwrap();
        let f2 = "1.0".parse::<Double>().unwrap();
        let expected = "0.5".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0x1.3333333333333p-2".parse::<Double>().unwrap(); // 0.3
        let f2 = "0x1.47ae147ae147bp-7".parse::<Double>().unwrap(); // 0.01
        // 0.009999999999999983
        let expected = "0x1.47ae147ae1471p-7".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0x1p64".parse::<Double>().unwrap(); // 1.8446744073709552e19
        let f2 = "1.5".parse::<Double>().unwrap();
        let expected = "1.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0x1p1000".parse::<Double>().unwrap();
        let f2 = "0x1p-1000".parse::<Double>().unwrap();
        let expected = "0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0.0".parse::<Double>().unwrap();
        let f2 = "1.0".parse::<Double>().unwrap();
        let expected = "0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "1.0".parse::<Double>().unwrap();
        let f2 = "0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).is_nan());
        assert_eq!(status, Status::INVALID_OP);
    }
    {
        let f1 = "0.0".parse::<Double>().unwrap();
        let f2 = "0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).is_nan());
        assert_eq!(status, Status::INVALID_OP);
    }
    {
        let f1 = Double::INFINITY;
        let f2 = "1.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).is_nan());
        assert_eq!(status, Status::INVALID_OP);
    }
}
