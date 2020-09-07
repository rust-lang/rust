use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::unpack;
use rustc_apfloat::{Category};
use rustc_apfloat::{Float, Status};

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

    for &(x, y, e_result, e_status, e_category) in &special_cases[..] {
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

    for &(x, y, e_result, e_status, e_category) in &special_cases[..] {
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

    for &(x, y, e_result, e_status, e_category) in &special_cases[..] {
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

    for &(x, y, e_result, e_status, e_category) in &special_cases[..] {
        let status;
        let result = unpack!(status=, x / y);
        assert_eq!(status, e_status);
        assert_eq!(result.category(), e_category);
        assert!(result.bitwise_eq(e_result.parse::<Single>().unwrap()));
    }
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
