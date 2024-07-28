use num::ToPrimitive;

use super::*;

#[test]
fn test_parse_rational() {
    assert_eq!(Rational::parse("1234").expect_finite(), BigRational::new(1234.into(), 1.into()));
    assert_eq!(
        Rational::parse("-1234").expect_finite(),
        BigRational::new((-1234).into(), 1.into())
    );
    assert_eq!(Rational::parse("1e+6").expect_finite(), BigRational::new(1000000.into(), 1.into()));
    assert_eq!(Rational::parse("1e-6").expect_finite(), BigRational::new(1.into(), 1000000.into()));
    assert_eq!(
        Rational::parse("10.4e6").expect_finite(),
        BigRational::new(10400000.into(), 1.into())
    );
    assert_eq!(
        Rational::parse("10.4e+6").expect_finite(),
        BigRational::new(10400000.into(), 1.into())
    );
    assert_eq!(
        Rational::parse("10.4e-6").expect_finite(),
        BigRational::new(13.into(), 1250000.into())
    );
    assert_eq!(
        Rational::parse("10.4243566462342456234124").expect_finite(),
        BigRational::new(104243566462342456234124_i128.into(), 10000000000000000000000_i128.into())
    );
    assert_eq!(Rational::parse("inf"), Rational::Inf);
    assert_eq!(Rational::parse("+inf"), Rational::Inf);
    assert_eq!(Rational::parse("-inf"), Rational::NegInf);
    assert_eq!(Rational::parse("NaN"), Rational::Nan);
}

#[test]
fn test_decode() {
    assert_eq!(decode(0f32), FloatRes::Zero);
    assert_eq!(decode(f32::INFINITY), FloatRes::Inf);
    assert_eq!(decode(f32::NEG_INFINITY), FloatRes::NegInf);
    assert_eq!(decode(1.0f32).normalize(), FloatRes::Real { sig: 1, exp: 0 });
    assert_eq!(decode(-1.0f32).normalize(), FloatRes::Real { sig: -1, exp: 0 });
    assert_eq!(decode(100.0f32).normalize(), FloatRes::Real { sig: 100, exp: 0 });
    assert_eq!(decode(100.5f32).normalize(), FloatRes::Real { sig: 201, exp: -1 });
    assert_eq!(decode(-4.004f32).normalize(), FloatRes::Real { sig: -8396997, exp: -21 });
    assert_eq!(decode(0.0004f32).normalize(), FloatRes::Real { sig: 13743895, exp: -35 });
    assert_eq!(decode(f32::from_bits(0x1)).normalize(), FloatRes::Real { sig: 1, exp: -149 });
}

#[test]
fn test_validate() {
    validate::<f32>("0").unwrap();
    validate::<f32>("-0").unwrap();
    validate::<f32>("1").unwrap();
    validate::<f32>("-1").unwrap();
    validate::<f32>("1.1").unwrap();
    validate::<f32>("-1.1").unwrap();
    validate::<f32>("1e10").unwrap();
    validate::<f32>("1e1000").unwrap();
    validate::<f32>("-1e1000").unwrap();
    validate::<f32>("1e-1000").unwrap();
    validate::<f32>("-1e-1000").unwrap();
}

#[test]
fn test_validate_real() {
    // Most of the arbitrary values come from checking against <http://weitz.de/ieee/>.
    let r = &BigRational::from_float(10.0).unwrap();
    FloatRes::<f32>::validate_real(r.clone(), 10, 0).unwrap();
    FloatRes::<f32>::validate_real(r.clone(), 10, -1).unwrap_err();
    FloatRes::<f32>::validate_real(r.clone(), 10, 1).unwrap_err();

    let r = &BigRational::from_float(0.25).unwrap();
    FloatRes::<f32>::validate_real(r.clone(), 1, -2).unwrap();
    FloatRes::<f32>::validate_real(r.clone(), 2, -2).unwrap_err();

    let r = &BigRational::from_float(1234.5678).unwrap();
    FloatRes::<f32>::validate_real(r.clone(), 0b100110100101001000101011, -13).unwrap();
    FloatRes::<f32>::validate_real(r.clone(), 0b100110100101001000101010, -13).unwrap_err();
    FloatRes::<f32>::validate_real(r.clone(), 0b100110100101001000101100, -13).unwrap_err();

    let r = &BigRational::from_float(-1234.5678).unwrap();
    FloatRes::<f32>::validate_real(r.clone(), -0b100110100101001000101011, -13).unwrap();
    FloatRes::<f32>::validate_real(r.clone(), -0b100110100101001000101010, -13).unwrap_err();
    FloatRes::<f32>::validate_real(r.clone(), -0b100110100101001000101100, -13).unwrap_err();
}

#[test]
#[allow(unused)]
fn test_validate_real_rounding() {
    // Check that we catch when values don't round to even.

    // For f32, the cutoff between 1.0 and the next value up (1.0000001) is
    // 1.000000059604644775390625. Anything below it should round down, anything above it should
    // round up, and the value itself should round _down_ because `1.0` has an even significand but
    // 1.0000001 is odd.
    let v1_low_down = Rational::parse("1.00000005960464477539062499999").expect_finite();
    let v1_mid_down = Rational::parse("1.000000059604644775390625").expect_finite();
    let v1_high_up = Rational::parse("1.00000005960464477539062500001").expect_finite();

    let exp = -(f32::MAN_BITS as i32);
    let v1_down_sig = 1 << f32::MAN_BITS;
    let v1_up_sig = (1 << f32::MAN_BITS) | 0b1;

    FloatRes::<f32>::validate_real(v1_low_down.clone(), v1_down_sig, exp).unwrap();
    FloatRes::<f32>::validate_real(v1_mid_down.clone(), v1_down_sig, exp).unwrap();
    FloatRes::<f32>::validate_real(v1_high_up.clone(), v1_up_sig, exp).unwrap();
    FloatRes::<f32>::validate_real(-v1_low_down.clone(), -v1_down_sig, exp).unwrap();
    FloatRes::<f32>::validate_real(-v1_mid_down.clone(), -v1_down_sig, exp).unwrap();
    FloatRes::<f32>::validate_real(-v1_high_up.clone(), -v1_up_sig, exp).unwrap();

    // 1.000000178813934326171875 is between 1.0000001 and the next value up, 1.0000002. The middle
    // value here should round _up_ since 1.0000002 has an even mantissa.
    let v2_low_down = Rational::parse("1.00000017881393432617187499999").expect_finite();
    let v2_mid_up = Rational::parse("1.000000178813934326171875").expect_finite();
    let v2_high_up = Rational::parse("1.00000017881393432617187500001").expect_finite();

    let v2_down_sig = v1_up_sig;
    let v2_up_sig = (1 << f32::MAN_BITS) | 0b10;

    FloatRes::<f32>::validate_real(v2_low_down.clone(), v2_down_sig, exp).unwrap();
    FloatRes::<f32>::validate_real(v2_mid_up.clone(), v2_up_sig, exp).unwrap();
    FloatRes::<f32>::validate_real(v2_high_up.clone(), v2_up_sig, exp).unwrap();
    FloatRes::<f32>::validate_real(-v2_low_down.clone(), -v2_down_sig, exp).unwrap();
    FloatRes::<f32>::validate_real(-v2_mid_up.clone(), -v2_up_sig, exp).unwrap();
    FloatRes::<f32>::validate_real(-v2_high_up.clone(), -v2_up_sig, exp).unwrap();

    // Rounding the wrong direction should error
    for res in [
        FloatRes::<f32>::validate_real(v1_mid_down.clone(), v1_up_sig, exp),
        FloatRes::<f32>::validate_real(v2_mid_up.clone(), v2_down_sig, exp),
        FloatRes::<f32>::validate_real(-v1_mid_down.clone(), -v1_up_sig, exp),
        FloatRes::<f32>::validate_real(-v2_mid_up.clone(), -v2_down_sig, exp),
    ] {
        let e = res.unwrap_err();
        let CheckFailure::InvalidReal { incorrect_midpoint_rounding: true, .. } = e else {
            panic!("{e:?}");
        };
    }
}

/// Just a quick check that the constants are what we expect.
#[test]
fn check_constants() {
    assert_eq!(f32::constants().max.to_f32().unwrap(), f32::MAX);
    assert_eq!(f32::constants().min_subnormal.to_f32().unwrap(), f32::from_bits(0x1));
    assert_eq!(f64::constants().max.to_f64().unwrap(), f64::MAX);
    assert_eq!(f64::constants().min_subnormal.to_f64().unwrap(), f64::from_bits(0x1));
}
