use crate::f32::consts;
use crate::num::FpCategory as Fp;
use crate::num::*;

#[test]
fn test_num_f32() {
    test_num(10f32, 2f32);
}

#[test]
fn test_min_nan() {
    assert_eq!(f32::NAN.min(2.0), 2.0);
    assert_eq!(2.0f32.min(f32::NAN), 2.0);
}

#[test]
fn test_max_nan() {
    assert_eq!(f32::NAN.max(2.0), 2.0);
    assert_eq!(2.0f32.max(f32::NAN), 2.0);
}

#[test]
fn test_nan() {
    let nan: f32 = f32::NAN;
    assert!(nan.is_nan());
    assert!(!nan.is_infinite());
    assert!(!nan.is_finite());
    assert!(!nan.is_normal());
    assert!(nan.is_sign_positive());
    assert!(!nan.is_sign_negative());
    assert_eq!(Fp::Nan, nan.classify());
}

#[test]
fn test_infinity() {
    let inf: f32 = f32::INFINITY;
    assert!(inf.is_infinite());
    assert!(!inf.is_finite());
    assert!(inf.is_sign_positive());
    assert!(!inf.is_sign_negative());
    assert!(!inf.is_nan());
    assert!(!inf.is_normal());
    assert_eq!(Fp::Infinite, inf.classify());
}

#[test]
fn test_neg_infinity() {
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert!(neg_inf.is_infinite());
    assert!(!neg_inf.is_finite());
    assert!(!neg_inf.is_sign_positive());
    assert!(neg_inf.is_sign_negative());
    assert!(!neg_inf.is_nan());
    assert!(!neg_inf.is_normal());
    assert_eq!(Fp::Infinite, neg_inf.classify());
}

#[test]
fn test_zero() {
    let zero: f32 = 0.0f32;
    assert_eq!(0.0, zero);
    assert!(!zero.is_infinite());
    assert!(zero.is_finite());
    assert!(zero.is_sign_positive());
    assert!(!zero.is_sign_negative());
    assert!(!zero.is_nan());
    assert!(!zero.is_normal());
    assert_eq!(Fp::Zero, zero.classify());
}

#[test]
fn test_neg_zero() {
    let neg_zero: f32 = -0.0;
    assert_eq!(0.0, neg_zero);
    assert!(!neg_zero.is_infinite());
    assert!(neg_zero.is_finite());
    assert!(!neg_zero.is_sign_positive());
    assert!(neg_zero.is_sign_negative());
    assert!(!neg_zero.is_nan());
    assert!(!neg_zero.is_normal());
    assert_eq!(Fp::Zero, neg_zero.classify());
}

#[test]
fn test_one() {
    let one: f32 = 1.0f32;
    assert_eq!(1.0, one);
    assert!(!one.is_infinite());
    assert!(one.is_finite());
    assert!(one.is_sign_positive());
    assert!(!one.is_sign_negative());
    assert!(!one.is_nan());
    assert!(one.is_normal());
    assert_eq!(Fp::Normal, one.classify());
}

#[test]
fn test_is_nan() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert!(nan.is_nan());
    assert!(!0.0f32.is_nan());
    assert!(!5.3f32.is_nan());
    assert!(!(-10.732f32).is_nan());
    assert!(!inf.is_nan());
    assert!(!neg_inf.is_nan());
}

#[test]
fn test_is_infinite() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert!(!nan.is_infinite());
    assert!(inf.is_infinite());
    assert!(neg_inf.is_infinite());
    assert!(!0.0f32.is_infinite());
    assert!(!42.8f32.is_infinite());
    assert!(!(-109.2f32).is_infinite());
}

#[test]
fn test_is_finite() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert!(!nan.is_finite());
    assert!(!inf.is_finite());
    assert!(!neg_inf.is_finite());
    assert!(0.0f32.is_finite());
    assert!(42.8f32.is_finite());
    assert!((-109.2f32).is_finite());
}

#[test]
fn test_is_normal() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    let zero: f32 = 0.0f32;
    let neg_zero: f32 = -0.0;
    assert!(!nan.is_normal());
    assert!(!inf.is_normal());
    assert!(!neg_inf.is_normal());
    assert!(!zero.is_normal());
    assert!(!neg_zero.is_normal());
    assert!(1f32.is_normal());
    assert!(1e-37f32.is_normal());
    assert!(!1e-38f32.is_normal());
}

#[test]
fn test_classify() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    let zero: f32 = 0.0f32;
    let neg_zero: f32 = -0.0;
    assert_eq!(nan.classify(), Fp::Nan);
    assert_eq!(inf.classify(), Fp::Infinite);
    assert_eq!(neg_inf.classify(), Fp::Infinite);
    assert_eq!(zero.classify(), Fp::Zero);
    assert_eq!(neg_zero.classify(), Fp::Zero);
    assert_eq!(1f32.classify(), Fp::Normal);
    assert_eq!(1e-37f32.classify(), Fp::Normal);
    assert_eq!(1e-38f32.classify(), Fp::Subnormal);
}

#[test]
fn test_floor() {
    assert_approx_eq!(1.0f32.floor(), 1.0f32);
    assert_approx_eq!(1.3f32.floor(), 1.0f32);
    assert_approx_eq!(1.5f32.floor(), 1.0f32);
    assert_approx_eq!(1.7f32.floor(), 1.0f32);
    assert_approx_eq!(0.0f32.floor(), 0.0f32);
    assert_approx_eq!((-0.0f32).floor(), -0.0f32);
    assert_approx_eq!((-1.0f32).floor(), -1.0f32);
    assert_approx_eq!((-1.3f32).floor(), -2.0f32);
    assert_approx_eq!((-1.5f32).floor(), -2.0f32);
    assert_approx_eq!((-1.7f32).floor(), -2.0f32);
}

#[test]
fn test_ceil() {
    assert_approx_eq!(1.0f32.ceil(), 1.0f32);
    assert_approx_eq!(1.3f32.ceil(), 2.0f32);
    assert_approx_eq!(1.5f32.ceil(), 2.0f32);
    assert_approx_eq!(1.7f32.ceil(), 2.0f32);
    assert_approx_eq!(0.0f32.ceil(), 0.0f32);
    assert_approx_eq!((-0.0f32).ceil(), -0.0f32);
    assert_approx_eq!((-1.0f32).ceil(), -1.0f32);
    assert_approx_eq!((-1.3f32).ceil(), -1.0f32);
    assert_approx_eq!((-1.5f32).ceil(), -1.0f32);
    assert_approx_eq!((-1.7f32).ceil(), -1.0f32);
}

#[test]
fn test_round() {
    assert_approx_eq!(1.0f32.round(), 1.0f32);
    assert_approx_eq!(1.3f32.round(), 1.0f32);
    assert_approx_eq!(1.5f32.round(), 2.0f32);
    assert_approx_eq!(1.7f32.round(), 2.0f32);
    assert_approx_eq!(0.0f32.round(), 0.0f32);
    assert_approx_eq!((-0.0f32).round(), -0.0f32);
    assert_approx_eq!((-1.0f32).round(), -1.0f32);
    assert_approx_eq!((-1.3f32).round(), -1.0f32);
    assert_approx_eq!((-1.5f32).round(), -2.0f32);
    assert_approx_eq!((-1.7f32).round(), -2.0f32);
}

#[test]
fn test_trunc() {
    assert_approx_eq!(1.0f32.trunc(), 1.0f32);
    assert_approx_eq!(1.3f32.trunc(), 1.0f32);
    assert_approx_eq!(1.5f32.trunc(), 1.0f32);
    assert_approx_eq!(1.7f32.trunc(), 1.0f32);
    assert_approx_eq!(0.0f32.trunc(), 0.0f32);
    assert_approx_eq!((-0.0f32).trunc(), -0.0f32);
    assert_approx_eq!((-1.0f32).trunc(), -1.0f32);
    assert_approx_eq!((-1.3f32).trunc(), -1.0f32);
    assert_approx_eq!((-1.5f32).trunc(), -1.0f32);
    assert_approx_eq!((-1.7f32).trunc(), -1.0f32);
}

#[test]
fn test_fract() {
    assert_approx_eq!(1.0f32.fract(), 0.0f32);
    assert_approx_eq!(1.3f32.fract(), 0.3f32);
    assert_approx_eq!(1.5f32.fract(), 0.5f32);
    assert_approx_eq!(1.7f32.fract(), 0.7f32);
    assert_approx_eq!(0.0f32.fract(), 0.0f32);
    assert_approx_eq!((-0.0f32).fract(), -0.0f32);
    assert_approx_eq!((-1.0f32).fract(), -0.0f32);
    assert_approx_eq!((-1.3f32).fract(), -0.3f32);
    assert_approx_eq!((-1.5f32).fract(), -0.5f32);
    assert_approx_eq!((-1.7f32).fract(), -0.7f32);
}

#[test]
fn test_abs() {
    assert_eq!(f32::INFINITY.abs(), f32::INFINITY);
    assert_eq!(1f32.abs(), 1f32);
    assert_eq!(0f32.abs(), 0f32);
    assert_eq!((-0f32).abs(), 0f32);
    assert_eq!((-1f32).abs(), 1f32);
    assert_eq!(f32::NEG_INFINITY.abs(), f32::INFINITY);
    assert_eq!((1f32 / f32::NEG_INFINITY).abs(), 0f32);
    assert!(f32::NAN.abs().is_nan());
}

#[test]
fn test_signum() {
    assert_eq!(f32::INFINITY.signum(), 1f32);
    assert_eq!(1f32.signum(), 1f32);
    assert_eq!(0f32.signum(), 1f32);
    assert_eq!((-0f32).signum(), -1f32);
    assert_eq!((-1f32).signum(), -1f32);
    assert_eq!(f32::NEG_INFINITY.signum(), -1f32);
    assert_eq!((1f32 / f32::NEG_INFINITY).signum(), -1f32);
    assert!(f32::NAN.signum().is_nan());
}

#[test]
fn test_is_sign_positive() {
    assert!(f32::INFINITY.is_sign_positive());
    assert!(1f32.is_sign_positive());
    assert!(0f32.is_sign_positive());
    assert!(!(-0f32).is_sign_positive());
    assert!(!(-1f32).is_sign_positive());
    assert!(!f32::NEG_INFINITY.is_sign_positive());
    assert!(!(1f32 / f32::NEG_INFINITY).is_sign_positive());
    assert!(f32::NAN.is_sign_positive());
    assert!(!(-f32::NAN).is_sign_positive());
}

#[test]
fn test_is_sign_negative() {
    assert!(!f32::INFINITY.is_sign_negative());
    assert!(!1f32.is_sign_negative());
    assert!(!0f32.is_sign_negative());
    assert!((-0f32).is_sign_negative());
    assert!((-1f32).is_sign_negative());
    assert!(f32::NEG_INFINITY.is_sign_negative());
    assert!((1f32 / f32::NEG_INFINITY).is_sign_negative());
    assert!(!f32::NAN.is_sign_negative());
    assert!((-f32::NAN).is_sign_negative());
}

#[test]
fn test_mul_add() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_approx_eq!(12.3f32.mul_add(4.5, 6.7), 62.05);
    assert_approx_eq!((-12.3f32).mul_add(-4.5, -6.7), 48.65);
    assert_approx_eq!(0.0f32.mul_add(8.9, 1.2), 1.2);
    assert_approx_eq!(3.4f32.mul_add(-0.0, 5.6), 5.6);
    assert!(nan.mul_add(7.8, 9.0).is_nan());
    assert_eq!(inf.mul_add(7.8, 9.0), inf);
    assert_eq!(neg_inf.mul_add(7.8, 9.0), neg_inf);
    assert_eq!(8.9f32.mul_add(inf, 3.2), inf);
    assert_eq!((-3.2f32).mul_add(2.4, neg_inf), neg_inf);
}

#[test]
fn test_recip() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_eq!(1.0f32.recip(), 1.0);
    assert_eq!(2.0f32.recip(), 0.5);
    assert_eq!((-0.4f32).recip(), -2.5);
    assert_eq!(0.0f32.recip(), inf);
    assert!(nan.recip().is_nan());
    assert_eq!(inf.recip(), 0.0);
    assert_eq!(neg_inf.recip(), 0.0);
}

#[test]
fn test_powi() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_eq!(1.0f32.powi(1), 1.0);
    assert_approx_eq!((-3.1f32).powi(2), 9.61);
    assert_approx_eq!(5.9f32.powi(-2), 0.028727);
    assert_eq!(8.3f32.powi(0), 1.0);
    assert!(nan.powi(2).is_nan());
    assert_eq!(inf.powi(3), inf);
    assert_eq!(neg_inf.powi(2), inf);
}

#[test]
fn test_powf() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_eq!(1.0f32.powf(1.0), 1.0);
    assert_approx_eq!(3.4f32.powf(4.5), 246.408218);
    assert_approx_eq!(2.7f32.powf(-3.2), 0.041652);
    assert_approx_eq!((-3.1f32).powf(2.0), 9.61);
    assert_approx_eq!(5.9f32.powf(-2.0), 0.028727);
    assert_eq!(8.3f32.powf(0.0), 1.0);
    assert!(nan.powf(2.0).is_nan());
    assert_eq!(inf.powf(2.0), inf);
    assert_eq!(neg_inf.powf(3.0), neg_inf);
}

#[test]
fn test_sqrt_domain() {
    assert!(f32::NAN.sqrt().is_nan());
    assert!(f32::NEG_INFINITY.sqrt().is_nan());
    assert!((-1.0f32).sqrt().is_nan());
    assert_eq!((-0.0f32).sqrt(), -0.0);
    assert_eq!(0.0f32.sqrt(), 0.0);
    assert_eq!(1.0f32.sqrt(), 1.0);
    assert_eq!(f32::INFINITY.sqrt(), f32::INFINITY);
}

#[test]
fn test_exp() {
    assert_eq!(1.0, 0.0f32.exp());
    assert_approx_eq!(2.718282, 1.0f32.exp());
    assert_approx_eq!(148.413162, 5.0f32.exp());

    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    let nan: f32 = f32::NAN;
    assert_eq!(inf, inf.exp());
    assert_eq!(0.0, neg_inf.exp());
    assert!(nan.exp().is_nan());
}

#[test]
fn test_exp2() {
    assert_eq!(32.0, 5.0f32.exp2());
    assert_eq!(1.0, 0.0f32.exp2());

    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    let nan: f32 = f32::NAN;
    assert_eq!(inf, inf.exp2());
    assert_eq!(0.0, neg_inf.exp2());
    assert!(nan.exp2().is_nan());
}

#[test]
fn test_ln() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_approx_eq!(1.0f32.exp().ln(), 1.0);
    assert!(nan.ln().is_nan());
    assert_eq!(inf.ln(), inf);
    assert!(neg_inf.ln().is_nan());
    assert!((-2.3f32).ln().is_nan());
    assert_eq!((-0.0f32).ln(), neg_inf);
    assert_eq!(0.0f32.ln(), neg_inf);
    assert_approx_eq!(4.0f32.ln(), 1.386294);
}

#[test]
fn test_log() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_eq!(10.0f32.log(10.0), 1.0);
    assert_approx_eq!(2.3f32.log(3.5), 0.664858);
    assert_eq!(1.0f32.exp().log(1.0f32.exp()), 1.0);
    assert!(1.0f32.log(1.0).is_nan());
    assert!(1.0f32.log(-13.9).is_nan());
    assert!(nan.log(2.3).is_nan());
    assert_eq!(inf.log(10.0), inf);
    assert!(neg_inf.log(8.8).is_nan());
    assert!((-2.3f32).log(0.1).is_nan());
    assert_eq!((-0.0f32).log(2.0), neg_inf);
    assert_eq!(0.0f32.log(7.0), neg_inf);
}

#[test]
fn test_log2() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_approx_eq!(10.0f32.log2(), 3.321928);
    assert_approx_eq!(2.3f32.log2(), 1.201634);
    assert_approx_eq!(1.0f32.exp().log2(), 1.442695);
    assert!(nan.log2().is_nan());
    assert_eq!(inf.log2(), inf);
    assert!(neg_inf.log2().is_nan());
    assert!((-2.3f32).log2().is_nan());
    assert_eq!((-0.0f32).log2(), neg_inf);
    assert_eq!(0.0f32.log2(), neg_inf);
}

#[test]
fn test_log10() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_eq!(10.0f32.log10(), 1.0);
    assert_approx_eq!(2.3f32.log10(), 0.361728);
    assert_approx_eq!(1.0f32.exp().log10(), 0.434294);
    assert_eq!(1.0f32.log10(), 0.0);
    assert!(nan.log10().is_nan());
    assert_eq!(inf.log10(), inf);
    assert!(neg_inf.log10().is_nan());
    assert!((-2.3f32).log10().is_nan());
    assert_eq!((-0.0f32).log10(), neg_inf);
    assert_eq!(0.0f32.log10(), neg_inf);
}

#[test]
fn test_to_degrees() {
    let pi: f32 = consts::PI;
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_eq!(0.0f32.to_degrees(), 0.0);
    assert_approx_eq!((-5.8f32).to_degrees(), -332.315521);
    assert_eq!(pi.to_degrees(), 180.0);
    assert!(nan.to_degrees().is_nan());
    assert_eq!(inf.to_degrees(), inf);
    assert_eq!(neg_inf.to_degrees(), neg_inf);
    assert_eq!(1_f32.to_degrees(), 57.2957795130823208767981548141051703);
}

#[test]
fn test_to_radians() {
    let pi: f32 = consts::PI;
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_eq!(0.0f32.to_radians(), 0.0);
    assert_approx_eq!(154.6f32.to_radians(), 2.698279);
    assert_approx_eq!((-332.31f32).to_radians(), -5.799903);
    assert_eq!(180.0f32.to_radians(), pi);
    assert!(nan.to_radians().is_nan());
    assert_eq!(inf.to_radians(), inf);
    assert_eq!(neg_inf.to_radians(), neg_inf);
}

#[test]
fn test_asinh() {
    assert_eq!(0.0f32.asinh(), 0.0f32);
    assert_eq!((-0.0f32).asinh(), -0.0f32);

    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    let nan: f32 = f32::NAN;
    assert_eq!(inf.asinh(), inf);
    assert_eq!(neg_inf.asinh(), neg_inf);
    assert!(nan.asinh().is_nan());
    assert!((-0.0f32).asinh().is_sign_negative()); // issue 63271
    assert_approx_eq!(2.0f32.asinh(), 1.443635475178810342493276740273105f32);
    assert_approx_eq!((-2.0f32).asinh(), -1.443635475178810342493276740273105f32);
    // regression test for the catastrophic cancellation fixed in 72486
    assert_approx_eq!((-3000.0f32).asinh(), -8.699514775987968673236893537700647f32);
}

#[test]
fn test_acosh() {
    assert_eq!(1.0f32.acosh(), 0.0f32);
    assert!(0.999f32.acosh().is_nan());

    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    let nan: f32 = f32::NAN;
    assert_eq!(inf.acosh(), inf);
    assert!(neg_inf.acosh().is_nan());
    assert!(nan.acosh().is_nan());
    assert_approx_eq!(2.0f32.acosh(), 1.31695789692481670862504634730796844f32);
    assert_approx_eq!(3.0f32.acosh(), 1.76274717403908605046521864995958461f32);
}

#[test]
fn test_atanh() {
    assert_eq!(0.0f32.atanh(), 0.0f32);
    assert_eq!((-0.0f32).atanh(), -0.0f32);

    let inf32: f32 = f32::INFINITY;
    let neg_inf32: f32 = f32::NEG_INFINITY;
    assert_eq!(1.0f32.atanh(), inf32);
    assert_eq!((-1.0f32).atanh(), neg_inf32);

    assert!(2f64.atanh().atanh().is_nan());
    assert!((-2f64).atanh().atanh().is_nan());

    let inf64: f32 = f32::INFINITY;
    let neg_inf64: f32 = f32::NEG_INFINITY;
    let nan32: f32 = f32::NAN;
    assert!(inf64.atanh().is_nan());
    assert!(neg_inf64.atanh().is_nan());
    assert!(nan32.atanh().is_nan());

    assert_approx_eq!(0.5f32.atanh(), 0.54930614433405484569762261846126285f32);
    assert_approx_eq!((-0.5f32).atanh(), -0.54930614433405484569762261846126285f32);
}

#[test]
fn test_real_consts() {
    use super::consts;

    let pi: f32 = consts::PI;
    let frac_pi_2: f32 = consts::FRAC_PI_2;
    let frac_pi_3: f32 = consts::FRAC_PI_3;
    let frac_pi_4: f32 = consts::FRAC_PI_4;
    let frac_pi_6: f32 = consts::FRAC_PI_6;
    let frac_pi_8: f32 = consts::FRAC_PI_8;
    let frac_1_pi: f32 = consts::FRAC_1_PI;
    let frac_2_pi: f32 = consts::FRAC_2_PI;
    let frac_2_sqrtpi: f32 = consts::FRAC_2_SQRT_PI;
    let sqrt2: f32 = consts::SQRT_2;
    let frac_1_sqrt2: f32 = consts::FRAC_1_SQRT_2;
    let e: f32 = consts::E;
    let log2_e: f32 = consts::LOG2_E;
    let log10_e: f32 = consts::LOG10_E;
    let ln_2: f32 = consts::LN_2;
    let ln_10: f32 = consts::LN_10;

    assert_approx_eq!(frac_pi_2, pi / 2f32);
    assert_approx_eq!(frac_pi_3, pi / 3f32);
    assert_approx_eq!(frac_pi_4, pi / 4f32);
    assert_approx_eq!(frac_pi_6, pi / 6f32);
    assert_approx_eq!(frac_pi_8, pi / 8f32);
    assert_approx_eq!(frac_1_pi, 1f32 / pi);
    assert_approx_eq!(frac_2_pi, 2f32 / pi);
    assert_approx_eq!(frac_2_sqrtpi, 2f32 / pi.sqrt());
    assert_approx_eq!(sqrt2, 2f32.sqrt());
    assert_approx_eq!(frac_1_sqrt2, 1f32 / 2f32.sqrt());
    assert_approx_eq!(log2_e, e.log2());
    assert_approx_eq!(log10_e, e.log10());
    assert_approx_eq!(ln_2, 2f32.ln());
    assert_approx_eq!(ln_10, 10f32.ln());
}

#[test]
fn test_float_bits_conv() {
    assert_eq!((1f32).to_bits(), 0x3f800000);
    assert_eq!((12.5f32).to_bits(), 0x41480000);
    assert_eq!((1337f32).to_bits(), 0x44a72000);
    assert_eq!((-14.25f32).to_bits(), 0xc1640000);
    assert_approx_eq!(f32::from_bits(0x3f800000), 1.0);
    assert_approx_eq!(f32::from_bits(0x41480000), 12.5);
    assert_approx_eq!(f32::from_bits(0x44a72000), 1337.0);
    assert_approx_eq!(f32::from_bits(0xc1640000), -14.25);

    // Check that NaNs roundtrip their bits regardless of signaling-ness
    // 0xA is 0b1010; 0x5 is 0b0101 -- so these two together clobbers all the mantissa bits
    let masked_nan1 = f32::NAN.to_bits() ^ 0x002A_AAAA;
    let masked_nan2 = f32::NAN.to_bits() ^ 0x0055_5555;
    assert!(f32::from_bits(masked_nan1).is_nan());
    assert!(f32::from_bits(masked_nan2).is_nan());

    assert_eq!(f32::from_bits(masked_nan1).to_bits(), masked_nan1);
    assert_eq!(f32::from_bits(masked_nan2).to_bits(), masked_nan2);
}

#[test]
#[should_panic]
fn test_clamp_min_greater_than_max() {
    let _ = 1.0f32.clamp(3.0, 1.0);
}

#[test]
#[should_panic]
fn test_clamp_min_is_nan() {
    let _ = 1.0f32.clamp(f32::NAN, 1.0);
}

#[test]
#[should_panic]
fn test_clamp_max_is_nan() {
    let _ = 1.0f32.clamp(3.0, f32::NAN);
}

#[test]
fn test_total_cmp() {
    use core::cmp::Ordering;

    fn quiet_bit_mask() -> u32 {
        1 << (f32::MANTISSA_DIGITS - 2)
    }

    fn min_subnorm() -> f32 {
        f32::MIN_POSITIVE / f32::powf(2.0, f32::MANTISSA_DIGITS as f32 - 1.0)
    }

    fn max_subnorm() -> f32 {
        f32::MIN_POSITIVE - min_subnorm()
    }

    fn q_nan() -> f32 {
        f32::from_bits(f32::NAN.to_bits() | quiet_bit_mask())
    }

    fn s_nan() -> f32 {
        f32::from_bits((f32::NAN.to_bits() & !quiet_bit_mask()) + 42)
    }

    assert_eq!(Ordering::Equal, (-q_nan()).total_cmp(&-q_nan()));
    assert_eq!(Ordering::Equal, (-s_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Equal, (-f32::INFINITY).total_cmp(&-f32::INFINITY));
    assert_eq!(Ordering::Equal, (-f32::MAX).total_cmp(&-f32::MAX));
    assert_eq!(Ordering::Equal, (-2.5_f32).total_cmp(&-2.5));
    assert_eq!(Ordering::Equal, (-1.0_f32).total_cmp(&-1.0));
    assert_eq!(Ordering::Equal, (-1.5_f32).total_cmp(&-1.5));
    assert_eq!(Ordering::Equal, (-0.5_f32).total_cmp(&-0.5));
    assert_eq!(Ordering::Equal, (-f32::MIN_POSITIVE).total_cmp(&-f32::MIN_POSITIVE));
    assert_eq!(Ordering::Equal, (-max_subnorm()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Equal, (-min_subnorm()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Equal, (-0.0_f32).total_cmp(&-0.0));
    assert_eq!(Ordering::Equal, 0.0_f32.total_cmp(&0.0));
    assert_eq!(Ordering::Equal, min_subnorm().total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Equal, max_subnorm().total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Equal, f32::MIN_POSITIVE.total_cmp(&f32::MIN_POSITIVE));
    assert_eq!(Ordering::Equal, 0.5_f32.total_cmp(&0.5));
    assert_eq!(Ordering::Equal, 1.0_f32.total_cmp(&1.0));
    assert_eq!(Ordering::Equal, 1.5_f32.total_cmp(&1.5));
    assert_eq!(Ordering::Equal, 2.5_f32.total_cmp(&2.5));
    assert_eq!(Ordering::Equal, f32::MAX.total_cmp(&f32::MAX));
    assert_eq!(Ordering::Equal, f32::INFINITY.total_cmp(&f32::INFINITY));
    assert_eq!(Ordering::Equal, s_nan().total_cmp(&s_nan()));
    assert_eq!(Ordering::Equal, q_nan().total_cmp(&q_nan()));

    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f32::INFINITY));
    assert_eq!(Ordering::Less, (-f32::INFINITY).total_cmp(&-f32::MAX));
    assert_eq!(Ordering::Less, (-f32::MAX).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-2.5_f32).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-1.5_f32).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-1.0_f32).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-0.5_f32).total_cmp(&-f32::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-f32::MIN_POSITIVE).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Less, (-max_subnorm()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-min_subnorm()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-0.0_f32).total_cmp(&0.0));
    assert_eq!(Ordering::Less, 0.0_f32.total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Less, min_subnorm().total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, max_subnorm().total_cmp(&f32::MIN_POSITIVE));
    assert_eq!(Ordering::Less, f32::MIN_POSITIVE.total_cmp(&0.5));
    assert_eq!(Ordering::Less, 0.5_f32.total_cmp(&1.0));
    assert_eq!(Ordering::Less, 1.0_f32.total_cmp(&1.5));
    assert_eq!(Ordering::Less, 1.5_f32.total_cmp(&2.5));
    assert_eq!(Ordering::Less, 2.5_f32.total_cmp(&f32::MAX));
    assert_eq!(Ordering::Less, f32::MAX.total_cmp(&f32::INFINITY));
    assert_eq!(Ordering::Less, f32::INFINITY.total_cmp(&s_nan()));
    assert_eq!(Ordering::Less, s_nan().total_cmp(&q_nan()));

    assert_eq!(Ordering::Greater, (-s_nan()).total_cmp(&-q_nan()));
    assert_eq!(Ordering::Greater, (-f32::INFINITY).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Greater, (-f32::MAX).total_cmp(&-f32::INFINITY));
    assert_eq!(Ordering::Greater, (-2.5_f32).total_cmp(&-f32::MAX));
    assert_eq!(Ordering::Greater, (-1.5_f32).total_cmp(&-2.5));
    assert_eq!(Ordering::Greater, (-1.0_f32).total_cmp(&-1.5));
    assert_eq!(Ordering::Greater, (-0.5_f32).total_cmp(&-1.0));
    assert_eq!(Ordering::Greater, (-f32::MIN_POSITIVE).total_cmp(&-0.5));
    assert_eq!(Ordering::Greater, (-max_subnorm()).total_cmp(&-f32::MIN_POSITIVE));
    assert_eq!(Ordering::Greater, (-min_subnorm()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Greater, (-0.0_f32).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Greater, 0.0_f32.total_cmp(&-0.0));
    assert_eq!(Ordering::Greater, min_subnorm().total_cmp(&0.0));
    assert_eq!(Ordering::Greater, max_subnorm().total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Greater, f32::MIN_POSITIVE.total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Greater, 0.5_f32.total_cmp(&f32::MIN_POSITIVE));
    assert_eq!(Ordering::Greater, 1.0_f32.total_cmp(&0.5));
    assert_eq!(Ordering::Greater, 1.5_f32.total_cmp(&1.0));
    assert_eq!(Ordering::Greater, 2.5_f32.total_cmp(&1.5));
    assert_eq!(Ordering::Greater, f32::MAX.total_cmp(&2.5));
    assert_eq!(Ordering::Greater, f32::INFINITY.total_cmp(&f32::MAX));
    assert_eq!(Ordering::Greater, s_nan().total_cmp(&f32::INFINITY));
    assert_eq!(Ordering::Greater, q_nan().total_cmp(&s_nan()));

    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f32::INFINITY));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f32::MAX));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f32::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&0.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f32::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&0.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&1.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&1.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&2.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f32::MAX));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f32::INFINITY));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&s_nan()));

    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f32::INFINITY));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f32::MAX));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f32::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&0.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f32::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&0.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&1.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&1.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&2.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f32::MAX));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f32::INFINITY));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&s_nan()));
}

#[test]
fn test_lerp_exact() {
    // simple values
    assert_eq!(f32::lerp(0.0, 2.0, 4.0), 2.0);
    assert_eq!(f32::lerp(1.0, 2.0, 4.0), 4.0);

    // boundary values
    assert_eq!(f32::lerp(0.0, f32::MIN, f32::MAX), f32::MIN);
    assert_eq!(f32::lerp(1.0, f32::MIN, f32::MAX), f32::MAX);
}

#[test]
fn test_lerp_consistent() {
    assert_eq!(f32::lerp(f32::MAX, f32::MIN, f32::MIN), f32::MIN);
    assert_eq!(f32::lerp(f32::MIN, f32::MAX, f32::MAX), f32::MAX);

    // as long as t is finite, a/b can be infinite
    assert_eq!(f32::lerp(f32::MAX, f32::NEG_INFINITY, f32::NEG_INFINITY), f32::NEG_INFINITY);
    assert_eq!(f32::lerp(f32::MIN, f32::INFINITY, f32::INFINITY), f32::INFINITY);
}

#[test]
fn test_lerp_nan_infinite() {
    // non-finite t is not NaN if a/b different
    assert!(!f32::lerp(f32::INFINITY, f32::MIN, f32::MAX).is_nan());
    assert!(!f32::lerp(f32::NEG_INFINITY, f32::MIN, f32::MAX).is_nan());
}

#[test]
fn test_lerp_values() {
    // just a few basic values
    assert_eq!(f32::lerp(0.25, 1.0, 2.0), 1.25);
    assert_eq!(f32::lerp(0.50, 1.0, 2.0), 1.50);
    assert_eq!(f32::lerp(0.75, 1.0, 2.0), 1.75);
}

#[test]
fn test_lerp_monotonic() {
    // near 0
    let below_zero = f32::lerp(-f32::EPSILON, f32::MIN, f32::MAX);
    let zero = f32::lerp(0.0, f32::MIN, f32::MAX);
    let above_zero = f32::lerp(f32::EPSILON, f32::MIN, f32::MAX);
    assert!(below_zero <= zero);
    assert!(zero <= above_zero);
    assert!(below_zero <= above_zero);

    // near 0.5
    let below_half = f32::lerp(0.5 - f32::EPSILON, f32::MIN, f32::MAX);
    let half = f32::lerp(0.5, f32::MIN, f32::MAX);
    let above_half = f32::lerp(0.5 + f32::EPSILON, f32::MIN, f32::MAX);
    assert!(below_half <= half);
    assert!(half <= above_half);
    assert!(below_half <= above_half);

    // near 1
    let below_one = f32::lerp(1.0 - f32::EPSILON, f32::MIN, f32::MAX);
    let one = f32::lerp(1.0, f32::MIN, f32::MAX);
    let above_one = f32::lerp(1.0 + f32::EPSILON, f32::MIN, f32::MAX);
    assert!(below_one <= one);
    assert!(one <= above_one);
    assert!(below_one <= above_one);
}
