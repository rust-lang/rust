use std::f64::consts;

#[allow(unused_macros)]
macro_rules! assert_f64_biteq {
    ($left : expr, $right : expr) => {
        let l: &f64 = &$left;
        let r: &f64 = &$right;
        let lb = l.to_bits();
        let rb = r.to_bits();
        assert_eq!(lb, rb, "float {l} ({lb:#018x}) is not bitequal to {r} ({rb:#018x})");
    };
}

#[test]
fn test_powf() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_eq!(1.0f64.powf(1.0), 1.0);
    assert_approx_eq!(3.4f64.powf(4.5), 246.408183);
    assert_approx_eq!(2.7f64.powf(-3.2), 0.041652);
    assert_approx_eq!((-3.1f64).powf(2.0), 9.61);
    assert_approx_eq!(5.9f64.powf(-2.0), 0.028727);
    assert_eq!(8.3f64.powf(0.0), 1.0);
    assert!(nan.powf(2.0).is_nan());
    assert_eq!(inf.powf(2.0), inf);
    assert_eq!(neg_inf.powf(3.0), neg_inf);
}

#[test]
fn test_exp() {
    assert_eq!(1.0, 0.0f64.exp());
    assert_approx_eq!(2.718282, 1.0f64.exp());
    assert_approx_eq!(148.413159, 5.0f64.exp());

    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    let nan: f64 = f64::NAN;
    assert_eq!(inf, inf.exp());
    assert_eq!(0.0, neg_inf.exp());
    assert!(nan.exp().is_nan());
}

#[test]
fn test_exp2() {
    assert_eq!(32.0, 5.0f64.exp2());
    assert_eq!(1.0, 0.0f64.exp2());

    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    let nan: f64 = f64::NAN;
    assert_eq!(inf, inf.exp2());
    assert_eq!(0.0, neg_inf.exp2());
    assert!(nan.exp2().is_nan());
}

#[test]
fn test_ln() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_approx_eq!(1.0f64.exp().ln(), 1.0);
    assert!(nan.ln().is_nan());
    assert_eq!(inf.ln(), inf);
    assert!(neg_inf.ln().is_nan());
    assert!((-2.3f64).ln().is_nan());
    assert_eq!((-0.0f64).ln(), neg_inf);
    assert_eq!(0.0f64.ln(), neg_inf);
    assert_approx_eq!(4.0f64.ln(), 1.386294);
}

#[test]
fn test_log() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_eq!(10.0f64.log(10.0), 1.0);
    assert_approx_eq!(2.3f64.log(3.5), 0.664858);
    assert_eq!(1.0f64.exp().log(1.0f64.exp()), 1.0);
    assert!(1.0f64.log(1.0).is_nan());
    assert!(1.0f64.log(-13.9).is_nan());
    assert!(nan.log(2.3).is_nan());
    assert_eq!(inf.log(10.0), inf);
    assert!(neg_inf.log(8.8).is_nan());
    assert!((-2.3f64).log(0.1).is_nan());
    assert_eq!((-0.0f64).log(2.0), neg_inf);
    assert_eq!(0.0f64.log(7.0), neg_inf);
}

#[test]
fn test_log2() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_approx_eq!(10.0f64.log2(), 3.321928);
    assert_approx_eq!(2.3f64.log2(), 1.201634);
    assert_approx_eq!(1.0f64.exp().log2(), 1.442695);
    assert!(nan.log2().is_nan());
    assert_eq!(inf.log2(), inf);
    assert!(neg_inf.log2().is_nan());
    assert!((-2.3f64).log2().is_nan());
    assert_eq!((-0.0f64).log2(), neg_inf);
    assert_eq!(0.0f64.log2(), neg_inf);
}

#[test]
fn test_log10() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_eq!(10.0f64.log10(), 1.0);
    assert_approx_eq!(2.3f64.log10(), 0.361728);
    assert_approx_eq!(1.0f64.exp().log10(), 0.434294);
    assert_eq!(1.0f64.log10(), 0.0);
    assert!(nan.log10().is_nan());
    assert_eq!(inf.log10(), inf);
    assert!(neg_inf.log10().is_nan());
    assert!((-2.3f64).log10().is_nan());
    assert_eq!((-0.0f64).log10(), neg_inf);
    assert_eq!(0.0f64.log10(), neg_inf);
}

#[test]
fn test_asinh() {
    assert_eq!(0.0f64.asinh(), 0.0f64);
    assert_eq!((-0.0f64).asinh(), -0.0f64);

    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    let nan: f64 = f64::NAN;
    assert_eq!(inf.asinh(), inf);
    assert_eq!(neg_inf.asinh(), neg_inf);
    assert!(nan.asinh().is_nan());
    assert!((-0.0f64).asinh().is_sign_negative());
    // issue 63271
    assert_approx_eq!(2.0f64.asinh(), 1.443635475178810342493276740273105f64);
    assert_approx_eq!((-2.0f64).asinh(), -1.443635475178810342493276740273105f64);
    // regression test for the catastrophic cancellation fixed in 72486
    assert_approx_eq!((-67452098.07139316f64).asinh(), -18.72007542627454439398548429400083);

    // test for low accuracy from issue 104548
    assert_approx_eq!(60.0f64, 60.0f64.sinh().asinh());
    // mul needed for approximate comparison to be meaningful
    assert_approx_eq!(1.0f64, 1e-15f64.sinh().asinh() * 1e15f64);
}

#[test]
fn test_acosh() {
    assert_eq!(1.0f64.acosh(), 0.0f64);
    assert!(0.999f64.acosh().is_nan());

    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    let nan: f64 = f64::NAN;
    assert_eq!(inf.acosh(), inf);
    assert!(neg_inf.acosh().is_nan());
    assert!(nan.acosh().is_nan());
    assert_approx_eq!(2.0f64.acosh(), 1.31695789692481670862504634730796844f64);
    assert_approx_eq!(3.0f64.acosh(), 1.76274717403908605046521864995958461f64);

    // test for low accuracy from issue 104548
    assert_approx_eq!(60.0f64, 60.0f64.cosh().acosh());
}

#[test]
fn test_atanh() {
    assert_eq!(0.0f64.atanh(), 0.0f64);
    assert_eq!((-0.0f64).atanh(), -0.0f64);

    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    let nan: f64 = f64::NAN;
    assert_eq!(1.0f64.atanh(), inf);
    assert_eq!((-1.0f64).atanh(), neg_inf);
    assert!(2f64.atanh().atanh().is_nan());
    assert!((-2f64).atanh().atanh().is_nan());
    assert!(inf.atanh().is_nan());
    assert!(neg_inf.atanh().is_nan());
    assert!(nan.atanh().is_nan());
    assert_approx_eq!(0.5f64.atanh(), 0.54930614433405484569762261846126285f64);
    assert_approx_eq!((-0.5f64).atanh(), -0.54930614433405484569762261846126285f64);
}

#[test]
fn test_gamma() {
    // precision can differ between platforms
    assert_approx_eq!(1.0f64.gamma(), 1.0f64);
    assert_approx_eq!(2.0f64.gamma(), 1.0f64);
    assert_approx_eq!(3.0f64.gamma(), 2.0f64);
    assert_approx_eq!(4.0f64.gamma(), 6.0f64);
    assert_approx_eq!(5.0f64.gamma(), 24.0f64);
    assert_approx_eq!(0.5f64.gamma(), consts::PI.sqrt());
    assert_approx_eq!((-0.5f64).gamma(), -2.0 * consts::PI.sqrt());
    assert_eq!(0.0f64.gamma(), f64::INFINITY);
    assert_eq!((-0.0f64).gamma(), f64::NEG_INFINITY);
    assert!((-1.0f64).gamma().is_nan());
    assert!((-2.0f64).gamma().is_nan());
    assert!(f64::NAN.gamma().is_nan());
    assert!(f64::NEG_INFINITY.gamma().is_nan());
    assert_eq!(f64::INFINITY.gamma(), f64::INFINITY);
    assert_eq!(171.71f64.gamma(), f64::INFINITY);
}

#[test]
fn test_ln_gamma() {
    assert_approx_eq!(1.0f64.ln_gamma().0, 0.0f64);
    assert_eq!(1.0f64.ln_gamma().1, 1);
    assert_approx_eq!(2.0f64.ln_gamma().0, 0.0f64);
    assert_eq!(2.0f64.ln_gamma().1, 1);
    assert_approx_eq!(3.0f64.ln_gamma().0, 2.0f64.ln());
    assert_eq!(3.0f64.ln_gamma().1, 1);
    assert_approx_eq!((-0.5f64).ln_gamma().0, (2.0 * consts::PI.sqrt()).ln());
    assert_eq!((-0.5f64).ln_gamma().1, -1);
}

#[test]
fn test_real_consts() {
    let pi: f64 = consts::PI;
    let frac_pi_2: f64 = consts::FRAC_PI_2;
    let frac_pi_3: f64 = consts::FRAC_PI_3;
    let frac_pi_4: f64 = consts::FRAC_PI_4;
    let frac_pi_6: f64 = consts::FRAC_PI_6;
    let frac_pi_8: f64 = consts::FRAC_PI_8;
    let frac_1_pi: f64 = consts::FRAC_1_PI;
    let frac_2_pi: f64 = consts::FRAC_2_PI;
    let frac_2_sqrtpi: f64 = consts::FRAC_2_SQRT_PI;
    let sqrt2: f64 = consts::SQRT_2;
    let frac_1_sqrt2: f64 = consts::FRAC_1_SQRT_2;
    let e: f64 = consts::E;
    let log2_e: f64 = consts::LOG2_E;
    let log10_e: f64 = consts::LOG10_E;
    let ln_2: f64 = consts::LN_2;
    let ln_10: f64 = consts::LN_10;

    assert_approx_eq!(frac_pi_2, pi / 2f64);
    assert_approx_eq!(frac_pi_3, pi / 3f64);
    assert_approx_eq!(frac_pi_4, pi / 4f64);
    assert_approx_eq!(frac_pi_6, pi / 6f64);
    assert_approx_eq!(frac_pi_8, pi / 8f64);
    assert_approx_eq!(frac_1_pi, 1f64 / pi);
    assert_approx_eq!(frac_2_pi, 2f64 / pi);
    assert_approx_eq!(frac_2_sqrtpi, 2f64 / pi.sqrt());
    assert_approx_eq!(sqrt2, 2f64.sqrt());
    assert_approx_eq!(frac_1_sqrt2, 1f64 / 2f64.sqrt());
    assert_approx_eq!(log2_e, e.log2());
    assert_approx_eq!(log10_e, e.log10());
    assert_approx_eq!(ln_2, 2f64.ln());
    assert_approx_eq!(ln_10, 10f64.ln());
}
