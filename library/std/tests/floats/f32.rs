use std::f32::consts;

/// Miri adds some extra errors to float functions; make sure the tests still pass.
/// These values are purely used as a canary to test against and are thus not a stable guarantee Rust provides.
/// They serve as a way to get an idea of the real precision of floating point operations on different platforms.
const APPROX_DELTA: f32 = if cfg!(miri) { 1e-3 } else { 1e-6 };

#[allow(unused_macros)]
macro_rules! assert_f32_biteq {
    ($left : expr, $right : expr) => {
        let l: &f32 = &$left;
        let r: &f32 = &$right;
        let lb = l.to_bits();
        let rb = r.to_bits();
        assert_eq!(lb, rb, "float {l} ({lb:#010x}) is not bitequal to {r} ({rb:#010x})");
    };
}

#[test]
fn test_powf() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_eq!(1.0f32.powf(1.0), 1.0);
    assert_approx_eq!(3.4f32.powf(4.5), 246.408218, APPROX_DELTA);
    assert_approx_eq!(2.7f32.powf(-3.2), 0.041652);
    assert_approx_eq!((-3.1f32).powf(2.0), 9.61, APPROX_DELTA);
    assert_approx_eq!(5.9f32.powf(-2.0), 0.028727);
    assert_eq!(8.3f32.powf(0.0), 1.0);
    assert!(nan.powf(2.0).is_nan());
    assert_eq!(inf.powf(2.0), inf);
    assert_eq!(neg_inf.powf(3.0), neg_inf);
}

#[test]
fn test_exp() {
    assert_eq!(1.0, 0.0f32.exp());
    assert_approx_eq!(2.718282, 1.0f32.exp(), APPROX_DELTA);
    assert_approx_eq!(148.413162, 5.0f32.exp(), APPROX_DELTA);

    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    let nan: f32 = f32::NAN;
    assert_eq!(inf, inf.exp());
    assert_eq!(0.0, neg_inf.exp());
    assert!(nan.exp().is_nan());
}

#[test]
fn test_exp2() {
    assert_approx_eq!(32.0, 5.0f32.exp2(), APPROX_DELTA);
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
    assert_approx_eq!(4.0f32.ln(), 1.386294, APPROX_DELTA);
}

#[test]
fn test_log() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_approx_eq!(10.0f32.log(10.0), 1.0);
    assert_approx_eq!(2.3f32.log(3.5), 0.664858);
    assert_approx_eq!(1.0f32.exp().log(1.0f32.exp()), 1.0, APPROX_DELTA);
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
    assert_approx_eq!(10.0f32.log2(), 3.321928, APPROX_DELTA);
    assert_approx_eq!(2.3f32.log2(), 1.201634);
    assert_approx_eq!(1.0f32.exp().log2(), 1.442695, APPROX_DELTA);
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
    assert_approx_eq!(10.0f32.log10(), 1.0);
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

    // test for low accuracy from issue 104548
    assert_approx_eq!(60.0f32, 60.0f32.sinh().asinh());
    // mul needed for approximate comparison to be meaningful
    assert_approx_eq!(1.0f32, 1e-15f32.sinh().asinh() * 1e15f32);
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

    // test for low accuracy from issue 104548
    assert_approx_eq!(60.0f32, 60.0f32.cosh().acosh(), APPROX_DELTA);
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
fn test_gamma() {
    // precision can differ between platforms
    assert_approx_eq!(1.0f32.gamma(), 1.0f32);
    assert_approx_eq!(2.0f32.gamma(), 1.0f32);
    assert_approx_eq!(3.0f32.gamma(), 2.0f32);
    assert_approx_eq!(4.0f32.gamma(), 6.0f32);
    assert_approx_eq!(5.0f32.gamma(), 24.0f32);
    assert_approx_eq!(0.5f32.gamma(), consts::PI.sqrt());
    assert_approx_eq!((-0.5f32).gamma(), -2.0 * consts::PI.sqrt());
    assert_eq!(0.0f32.gamma(), f32::INFINITY);
    assert_eq!((-0.0f32).gamma(), f32::NEG_INFINITY);
    assert!((-1.0f32).gamma().is_nan());
    assert!((-2.0f32).gamma().is_nan());
    assert!(f32::NAN.gamma().is_nan());
    assert!(f32::NEG_INFINITY.gamma().is_nan());
    assert_eq!(f32::INFINITY.gamma(), f32::INFINITY);
    assert_eq!(171.71f32.gamma(), f32::INFINITY);
}

#[test]
fn test_ln_gamma() {
    assert_approx_eq!(1.0f32.ln_gamma().0, 0.0f32);
    assert_eq!(1.0f32.ln_gamma().1, 1);
    assert_approx_eq!(2.0f32.ln_gamma().0, 0.0f32);
    assert_eq!(2.0f32.ln_gamma().1, 1);
    assert_approx_eq!(3.0f32.ln_gamma().0, 2.0f32.ln());
    assert_eq!(3.0f32.ln_gamma().1, 1);
    assert_approx_eq!((-0.5f32).ln_gamma().0, (2.0 * consts::PI.sqrt()).ln());
    assert_eq!((-0.5f32).ln_gamma().1, -1);
}

#[test]
fn test_real_consts() {
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
    assert_approx_eq!(frac_pi_3, pi / 3f32, APPROX_DELTA);
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
    assert_approx_eq!(ln_10, 10f32.ln(), APPROX_DELTA);
}
