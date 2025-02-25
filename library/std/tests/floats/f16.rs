// FIXME(f16_f128): only tested on platforms that have symbols and aren't buggy
#![cfg(target_has_reliable_f16)]

use std::f16::consts;

/// Tolerance for results on the order of 10.0e-2
#[allow(unused)]
const TOL_N2: f16 = 0.0001;

/// Tolerance for results on the order of 10.0e+0
#[allow(unused)]
const TOL_0: f16 = 0.01;

/// Tolerance for results on the order of 10.0e+2
#[allow(unused)]
const TOL_P2: f16 = 0.5;

/// Tolerance for results on the order of 10.0e+4
#[allow(unused)]
const TOL_P4: f16 = 10.0;

/// Compare by representation
#[allow(unused_macros)]
macro_rules! assert_f16_biteq {
    ($a:expr, $b:expr) => {
        let (l, r): (&f16, &f16) = (&$a, &$b);
        let lb = l.to_bits();
        let rb = r.to_bits();
        assert_eq!(lb, rb, "float {l:?} ({lb:#04x}) is not bitequal to {r:?} ({rb:#04x})");
    };
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_powf() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert_eq!(1.0f16.powf(1.0), 1.0);
    assert_approx_eq!(3.4f16.powf(4.5), 246.408183, TOL_P2);
    assert_approx_eq!(2.7f16.powf(-3.2), 0.041652, TOL_N2);
    assert_approx_eq!((-3.1f16).powf(2.0), 9.61, TOL_P2);
    assert_approx_eq!(5.9f16.powf(-2.0), 0.028727, TOL_N2);
    assert_eq!(8.3f16.powf(0.0), 1.0);
    assert!(nan.powf(2.0).is_nan());
    assert_eq!(inf.powf(2.0), inf);
    assert_eq!(neg_inf.powf(3.0), neg_inf);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_exp() {
    assert_eq!(1.0, 0.0f16.exp());
    assert_approx_eq!(2.718282, 1.0f16.exp(), TOL_0);
    assert_approx_eq!(148.413159, 5.0f16.exp(), TOL_0);

    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    let nan: f16 = f16::NAN;
    assert_eq!(inf, inf.exp());
    assert_eq!(0.0, neg_inf.exp());
    assert!(nan.exp().is_nan());
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_exp2() {
    assert_eq!(32.0, 5.0f16.exp2());
    assert_eq!(1.0, 0.0f16.exp2());

    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    let nan: f16 = f16::NAN;
    assert_eq!(inf, inf.exp2());
    assert_eq!(0.0, neg_inf.exp2());
    assert!(nan.exp2().is_nan());
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_ln() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert_approx_eq!(1.0f16.exp().ln(), 1.0, TOL_0);
    assert!(nan.ln().is_nan());
    assert_eq!(inf.ln(), inf);
    assert!(neg_inf.ln().is_nan());
    assert!((-2.3f16).ln().is_nan());
    assert_eq!((-0.0f16).ln(), neg_inf);
    assert_eq!(0.0f16.ln(), neg_inf);
    assert_approx_eq!(4.0f16.ln(), 1.386294, TOL_0);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_log() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert_eq!(10.0f16.log(10.0), 1.0);
    assert_approx_eq!(2.3f16.log(3.5), 0.664858, TOL_0);
    assert_eq!(1.0f16.exp().log(1.0f16.exp()), 1.0);
    assert!(1.0f16.log(1.0).is_nan());
    assert!(1.0f16.log(-13.9).is_nan());
    assert!(nan.log(2.3).is_nan());
    assert_eq!(inf.log(10.0), inf);
    assert!(neg_inf.log(8.8).is_nan());
    assert!((-2.3f16).log(0.1).is_nan());
    assert_eq!((-0.0f16).log(2.0), neg_inf);
    assert_eq!(0.0f16.log(7.0), neg_inf);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_log2() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert_approx_eq!(10.0f16.log2(), 3.321928, TOL_0);
    assert_approx_eq!(2.3f16.log2(), 1.201634, TOL_0);
    assert_approx_eq!(1.0f16.exp().log2(), 1.442695, TOL_0);
    assert!(nan.log2().is_nan());
    assert_eq!(inf.log2(), inf);
    assert!(neg_inf.log2().is_nan());
    assert!((-2.3f16).log2().is_nan());
    assert_eq!((-0.0f16).log2(), neg_inf);
    assert_eq!(0.0f16.log2(), neg_inf);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_log10() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert_eq!(10.0f16.log10(), 1.0);
    assert_approx_eq!(2.3f16.log10(), 0.361728, TOL_0);
    assert_approx_eq!(1.0f16.exp().log10(), 0.434294, TOL_0);
    assert_eq!(1.0f16.log10(), 0.0);
    assert!(nan.log10().is_nan());
    assert_eq!(inf.log10(), inf);
    assert!(neg_inf.log10().is_nan());
    assert!((-2.3f16).log10().is_nan());
    assert_eq!((-0.0f16).log10(), neg_inf);
    assert_eq!(0.0f16.log10(), neg_inf);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_asinh() {
    assert_eq!(0.0f16.asinh(), 0.0f16);
    assert_eq!((-0.0f16).asinh(), -0.0f16);

    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    let nan: f16 = f16::NAN;
    assert_eq!(inf.asinh(), inf);
    assert_eq!(neg_inf.asinh(), neg_inf);
    assert!(nan.asinh().is_nan());
    assert!((-0.0f16).asinh().is_sign_negative());
    // issue 63271
    assert_approx_eq!(2.0f16.asinh(), 1.443635475178810342493276740273105f16, TOL_0);
    assert_approx_eq!((-2.0f16).asinh(), -1.443635475178810342493276740273105f16, TOL_0);
    // regression test for the catastrophic cancellation fixed in 72486
    assert_approx_eq!((-200.0f16).asinh(), -5.991470797049389, TOL_0);

    // test for low accuracy from issue 104548
    assert_approx_eq!(10.0f16, 10.0f16.sinh().asinh(), TOL_0);
    // mul needed for approximate comparison to be meaningful
    assert_approx_eq!(1.0f16, 1e-3f16.sinh().asinh() * 1e3f16, TOL_0);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_acosh() {
    assert_eq!(1.0f16.acosh(), 0.0f16);
    assert!(0.999f16.acosh().is_nan());

    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    let nan: f16 = f16::NAN;
    assert_eq!(inf.acosh(), inf);
    assert!(neg_inf.acosh().is_nan());
    assert!(nan.acosh().is_nan());
    assert_approx_eq!(2.0f16.acosh(), 1.31695789692481670862504634730796844f16, TOL_0);
    assert_approx_eq!(3.0f16.acosh(), 1.76274717403908605046521864995958461f16, TOL_0);

    // test for low accuracy from issue 104548
    assert_approx_eq!(10.0f16, 10.0f16.cosh().acosh(), TOL_P2);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_atanh() {
    assert_eq!(0.0f16.atanh(), 0.0f16);
    assert_eq!((-0.0f16).atanh(), -0.0f16);

    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    let nan: f16 = f16::NAN;
    assert_eq!(1.0f16.atanh(), inf);
    assert_eq!((-1.0f16).atanh(), neg_inf);
    assert!(2f16.atanh().atanh().is_nan());
    assert!((-2f16).atanh().atanh().is_nan());
    assert!(inf.atanh().is_nan());
    assert!(neg_inf.atanh().is_nan());
    assert!(nan.atanh().is_nan());
    assert_approx_eq!(0.5f16.atanh(), 0.54930614433405484569762261846126285f16, TOL_0);
    assert_approx_eq!((-0.5f16).atanh(), -0.54930614433405484569762261846126285f16, TOL_0);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_gamma() {
    // precision can differ among platforms
    assert_approx_eq!(1.0f16.gamma(), 1.0f16, TOL_0);
    assert_approx_eq!(2.0f16.gamma(), 1.0f16, TOL_0);
    assert_approx_eq!(3.0f16.gamma(), 2.0f16, TOL_0);
    assert_approx_eq!(4.0f16.gamma(), 6.0f16, TOL_0);
    assert_approx_eq!(5.0f16.gamma(), 24.0f16, TOL_0);
    assert_approx_eq!(0.5f16.gamma(), consts::PI.sqrt(), TOL_0);
    assert_approx_eq!((-0.5f16).gamma(), -2.0 * consts::PI.sqrt(), TOL_0);
    assert_eq!(0.0f16.gamma(), f16::INFINITY);
    assert_eq!((-0.0f16).gamma(), f16::NEG_INFINITY);
    assert!((-1.0f16).gamma().is_nan());
    assert!((-2.0f16).gamma().is_nan());
    assert!(f16::NAN.gamma().is_nan());
    assert!(f16::NEG_INFINITY.gamma().is_nan());
    assert_eq!(f16::INFINITY.gamma(), f16::INFINITY);
    assert_eq!(171.71f16.gamma(), f16::INFINITY);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_ln_gamma() {
    assert_approx_eq!(1.0f16.ln_gamma().0, 0.0f16, TOL_0);
    assert_eq!(1.0f16.ln_gamma().1, 1);
    assert_approx_eq!(2.0f16.ln_gamma().0, 0.0f16, TOL_0);
    assert_eq!(2.0f16.ln_gamma().1, 1);
    assert_approx_eq!(3.0f16.ln_gamma().0, 2.0f16.ln(), TOL_0);
    assert_eq!(3.0f16.ln_gamma().1, 1);
    assert_approx_eq!((-0.5f16).ln_gamma().0, (2.0 * consts::PI.sqrt()).ln(), TOL_0);
    assert_eq!((-0.5f16).ln_gamma().1, -1);
}

#[test]
fn test_real_consts() {
    // FIXME(f16_f128): add math tests when available

    let pi: f16 = consts::PI;
    let frac_pi_2: f16 = consts::FRAC_PI_2;
    let frac_pi_3: f16 = consts::FRAC_PI_3;
    let frac_pi_4: f16 = consts::FRAC_PI_4;
    let frac_pi_6: f16 = consts::FRAC_PI_6;
    let frac_pi_8: f16 = consts::FRAC_PI_8;
    let frac_1_pi: f16 = consts::FRAC_1_PI;
    let frac_2_pi: f16 = consts::FRAC_2_PI;

    assert_approx_eq!(frac_pi_2, pi / 2f16, TOL_0);
    assert_approx_eq!(frac_pi_3, pi / 3f16, TOL_0);
    assert_approx_eq!(frac_pi_4, pi / 4f16, TOL_0);
    assert_approx_eq!(frac_pi_6, pi / 6f16, TOL_0);
    assert_approx_eq!(frac_pi_8, pi / 8f16, TOL_0);
    assert_approx_eq!(frac_1_pi, 1f16 / pi, TOL_0);
    assert_approx_eq!(frac_2_pi, 2f16 / pi, TOL_0);

    #[cfg(not(miri))]
    #[cfg(target_has_reliable_f16_math)]
    {
        let frac_2_sqrtpi: f16 = consts::FRAC_2_SQRT_PI;
        let sqrt2: f16 = consts::SQRT_2;
        let frac_1_sqrt2: f16 = consts::FRAC_1_SQRT_2;
        let e: f16 = consts::E;
        let log2_e: f16 = consts::LOG2_E;
        let log10_e: f16 = consts::LOG10_E;
        let ln_2: f16 = consts::LN_2;
        let ln_10: f16 = consts::LN_10;

        assert_approx_eq!(frac_2_sqrtpi, 2f16 / pi.sqrt(), TOL_0);
        assert_approx_eq!(sqrt2, 2f16.sqrt(), TOL_0);
        assert_approx_eq!(frac_1_sqrt2, 1f16 / 2f16.sqrt(), TOL_0);
        assert_approx_eq!(log2_e, e.log2(), TOL_0);
        assert_approx_eq!(log10_e, e.log10(), TOL_0);
        assert_approx_eq!(ln_2, 2f16.ln(), TOL_0);
        assert_approx_eq!(ln_10, 10f16.ln(), TOL_0);
    }
}
