// FIXME(f16_f128): only tested on platforms that have symbols and aren't buggy
#![cfg(target_has_reliable_f128)]

use std::f128::consts;
use std::ops::{Add, Div, Mul, Sub};

// Note these tolerances make sense around zero, but not for more extreme exponents.

/// Default tolerances. Works for values that should be near precise but not exact. Roughly
/// the precision carried by `100 * 100`.
#[cfg(not(miri))]
#[cfg(target_has_reliable_f128_math)]
const TOL: f128 = 1e-12;

/// For operations that are near exact, usually not involving math of different
/// signs.
const TOL_PRECISE: f128 = 1e-28;

/// Tolerances for math that is allowed to be imprecise, usually due to multiple chained
/// operations.
#[cfg(not(miri))]
#[cfg(target_has_reliable_f128_math)]
const TOL_IMPR: f128 = 1e-10;

/// Compare by representation
#[allow(unused_macros)]
macro_rules! assert_f128_biteq {
    ($a:expr, $b:expr) => {
        let (l, r): (&f128, &f128) = (&$a, &$b);
        let lb = l.to_bits();
        let rb = r.to_bits();
        assert_eq!(lb, rb, "float {l:?} is not bitequal to {r:?}.\na: {lb:#034x}\nb: {rb:#034x}");
    };
}

#[test]
fn test_num_f128() {
    // FIXME(f16_f128): replace with a `test_num` call once the required `fmodl`/`fmodf128`
    // function is available on all platforms.
    let ten = 10f128;
    let two = 2f128;
    assert_eq!(ten.add(two), ten + two);
    assert_eq!(ten.sub(two), ten - two);
    assert_eq!(ten.mul(two), ten * two);
    assert_eq!(ten.div(two), ten / two);
}

// Many math functions allow for less accurate results, so the next tolerance up is used

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f128_math)]
fn test_powf() {
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert_eq!(1.0f128.powf(1.0), 1.0);
    assert_approx_eq!(3.4f128.powf(4.5), 246.40818323761892815995637964326426756, TOL_IMPR);
    assert_approx_eq!(2.7f128.powf(-3.2), 0.041652009108526178281070304373500889273, TOL_IMPR);
    assert_approx_eq!((-3.1f128).powf(2.0), 9.6100000000000005506706202140776519387, TOL_IMPR);
    assert_approx_eq!(5.9f128.powf(-2.0), 0.028727377190462507313100483690639638451, TOL_IMPR);
    assert_eq!(8.3f128.powf(0.0), 1.0);
    assert!(nan.powf(2.0).is_nan());
    assert_eq!(inf.powf(2.0), inf);
    assert_eq!(neg_inf.powf(3.0), neg_inf);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f128_math)]
fn test_exp() {
    assert_eq!(1.0, 0.0f128.exp());
    assert_approx_eq!(consts::E, 1.0f128.exp(), TOL);
    assert_approx_eq!(148.41315910257660342111558004055227962348775, 5.0f128.exp(), TOL);

    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    let nan: f128 = f128::NAN;
    assert_eq!(inf, inf.exp());
    assert_eq!(0.0, neg_inf.exp());
    assert!(nan.exp().is_nan());
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f128_math)]
fn test_exp2() {
    assert_eq!(32.0, 5.0f128.exp2());
    assert_eq!(1.0, 0.0f128.exp2());

    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    let nan: f128 = f128::NAN;
    assert_eq!(inf, inf.exp2());
    assert_eq!(0.0, neg_inf.exp2());
    assert!(nan.exp2().is_nan());
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f128_math)]
fn test_ln() {
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert_approx_eq!(1.0f128.exp().ln(), 1.0, TOL);
    assert!(nan.ln().is_nan());
    assert_eq!(inf.ln(), inf);
    assert!(neg_inf.ln().is_nan());
    assert!((-2.3f128).ln().is_nan());
    assert_eq!((-0.0f128).ln(), neg_inf);
    assert_eq!(0.0f128.ln(), neg_inf);
    assert_approx_eq!(4.0f128.ln(), 1.3862943611198906188344642429163531366, TOL);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f128_math)]
fn test_log() {
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert_eq!(10.0f128.log(10.0), 1.0);
    assert_approx_eq!(2.3f128.log(3.5), 0.66485771361478710036766645911922010272, TOL);
    assert_eq!(1.0f128.exp().log(1.0f128.exp()), 1.0);
    assert!(1.0f128.log(1.0).is_nan());
    assert!(1.0f128.log(-13.9).is_nan());
    assert!(nan.log(2.3).is_nan());
    assert_eq!(inf.log(10.0), inf);
    assert!(neg_inf.log(8.8).is_nan());
    assert!((-2.3f128).log(0.1).is_nan());
    assert_eq!((-0.0f128).log(2.0), neg_inf);
    assert_eq!(0.0f128.log(7.0), neg_inf);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f128_math)]
fn test_log2() {
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert_approx_eq!(10.0f128.log2(), 3.32192809488736234787031942948939017, TOL);
    assert_approx_eq!(2.3f128.log2(), 1.2016338611696504130002982471978765921, TOL);
    assert_approx_eq!(1.0f128.exp().log2(), 1.4426950408889634073599246810018921381, TOL);
    assert!(nan.log2().is_nan());
    assert_eq!(inf.log2(), inf);
    assert!(neg_inf.log2().is_nan());
    assert!((-2.3f128).log2().is_nan());
    assert_eq!((-0.0f128).log2(), neg_inf);
    assert_eq!(0.0f128.log2(), neg_inf);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f128_math)]
fn test_log10() {
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert_eq!(10.0f128.log10(), 1.0);
    assert_approx_eq!(2.3f128.log10(), 0.36172783601759284532595218865859309898, TOL);
    assert_approx_eq!(1.0f128.exp().log10(), 0.43429448190325182765112891891660508222, TOL);
    assert_eq!(1.0f128.log10(), 0.0);
    assert!(nan.log10().is_nan());
    assert_eq!(inf.log10(), inf);
    assert!(neg_inf.log10().is_nan());
    assert!((-2.3f128).log10().is_nan());
    assert_eq!((-0.0f128).log10(), neg_inf);
    assert_eq!(0.0f128.log10(), neg_inf);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f128_math)]
fn test_asinh() {
    // Lower accuracy results are allowed, use increased tolerances
    assert_eq!(0.0f128.asinh(), 0.0f128);
    assert_eq!((-0.0f128).asinh(), -0.0f128);

    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    let nan: f128 = f128::NAN;
    assert_eq!(inf.asinh(), inf);
    assert_eq!(neg_inf.asinh(), neg_inf);
    assert!(nan.asinh().is_nan());
    assert!((-0.0f128).asinh().is_sign_negative());

    // issue 63271
    assert_approx_eq!(2.0f128.asinh(), 1.443635475178810342493276740273105f128, TOL_IMPR);
    assert_approx_eq!((-2.0f128).asinh(), -1.443635475178810342493276740273105f128, TOL_IMPR);
    // regression test for the catastrophic cancellation fixed in 72486
    assert_approx_eq!(
        (-67452098.07139316f128).asinh(),
        -18.720075426274544393985484294000831757220,
        TOL_IMPR
    );

    // test for low accuracy from issue 104548
    assert_approx_eq!(60.0f128, 60.0f128.sinh().asinh(), TOL_IMPR);
    // mul needed for approximate comparison to be meaningful
    assert_approx_eq!(1.0f128, 1e-15f128.sinh().asinh() * 1e15f128, TOL_IMPR);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f128_math)]
fn test_acosh() {
    assert_eq!(1.0f128.acosh(), 0.0f128);
    assert!(0.999f128.acosh().is_nan());

    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    let nan: f128 = f128::NAN;
    assert_eq!(inf.acosh(), inf);
    assert!(neg_inf.acosh().is_nan());
    assert!(nan.acosh().is_nan());
    assert_approx_eq!(2.0f128.acosh(), 1.31695789692481670862504634730796844f128, TOL_IMPR);
    assert_approx_eq!(3.0f128.acosh(), 1.76274717403908605046521864995958461f128, TOL_IMPR);

    // test for low accuracy from issue 104548
    assert_approx_eq!(60.0f128, 60.0f128.cosh().acosh(), TOL_IMPR);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f128_math)]
fn test_atanh() {
    assert_eq!(0.0f128.atanh(), 0.0f128);
    assert_eq!((-0.0f128).atanh(), -0.0f128);

    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    let nan: f128 = f128::NAN;
    assert_eq!(1.0f128.atanh(), inf);
    assert_eq!((-1.0f128).atanh(), neg_inf);
    assert!(2f128.atanh().atanh().is_nan());
    assert!((-2f128).atanh().atanh().is_nan());
    assert!(inf.atanh().is_nan());
    assert!(neg_inf.atanh().is_nan());
    assert!(nan.atanh().is_nan());
    assert_approx_eq!(0.5f128.atanh(), 0.54930614433405484569762261846126285f128, TOL_IMPR);
    assert_approx_eq!((-0.5f128).atanh(), -0.54930614433405484569762261846126285f128, TOL_IMPR);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f128_math)]
fn test_gamma() {
    // precision can differ among platforms
    assert_approx_eq!(1.0f128.gamma(), 1.0f128, TOL_IMPR);
    assert_approx_eq!(2.0f128.gamma(), 1.0f128, TOL_IMPR);
    assert_approx_eq!(3.0f128.gamma(), 2.0f128, TOL_IMPR);
    assert_approx_eq!(4.0f128.gamma(), 6.0f128, TOL_IMPR);
    assert_approx_eq!(5.0f128.gamma(), 24.0f128, TOL_IMPR);
    assert_approx_eq!(0.5f128.gamma(), consts::PI.sqrt(), TOL_IMPR);
    assert_approx_eq!((-0.5f128).gamma(), -2.0 * consts::PI.sqrt(), TOL_IMPR);
    assert_eq!(0.0f128.gamma(), f128::INFINITY);
    assert_eq!((-0.0f128).gamma(), f128::NEG_INFINITY);
    assert!((-1.0f128).gamma().is_nan());
    assert!((-2.0f128).gamma().is_nan());
    assert!(f128::NAN.gamma().is_nan());
    assert!(f128::NEG_INFINITY.gamma().is_nan());
    assert_eq!(f128::INFINITY.gamma(), f128::INFINITY);
    assert_eq!(1760.9f128.gamma(), f128::INFINITY);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f128_math)]
fn test_ln_gamma() {
    assert_approx_eq!(1.0f128.ln_gamma().0, 0.0f128, TOL_IMPR);
    assert_eq!(1.0f128.ln_gamma().1, 1);
    assert_approx_eq!(2.0f128.ln_gamma().0, 0.0f128, TOL_IMPR);
    assert_eq!(2.0f128.ln_gamma().1, 1);
    assert_approx_eq!(3.0f128.ln_gamma().0, 2.0f128.ln(), TOL_IMPR);
    assert_eq!(3.0f128.ln_gamma().1, 1);
    assert_approx_eq!((-0.5f128).ln_gamma().0, (2.0 * consts::PI.sqrt()).ln(), TOL_IMPR);
    assert_eq!((-0.5f128).ln_gamma().1, -1);
}

#[test]
fn test_real_consts() {
    let pi: f128 = consts::PI;
    let frac_pi_2: f128 = consts::FRAC_PI_2;
    let frac_pi_3: f128 = consts::FRAC_PI_3;
    let frac_pi_4: f128 = consts::FRAC_PI_4;
    let frac_pi_6: f128 = consts::FRAC_PI_6;
    let frac_pi_8: f128 = consts::FRAC_PI_8;
    let frac_1_pi: f128 = consts::FRAC_1_PI;
    let frac_2_pi: f128 = consts::FRAC_2_PI;

    assert_approx_eq!(frac_pi_2, pi / 2f128, TOL_PRECISE);
    assert_approx_eq!(frac_pi_3, pi / 3f128, TOL_PRECISE);
    assert_approx_eq!(frac_pi_4, pi / 4f128, TOL_PRECISE);
    assert_approx_eq!(frac_pi_6, pi / 6f128, TOL_PRECISE);
    assert_approx_eq!(frac_pi_8, pi / 8f128, TOL_PRECISE);
    assert_approx_eq!(frac_1_pi, 1f128 / pi, TOL_PRECISE);
    assert_approx_eq!(frac_2_pi, 2f128 / pi, TOL_PRECISE);

    #[cfg(not(miri))]
    #[cfg(target_has_reliable_f128_math)]
    {
        let frac_2_sqrtpi: f128 = consts::FRAC_2_SQRT_PI;
        let sqrt2: f128 = consts::SQRT_2;
        let frac_1_sqrt2: f128 = consts::FRAC_1_SQRT_2;
        let e: f128 = consts::E;
        let log2_e: f128 = consts::LOG2_E;
        let log10_e: f128 = consts::LOG10_E;
        let ln_2: f128 = consts::LN_2;
        let ln_10: f128 = consts::LN_10;

        assert_approx_eq!(frac_2_sqrtpi, 2f128 / pi.sqrt(), TOL_PRECISE);
        assert_approx_eq!(sqrt2, 2f128.sqrt(), TOL_PRECISE);
        assert_approx_eq!(frac_1_sqrt2, 1f128 / 2f128.sqrt(), TOL_PRECISE);
        assert_approx_eq!(log2_e, e.log2(), TOL_PRECISE);
        assert_approx_eq!(log10_e, e.log10(), TOL_PRECISE);
        assert_approx_eq!(ln_2, 2f128.ln(), TOL_PRECISE);
        assert_approx_eq!(ln_10, 10f128.ln(), TOL_PRECISE);
    }
}
