// FIXME(f16_f128): only tested on platforms that have symbols and aren't buggy
#![cfg(target_has_reliable_f128)]

use std::f128::consts;
use std::ops::{Add, Div, Mul, Sub};

// Note these tolerances make sense around zero, but not for more extreme exponents.

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
