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

// FIXME(f16_f128,miri): many of these have to be disabled since miri does not yet support
// the intrinsics.

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
