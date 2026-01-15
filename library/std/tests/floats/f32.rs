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
