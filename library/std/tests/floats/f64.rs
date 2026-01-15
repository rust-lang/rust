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
