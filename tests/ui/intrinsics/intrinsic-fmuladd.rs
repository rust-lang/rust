//@ run-pass
#![feature(core_intrinsics)]
#![feature(cfg_target_has_reliable_f16_f128)]
#![cfg_attr(target_has_reliable_f16, feature(f16))]
#![cfg_attr(target_has_reliable_f128, feature(f128))]
// expect the unexpected (`target_has_reliable_*` are not "known" configs since they are unstable)
#![expect(unexpected_cfgs)]

use std::intrinsics::*;

macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {{
        assert_approx_eq!($a, $b, 1.0e-6);
    }};
    ($a:expr, $b:expr, $tol:expr) => {{
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < $tol, "{:?} is not approximately equal to {:?}", *a, *b);
    }};
}

fn main() {
    {
        let nan: f32 = f32::NAN;
        let inf: f32 = f32::INFINITY;
        let neg_inf: f32 = f32::NEG_INFINITY;
        assert_approx_eq!(fmuladd(1.23f32, 4.5, 0.67), 6.205);
        assert_approx_eq!(fmuladd(-1.23f32, -4.5, -0.67), 4.865);
        assert_approx_eq!(fmuladd(0.0f32, 8.9, 1.2), 1.2);
        assert_approx_eq!(fmuladd(3.4f32, -0.0, 5.6), 5.6);
        assert!(fmuladd(nan, 7.8, 9.0).is_nan());
        assert_eq!(fmuladd(inf, 7.8, 9.0), inf);
        assert_eq!(fmuladd(neg_inf, 7.8, 9.0), neg_inf);
        assert_eq!(fmuladd(8.9, inf, 3.2), inf);
        assert_eq!(fmuladd(-3.2, 2.4, neg_inf), neg_inf);
    }
    {
        let nan: f64 = f64::NAN;
        let inf: f64 = f64::INFINITY;
        let neg_inf: f64 = f64::NEG_INFINITY;
        assert_approx_eq!(fmuladd(1.23f64, 4.5, 0.67), 6.205);
        assert_approx_eq!(fmuladd(-1.23f64, -4.5, -0.67), 4.865);
        assert_approx_eq!(fmuladd(0.0f64, 8.9, 1.2), 1.2);
        assert_approx_eq!(fmuladd(3.4f64, -0.0, 5.6), 5.6);
        assert!(fmuladd(nan, 7.8, 9.0).is_nan());
        assert_eq!(fmuladd(inf, 7.8, 9.0), inf);
        assert_eq!(fmuladd(neg_inf, 7.8, 9.0), neg_inf);
        assert_eq!(fmuladd(8.9, inf, 3.2), inf);
        assert_eq!(fmuladd(-3.2, 2.4, neg_inf), neg_inf);
    }
    #[cfg(target_has_reliable_f16)]
    {
        // `f16` only has ~3 decimal digits of precision, so use a looser tolerance.
        let nan: f16 = f16::NAN;
        let inf: f16 = f16::INFINITY;
        let neg_inf: f16 = f16::NEG_INFINITY;
        assert_approx_eq!(fmuladd(1.23f16, 4.5, 0.67), 6.205, 1.0e-2);
        assert_approx_eq!(fmuladd(-1.23f16, -4.5, -0.67), 4.865, 1.0e-2);
        assert_approx_eq!(fmuladd(0.0f16, 8.9, 1.2), 1.2, 1.0e-2);
        assert_approx_eq!(fmuladd(3.4f16, -0.0, 5.6), 5.6, 1.0e-2);
        assert!(fmuladd(nan, 7.8, 9.0).is_nan());
        assert_eq!(fmuladd(inf, 7.8, 9.0), inf);
        assert_eq!(fmuladd(neg_inf, 7.8, 9.0), neg_inf);
        assert_eq!(fmuladd(8.9, inf, 3.2), inf);
        assert_eq!(fmuladd(-3.2, 2.4, neg_inf), neg_inf);
    }
    #[cfg(target_has_reliable_f128)]
    {
        let nan: f128 = f128::NAN;
        let inf: f128 = f128::INFINITY;
        let neg_inf: f128 = f128::NEG_INFINITY;
        assert_approx_eq!(fmuladd(1.23f128, 4.5, 0.67), 6.205);
        assert_approx_eq!(fmuladd(-1.23f128, -4.5, -0.67), 4.865);
        assert_approx_eq!(fmuladd(0.0f128, 8.9, 1.2), 1.2);
        assert_approx_eq!(fmuladd(3.4f128, -0.0, 5.6), 5.6);
        assert!(fmuladd(nan, 7.8, 9.0).is_nan());
        assert_eq!(fmuladd(inf, 7.8, 9.0), inf);
        assert_eq!(fmuladd(neg_inf, 7.8, 9.0), neg_inf);
        assert_eq!(fmuladd(8.9, inf, 3.2), inf);
        assert_eq!(fmuladd(-3.2, 2.4, neg_inf), neg_inf);
    }
}
