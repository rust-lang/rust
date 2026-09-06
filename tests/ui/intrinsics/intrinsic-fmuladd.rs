//@ run-pass
#![feature(core_intrinsics)]

use std::intrinsics::*;

macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {{
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < 1.0e-6, "{} is not approximately equal to {}", *a, *b);
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
}
