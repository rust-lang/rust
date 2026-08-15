// Compiler:
//
// Run-time:
//   status: 0

// FIXME: remove these tests (extracted from libcore) when we run the libcore tests in the CI of the
// Rust repo.

#![feature(core_intrinsics)]

use std::f32::consts;
use std::intrinsics;

const EXP_APPROX: Float = 1e-6;
const ZERO: Float = 0.0;
const ONE: Float = 1.0;

macro_rules! assert_biteq {
    ($left:expr, $right:expr $(,)?) => {{
        let l = $left;
        let r = $right;

        // Hack to coerce left and right to the same type
        let mut _eq_ty = l;
        _eq_ty = r;

        // Hack to get the width from a value
        assert!(l.to_bits() == r.to_bits());
    }};
}

macro_rules! assert_approx_eq {
    ($a:expr, $b:expr $(,)?) => {{ assert_approx_eq!($a, $b, $crate::num::floats::lim_for_ty($a)) }};
    ($a:expr, $b:expr, $lim:expr) => {{
        let (a, b) = (&$a, &$b);
        let diff = (*a - *b).abs();
        assert!(diff <= $lim,);
    }};
}

type Float = f32;
const fn flt(x: Float) -> Float {
    x
}

fn test_exp() {
    assert_biteq!(1.0, flt(0.0).exp());
    assert_approx_eq!(consts::E, flt(1.0).exp(), EXP_APPROX);
    assert_approx_eq!(148.41315910257660342111558004055227962348775, flt(5.0).exp(), EXP_APPROX);

    let inf: Float = Float::INFINITY;
    let neg_inf: Float = Float::NEG_INFINITY;
    let nan: Float = Float::NAN;
    assert_biteq!(inf, inf.exp());
    assert_biteq!(0.0, neg_inf.exp());
    assert!(nan.exp().is_nan());
}

#[inline(never)]
fn my_abs(num: f32) -> f32 {
    unsafe { intrinsics::fabs(num) }
}

fn test_abs() {
    assert_biteq!(Float::INFINITY.abs(), Float::INFINITY);
    assert_biteq!(ONE.abs(), ONE);
    assert_biteq!(ZERO.abs(), ZERO);
    assert_biteq!((-ZERO).abs(), ZERO);
    assert_biteq!((-ONE).abs(), ONE);
    assert_biteq!(Float::NEG_INFINITY.abs(), Float::INFINITY);
    assert_biteq!((ONE / Float::NEG_INFINITY).abs(), ZERO);
    assert!(Float::NAN.abs().is_nan());
}

fn main() {
    test_abs();
    test_exp();
}
