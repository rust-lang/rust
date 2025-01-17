// FIXME(f16_f128): only tested on platforms that have symbols and aren't buggy
#![cfg(reliable_f16)]

use std::f16::consts;
use std::num::FpCategory as Fp;

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

/// Smallest number
const TINY_BITS: u16 = 0x1;

/// Next smallest number
const TINY_UP_BITS: u16 = 0x2;

/// Exponent = 0b11...10, Sifnificand 0b1111..10. Min val > 0
const MAX_DOWN_BITS: u16 = 0x7bfe;

/// Zeroed exponent, full significant
const LARGEST_SUBNORMAL_BITS: u16 = 0x03ff;

/// Exponent = 0b1, zeroed significand
const SMALLEST_NORMAL_BITS: u16 = 0x0400;

/// First pattern over the mantissa
const NAN_MASK1: u16 = 0x02aa;

/// Second pattern over the mantissa
const NAN_MASK2: u16 = 0x0155;

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
fn test_num_f16() {
    crate::test_num(10f16, 2f16);
}

#[test]
#[cfg(reliable_f16_math)]
fn test_min_nan() {
    assert_eq!(f16::NAN.min(2.0), 2.0);
    assert_eq!(2.0f16.min(f16::NAN), 2.0);
}

#[test]
#[cfg(reliable_f16_math)]
fn test_max_nan() {
    assert_eq!(f16::NAN.max(2.0), 2.0);
    assert_eq!(2.0f16.max(f16::NAN), 2.0);
}

#[test]
#[cfg(reliable_f16_math)]
fn test_minimum() {
    assert!(f16::NAN.minimum(2.0).is_nan());
    assert!(2.0f16.minimum(f16::NAN).is_nan());
}

#[test]
#[cfg(reliable_f16_math)]
fn test_maximum() {
    assert!(f16::NAN.maximum(2.0).is_nan());
    assert!(2.0f16.maximum(f16::NAN).is_nan());
}

#[test]
fn test_nan() {
    let nan: f16 = f16::NAN;
    assert!(nan.is_nan());
    assert!(!nan.is_infinite());
    assert!(!nan.is_finite());
    assert!(nan.is_sign_positive());
    assert!(!nan.is_sign_negative());
    assert!(!nan.is_normal());
    assert_eq!(Fp::Nan, nan.classify());
}

#[test]
fn test_infinity() {
    let inf: f16 = f16::INFINITY;
    assert!(inf.is_infinite());
    assert!(!inf.is_finite());
    assert!(inf.is_sign_positive());
    assert!(!inf.is_sign_negative());
    assert!(!inf.is_nan());
    assert!(!inf.is_normal());
    assert_eq!(Fp::Infinite, inf.classify());
}

#[test]
fn test_neg_infinity() {
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert!(neg_inf.is_infinite());
    assert!(!neg_inf.is_finite());
    assert!(!neg_inf.is_sign_positive());
    assert!(neg_inf.is_sign_negative());
    assert!(!neg_inf.is_nan());
    assert!(!neg_inf.is_normal());
    assert_eq!(Fp::Infinite, neg_inf.classify());
}

#[test]
fn test_zero() {
    let zero: f16 = 0.0f16;
    assert_eq!(0.0, zero);
    assert!(!zero.is_infinite());
    assert!(zero.is_finite());
    assert!(zero.is_sign_positive());
    assert!(!zero.is_sign_negative());
    assert!(!zero.is_nan());
    assert!(!zero.is_normal());
    assert_eq!(Fp::Zero, zero.classify());
}

#[test]
fn test_neg_zero() {
    let neg_zero: f16 = -0.0;
    assert_eq!(0.0, neg_zero);
    assert!(!neg_zero.is_infinite());
    assert!(neg_zero.is_finite());
    assert!(!neg_zero.is_sign_positive());
    assert!(neg_zero.is_sign_negative());
    assert!(!neg_zero.is_nan());
    assert!(!neg_zero.is_normal());
    assert_eq!(Fp::Zero, neg_zero.classify());
}

#[test]
fn test_one() {
    let one: f16 = 1.0f16;
    assert_eq!(1.0, one);
    assert!(!one.is_infinite());
    assert!(one.is_finite());
    assert!(one.is_sign_positive());
    assert!(!one.is_sign_negative());
    assert!(!one.is_nan());
    assert!(one.is_normal());
    assert_eq!(Fp::Normal, one.classify());
}

#[test]
fn test_is_nan() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert!(nan.is_nan());
    assert!(!0.0f16.is_nan());
    assert!(!5.3f16.is_nan());
    assert!(!(-10.732f16).is_nan());
    assert!(!inf.is_nan());
    assert!(!neg_inf.is_nan());
}

#[test]
fn test_is_infinite() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert!(!nan.is_infinite());
    assert!(inf.is_infinite());
    assert!(neg_inf.is_infinite());
    assert!(!0.0f16.is_infinite());
    assert!(!42.8f16.is_infinite());
    assert!(!(-109.2f16).is_infinite());
}

#[test]
fn test_is_finite() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert!(!nan.is_finite());
    assert!(!inf.is_finite());
    assert!(!neg_inf.is_finite());
    assert!(0.0f16.is_finite());
    assert!(42.8f16.is_finite());
    assert!((-109.2f16).is_finite());
}

#[test]
fn test_is_normal() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    let zero: f16 = 0.0f16;
    let neg_zero: f16 = -0.0;
    assert!(!nan.is_normal());
    assert!(!inf.is_normal());
    assert!(!neg_inf.is_normal());
    assert!(!zero.is_normal());
    assert!(!neg_zero.is_normal());
    assert!(1f16.is_normal());
    assert!(1e-4f16.is_normal());
    assert!(!1e-5f16.is_normal());
}

#[test]
fn test_classify() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    let zero: f16 = 0.0f16;
    let neg_zero: f16 = -0.0;
    assert_eq!(nan.classify(), Fp::Nan);
    assert_eq!(inf.classify(), Fp::Infinite);
    assert_eq!(neg_inf.classify(), Fp::Infinite);
    assert_eq!(zero.classify(), Fp::Zero);
    assert_eq!(neg_zero.classify(), Fp::Zero);
    assert_eq!(1f16.classify(), Fp::Normal);
    assert_eq!(1e-4f16.classify(), Fp::Normal);
    assert_eq!(1e-5f16.classify(), Fp::Subnormal);
}

#[test]
#[cfg(reliable_f16_math)]
fn test_floor() {
    assert_approx_eq!(1.0f16.floor(), 1.0f16, TOL_0);
    assert_approx_eq!(1.3f16.floor(), 1.0f16, TOL_0);
    assert_approx_eq!(1.5f16.floor(), 1.0f16, TOL_0);
    assert_approx_eq!(1.7f16.floor(), 1.0f16, TOL_0);
    assert_approx_eq!(0.0f16.floor(), 0.0f16, TOL_0);
    assert_approx_eq!((-0.0f16).floor(), -0.0f16, TOL_0);
    assert_approx_eq!((-1.0f16).floor(), -1.0f16, TOL_0);
    assert_approx_eq!((-1.3f16).floor(), -2.0f16, TOL_0);
    assert_approx_eq!((-1.5f16).floor(), -2.0f16, TOL_0);
    assert_approx_eq!((-1.7f16).floor(), -2.0f16, TOL_0);
}

#[test]
#[cfg(reliable_f16_math)]
fn test_ceil() {
    assert_approx_eq!(1.0f16.ceil(), 1.0f16, TOL_0);
    assert_approx_eq!(1.3f16.ceil(), 2.0f16, TOL_0);
    assert_approx_eq!(1.5f16.ceil(), 2.0f16, TOL_0);
    assert_approx_eq!(1.7f16.ceil(), 2.0f16, TOL_0);
    assert_approx_eq!(0.0f16.ceil(), 0.0f16, TOL_0);
    assert_approx_eq!((-0.0f16).ceil(), -0.0f16, TOL_0);
    assert_approx_eq!((-1.0f16).ceil(), -1.0f16, TOL_0);
    assert_approx_eq!((-1.3f16).ceil(), -1.0f16, TOL_0);
    assert_approx_eq!((-1.5f16).ceil(), -1.0f16, TOL_0);
    assert_approx_eq!((-1.7f16).ceil(), -1.0f16, TOL_0);
}

#[test]
#[cfg(reliable_f16_math)]
fn test_round() {
    assert_approx_eq!(2.5f16.round(), 3.0f16, TOL_0);
    assert_approx_eq!(1.0f16.round(), 1.0f16, TOL_0);
    assert_approx_eq!(1.3f16.round(), 1.0f16, TOL_0);
    assert_approx_eq!(1.5f16.round(), 2.0f16, TOL_0);
    assert_approx_eq!(1.7f16.round(), 2.0f16, TOL_0);
    assert_approx_eq!(0.0f16.round(), 0.0f16, TOL_0);
    assert_approx_eq!((-0.0f16).round(), -0.0f16, TOL_0);
    assert_approx_eq!((-1.0f16).round(), -1.0f16, TOL_0);
    assert_approx_eq!((-1.3f16).round(), -1.0f16, TOL_0);
    assert_approx_eq!((-1.5f16).round(), -2.0f16, TOL_0);
    assert_approx_eq!((-1.7f16).round(), -2.0f16, TOL_0);
}

#[test]
#[cfg(reliable_f16_math)]
fn test_round_ties_even() {
    assert_approx_eq!(2.5f16.round_ties_even(), 2.0f16, TOL_0);
    assert_approx_eq!(1.0f16.round_ties_even(), 1.0f16, TOL_0);
    assert_approx_eq!(1.3f16.round_ties_even(), 1.0f16, TOL_0);
    assert_approx_eq!(1.5f16.round_ties_even(), 2.0f16, TOL_0);
    assert_approx_eq!(1.7f16.round_ties_even(), 2.0f16, TOL_0);
    assert_approx_eq!(0.0f16.round_ties_even(), 0.0f16, TOL_0);
    assert_approx_eq!((-0.0f16).round_ties_even(), -0.0f16, TOL_0);
    assert_approx_eq!((-1.0f16).round_ties_even(), -1.0f16, TOL_0);
    assert_approx_eq!((-1.3f16).round_ties_even(), -1.0f16, TOL_0);
    assert_approx_eq!((-1.5f16).round_ties_even(), -2.0f16, TOL_0);
    assert_approx_eq!((-1.7f16).round_ties_even(), -2.0f16, TOL_0);
}

#[test]
#[cfg(reliable_f16_math)]
fn test_trunc() {
    assert_approx_eq!(1.0f16.trunc(), 1.0f16, TOL_0);
    assert_approx_eq!(1.3f16.trunc(), 1.0f16, TOL_0);
    assert_approx_eq!(1.5f16.trunc(), 1.0f16, TOL_0);
    assert_approx_eq!(1.7f16.trunc(), 1.0f16, TOL_0);
    assert_approx_eq!(0.0f16.trunc(), 0.0f16, TOL_0);
    assert_approx_eq!((-0.0f16).trunc(), -0.0f16, TOL_0);
    assert_approx_eq!((-1.0f16).trunc(), -1.0f16, TOL_0);
    assert_approx_eq!((-1.3f16).trunc(), -1.0f16, TOL_0);
    assert_approx_eq!((-1.5f16).trunc(), -1.0f16, TOL_0);
    assert_approx_eq!((-1.7f16).trunc(), -1.0f16, TOL_0);
}

#[test]
#[cfg(reliable_f16_math)]
fn test_fract() {
    assert_approx_eq!(1.0f16.fract(), 0.0f16, TOL_0);
    assert_approx_eq!(1.3f16.fract(), 0.3f16, TOL_0);
    assert_approx_eq!(1.5f16.fract(), 0.5f16, TOL_0);
    assert_approx_eq!(1.7f16.fract(), 0.7f16, TOL_0);
    assert_approx_eq!(0.0f16.fract(), 0.0f16, TOL_0);
    assert_approx_eq!((-0.0f16).fract(), -0.0f16, TOL_0);
    assert_approx_eq!((-1.0f16).fract(), -0.0f16, TOL_0);
    assert_approx_eq!((-1.3f16).fract(), -0.3f16, TOL_0);
    assert_approx_eq!((-1.5f16).fract(), -0.5f16, TOL_0);
    assert_approx_eq!((-1.7f16).fract(), -0.7f16, TOL_0);
}

#[test]
#[cfg(reliable_f16_math)]
fn test_abs() {
    assert_eq!(f16::INFINITY.abs(), f16::INFINITY);
    assert_eq!(1f16.abs(), 1f16);
    assert_eq!(0f16.abs(), 0f16);
    assert_eq!((-0f16).abs(), 0f16);
    assert_eq!((-1f16).abs(), 1f16);
    assert_eq!(f16::NEG_INFINITY.abs(), f16::INFINITY);
    assert_eq!((1f16 / f16::NEG_INFINITY).abs(), 0f16);
    assert!(f16::NAN.abs().is_nan());
}

#[test]
fn test_is_sign_positive() {
    assert!(f16::INFINITY.is_sign_positive());
    assert!(1f16.is_sign_positive());
    assert!(0f16.is_sign_positive());
    assert!(!(-0f16).is_sign_positive());
    assert!(!(-1f16).is_sign_positive());
    assert!(!f16::NEG_INFINITY.is_sign_positive());
    assert!(!(1f16 / f16::NEG_INFINITY).is_sign_positive());
    assert!(f16::NAN.is_sign_positive());
    assert!(!(-f16::NAN).is_sign_positive());
}

#[test]
fn test_is_sign_negative() {
    assert!(!f16::INFINITY.is_sign_negative());
    assert!(!1f16.is_sign_negative());
    assert!(!0f16.is_sign_negative());
    assert!((-0f16).is_sign_negative());
    assert!((-1f16).is_sign_negative());
    assert!(f16::NEG_INFINITY.is_sign_negative());
    assert!((1f16 / f16::NEG_INFINITY).is_sign_negative());
    assert!(!f16::NAN.is_sign_negative());
    assert!((-f16::NAN).is_sign_negative());
}

#[test]
fn test_next_up() {
    let tiny = f16::from_bits(TINY_BITS);
    let tiny_up = f16::from_bits(TINY_UP_BITS);
    let max_down = f16::from_bits(MAX_DOWN_BITS);
    let largest_subnormal = f16::from_bits(LARGEST_SUBNORMAL_BITS);
    let smallest_normal = f16::from_bits(SMALLEST_NORMAL_BITS);
    assert_f16_biteq!(f16::NEG_INFINITY.next_up(), f16::MIN);
    assert_f16_biteq!(f16::MIN.next_up(), -max_down);
    assert_f16_biteq!((-1.0 - f16::EPSILON).next_up(), -1.0);
    assert_f16_biteq!((-smallest_normal).next_up(), -largest_subnormal);
    assert_f16_biteq!((-tiny_up).next_up(), -tiny);
    assert_f16_biteq!((-tiny).next_up(), -0.0f16);
    assert_f16_biteq!((-0.0f16).next_up(), tiny);
    assert_f16_biteq!(0.0f16.next_up(), tiny);
    assert_f16_biteq!(tiny.next_up(), tiny_up);
    assert_f16_biteq!(largest_subnormal.next_up(), smallest_normal);
    assert_f16_biteq!(1.0f16.next_up(), 1.0 + f16::EPSILON);
    assert_f16_biteq!(f16::MAX.next_up(), f16::INFINITY);
    assert_f16_biteq!(f16::INFINITY.next_up(), f16::INFINITY);

    // Check that NaNs roundtrip.
    let nan0 = f16::NAN;
    let nan1 = f16::from_bits(f16::NAN.to_bits() ^ NAN_MASK1);
    let nan2 = f16::from_bits(f16::NAN.to_bits() ^ NAN_MASK2);
    assert_f16_biteq!(nan0.next_up(), nan0);
    assert_f16_biteq!(nan1.next_up(), nan1);
    assert_f16_biteq!(nan2.next_up(), nan2);
}

#[test]
fn test_next_down() {
    let tiny = f16::from_bits(TINY_BITS);
    let tiny_up = f16::from_bits(TINY_UP_BITS);
    let max_down = f16::from_bits(MAX_DOWN_BITS);
    let largest_subnormal = f16::from_bits(LARGEST_SUBNORMAL_BITS);
    let smallest_normal = f16::from_bits(SMALLEST_NORMAL_BITS);
    assert_f16_biteq!(f16::NEG_INFINITY.next_down(), f16::NEG_INFINITY);
    assert_f16_biteq!(f16::MIN.next_down(), f16::NEG_INFINITY);
    assert_f16_biteq!((-max_down).next_down(), f16::MIN);
    assert_f16_biteq!((-1.0f16).next_down(), -1.0 - f16::EPSILON);
    assert_f16_biteq!((-largest_subnormal).next_down(), -smallest_normal);
    assert_f16_biteq!((-tiny).next_down(), -tiny_up);
    assert_f16_biteq!((-0.0f16).next_down(), -tiny);
    assert_f16_biteq!((0.0f16).next_down(), -tiny);
    assert_f16_biteq!(tiny.next_down(), 0.0f16);
    assert_f16_biteq!(tiny_up.next_down(), tiny);
    assert_f16_biteq!(smallest_normal.next_down(), largest_subnormal);
    assert_f16_biteq!((1.0 + f16::EPSILON).next_down(), 1.0f16);
    assert_f16_biteq!(f16::MAX.next_down(), max_down);
    assert_f16_biteq!(f16::INFINITY.next_down(), f16::MAX);

    // Check that NaNs roundtrip.
    let nan0 = f16::NAN;
    let nan1 = f16::from_bits(f16::NAN.to_bits() ^ NAN_MASK1);
    let nan2 = f16::from_bits(f16::NAN.to_bits() ^ NAN_MASK2);
    assert_f16_biteq!(nan0.next_down(), nan0);
    assert_f16_biteq!(nan1.next_down(), nan1);
    assert_f16_biteq!(nan2.next_down(), nan2);
}

#[test]
#[cfg(reliable_f16_math)]
fn test_mul_add() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert_approx_eq!(12.3f16.mul_add(4.5, 6.7), 62.05, TOL_P2);
    assert_approx_eq!((-12.3f16).mul_add(-4.5, -6.7), 48.65, TOL_P2);
    assert_approx_eq!(0.0f16.mul_add(8.9, 1.2), 1.2, TOL_0);
    assert_approx_eq!(3.4f16.mul_add(-0.0, 5.6), 5.6, TOL_0);
    assert!(nan.mul_add(7.8, 9.0).is_nan());
    assert_eq!(inf.mul_add(7.8, 9.0), inf);
    assert_eq!(neg_inf.mul_add(7.8, 9.0), neg_inf);
    assert_eq!(8.9f16.mul_add(inf, 3.2), inf);
    assert_eq!((-3.2f16).mul_add(2.4, neg_inf), neg_inf);
}

#[test]
#[cfg(reliable_f16_math)]
fn test_recip() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert_eq!(1.0f16.recip(), 1.0);
    assert_eq!(2.0f16.recip(), 0.5);
    assert_eq!((-0.4f16).recip(), -2.5);
    assert_eq!(0.0f16.recip(), inf);
    assert_approx_eq!(f16::MAX.recip(), 1.526624e-5f16, 1e-4);
    assert!(nan.recip().is_nan());
    assert_eq!(inf.recip(), 0.0);
    assert_eq!(neg_inf.recip(), 0.0);
}

#[test]
#[cfg(reliable_f16_math)]
fn test_powi() {
    // FIXME(llvm19): LLVM misoptimizes `powi.f16`
    // <https://github.com/llvm/llvm-project/issues/98665>
    // let nan: f16 = f16::NAN;
    // let inf: f16 = f16::INFINITY;
    // let neg_inf: f16 = f16::NEG_INFINITY;
    // assert_eq!(1.0f16.powi(1), 1.0);
    // assert_approx_eq!((-3.1f16).powi(2), 9.61, TOL_0);
    // assert_approx_eq!(5.9f16.powi(-2), 0.028727, TOL_N2);
    // assert_eq!(8.3f16.powi(0), 1.0);
    // assert!(nan.powi(2).is_nan());
    // assert_eq!(inf.powi(3), inf);
    // assert_eq!(neg_inf.powi(2), inf);
}

#[test]
#[cfg(reliable_f16_math)]
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
#[cfg(reliable_f16_math)]
fn test_sqrt_domain() {
    assert!(f16::NAN.sqrt().is_nan());
    assert!(f16::NEG_INFINITY.sqrt().is_nan());
    assert!((-1.0f16).sqrt().is_nan());
    assert_eq!((-0.0f16).sqrt(), -0.0);
    assert_eq!(0.0f16.sqrt(), 0.0);
    assert_eq!(1.0f16.sqrt(), 1.0);
    assert_eq!(f16::INFINITY.sqrt(), f16::INFINITY);
}

#[test]
#[cfg(reliable_f16_math)]
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
#[cfg(reliable_f16_math)]
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
#[cfg(reliable_f16_math)]
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
#[cfg(reliable_f16_math)]
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
#[cfg(reliable_f16_math)]
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
#[cfg(reliable_f16_math)]
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
fn test_to_degrees() {
    let pi: f16 = consts::PI;
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert_eq!(0.0f16.to_degrees(), 0.0);
    assert_approx_eq!((-5.8f16).to_degrees(), -332.315521, TOL_P2);
    assert_approx_eq!(pi.to_degrees(), 180.0, TOL_P2);
    assert!(nan.to_degrees().is_nan());
    assert_eq!(inf.to_degrees(), inf);
    assert_eq!(neg_inf.to_degrees(), neg_inf);
    assert_eq!(1_f16.to_degrees(), 57.2957795130823208767981548141051703);
}

#[test]
fn test_to_radians() {
    let pi: f16 = consts::PI;
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert_eq!(0.0f16.to_radians(), 0.0);
    assert_approx_eq!(154.6f16.to_radians(), 2.698279, TOL_0);
    assert_approx_eq!((-332.31f16).to_radians(), -5.799903, TOL_0);
    assert_approx_eq!(180.0f16.to_radians(), pi, TOL_0);
    assert!(nan.to_radians().is_nan());
    assert_eq!(inf.to_radians(), inf);
    assert_eq!(neg_inf.to_radians(), neg_inf);
}

#[test]
#[cfg(reliable_f16_math)]
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
#[cfg(reliable_f16_math)]
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
#[cfg(reliable_f16_math)]
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
#[cfg(reliable_f16_math)]
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
#[cfg(reliable_f16_math)]
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

    #[cfg(reliable_f16_math)]
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

#[test]
fn test_float_bits_conv() {
    assert_eq!((1f16).to_bits(), 0x3c00);
    assert_eq!((12.5f16).to_bits(), 0x4a40);
    assert_eq!((1337f16).to_bits(), 0x6539);
    assert_eq!((-14.25f16).to_bits(), 0xcb20);
    assert_approx_eq!(f16::from_bits(0x3c00), 1.0, TOL_0);
    assert_approx_eq!(f16::from_bits(0x4a40), 12.5, TOL_0);
    assert_approx_eq!(f16::from_bits(0x6539), 1337.0, TOL_P4);
    assert_approx_eq!(f16::from_bits(0xcb20), -14.25, TOL_0);

    // Check that NaNs roundtrip their bits regardless of signaling-ness
    let masked_nan1 = f16::NAN.to_bits() ^ NAN_MASK1;
    let masked_nan2 = f16::NAN.to_bits() ^ NAN_MASK2;
    assert!(f16::from_bits(masked_nan1).is_nan());
    assert!(f16::from_bits(masked_nan2).is_nan());

    assert_eq!(f16::from_bits(masked_nan1).to_bits(), masked_nan1);
    assert_eq!(f16::from_bits(masked_nan2).to_bits(), masked_nan2);
}

#[test]
#[should_panic]
fn test_clamp_min_greater_than_max() {
    let _ = 1.0f16.clamp(3.0, 1.0);
}

#[test]
#[should_panic]
fn test_clamp_min_is_nan() {
    let _ = 1.0f16.clamp(f16::NAN, 1.0);
}

#[test]
#[should_panic]
fn test_clamp_max_is_nan() {
    let _ = 1.0f16.clamp(3.0, f16::NAN);
}

#[test]
fn test_total_cmp() {
    use core::cmp::Ordering;

    fn quiet_bit_mask() -> u16 {
        1 << (f16::MANTISSA_DIGITS - 2)
    }

    // FIXME(f16_f128): test subnormals when powf is available
    // fn min_subnorm() -> f16 {
    //     f16::MIN_POSITIVE / f16::powf(2.0, f16::MANTISSA_DIGITS as f16 - 1.0)
    // }

    // fn max_subnorm() -> f16 {
    //     f16::MIN_POSITIVE - min_subnorm()
    // }

    fn q_nan() -> f16 {
        f16::from_bits(f16::NAN.to_bits() | quiet_bit_mask())
    }

    fn s_nan() -> f16 {
        f16::from_bits((f16::NAN.to_bits() & !quiet_bit_mask()) + 42)
    }

    assert_eq!(Ordering::Equal, (-q_nan()).total_cmp(&-q_nan()));
    assert_eq!(Ordering::Equal, (-s_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Equal, (-f16::INFINITY).total_cmp(&-f16::INFINITY));
    assert_eq!(Ordering::Equal, (-f16::MAX).total_cmp(&-f16::MAX));
    assert_eq!(Ordering::Equal, (-2.5_f16).total_cmp(&-2.5));
    assert_eq!(Ordering::Equal, (-1.0_f16).total_cmp(&-1.0));
    assert_eq!(Ordering::Equal, (-1.5_f16).total_cmp(&-1.5));
    assert_eq!(Ordering::Equal, (-0.5_f16).total_cmp(&-0.5));
    assert_eq!(Ordering::Equal, (-f16::MIN_POSITIVE).total_cmp(&-f16::MIN_POSITIVE));
    // assert_eq!(Ordering::Equal, (-max_subnorm()).total_cmp(&-max_subnorm()));
    // assert_eq!(Ordering::Equal, (-min_subnorm()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Equal, (-0.0_f16).total_cmp(&-0.0));
    assert_eq!(Ordering::Equal, 0.0_f16.total_cmp(&0.0));
    // assert_eq!(Ordering::Equal, min_subnorm().total_cmp(&min_subnorm()));
    // assert_eq!(Ordering::Equal, max_subnorm().total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Equal, f16::MIN_POSITIVE.total_cmp(&f16::MIN_POSITIVE));
    assert_eq!(Ordering::Equal, 0.5_f16.total_cmp(&0.5));
    assert_eq!(Ordering::Equal, 1.0_f16.total_cmp(&1.0));
    assert_eq!(Ordering::Equal, 1.5_f16.total_cmp(&1.5));
    assert_eq!(Ordering::Equal, 2.5_f16.total_cmp(&2.5));
    assert_eq!(Ordering::Equal, f16::MAX.total_cmp(&f16::MAX));
    assert_eq!(Ordering::Equal, f16::INFINITY.total_cmp(&f16::INFINITY));
    assert_eq!(Ordering::Equal, s_nan().total_cmp(&s_nan()));
    assert_eq!(Ordering::Equal, q_nan().total_cmp(&q_nan()));

    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f16::INFINITY));
    assert_eq!(Ordering::Less, (-f16::INFINITY).total_cmp(&-f16::MAX));
    assert_eq!(Ordering::Less, (-f16::MAX).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-2.5_f16).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-1.5_f16).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-1.0_f16).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-0.5_f16).total_cmp(&-f16::MIN_POSITIVE));
    // assert_eq!(Ordering::Less, (-f16::MIN_POSITIVE).total_cmp(&-max_subnorm()));
    // assert_eq!(Ordering::Less, (-max_subnorm()).total_cmp(&-min_subnorm()));
    // assert_eq!(Ordering::Less, (-min_subnorm()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-0.0_f16).total_cmp(&0.0));
    // assert_eq!(Ordering::Less, 0.0_f16.total_cmp(&min_subnorm()));
    // assert_eq!(Ordering::Less, min_subnorm().total_cmp(&max_subnorm()));
    // assert_eq!(Ordering::Less, max_subnorm().total_cmp(&f16::MIN_POSITIVE));
    assert_eq!(Ordering::Less, f16::MIN_POSITIVE.total_cmp(&0.5));
    assert_eq!(Ordering::Less, 0.5_f16.total_cmp(&1.0));
    assert_eq!(Ordering::Less, 1.0_f16.total_cmp(&1.5));
    assert_eq!(Ordering::Less, 1.5_f16.total_cmp(&2.5));
    assert_eq!(Ordering::Less, 2.5_f16.total_cmp(&f16::MAX));
    assert_eq!(Ordering::Less, f16::MAX.total_cmp(&f16::INFINITY));
    assert_eq!(Ordering::Less, f16::INFINITY.total_cmp(&s_nan()));
    assert_eq!(Ordering::Less, s_nan().total_cmp(&q_nan()));

    assert_eq!(Ordering::Greater, (-s_nan()).total_cmp(&-q_nan()));
    assert_eq!(Ordering::Greater, (-f16::INFINITY).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Greater, (-f16::MAX).total_cmp(&-f16::INFINITY));
    assert_eq!(Ordering::Greater, (-2.5_f16).total_cmp(&-f16::MAX));
    assert_eq!(Ordering::Greater, (-1.5_f16).total_cmp(&-2.5));
    assert_eq!(Ordering::Greater, (-1.0_f16).total_cmp(&-1.5));
    assert_eq!(Ordering::Greater, (-0.5_f16).total_cmp(&-1.0));
    assert_eq!(Ordering::Greater, (-f16::MIN_POSITIVE).total_cmp(&-0.5));
    // assert_eq!(Ordering::Greater, (-max_subnorm()).total_cmp(&-f16::MIN_POSITIVE));
    // assert_eq!(Ordering::Greater, (-min_subnorm()).total_cmp(&-max_subnorm()));
    // assert_eq!(Ordering::Greater, (-0.0_f16).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Greater, 0.0_f16.total_cmp(&-0.0));
    // assert_eq!(Ordering::Greater, min_subnorm().total_cmp(&0.0));
    // assert_eq!(Ordering::Greater, max_subnorm().total_cmp(&min_subnorm()));
    // assert_eq!(Ordering::Greater, f16::MIN_POSITIVE.total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Greater, 0.5_f16.total_cmp(&f16::MIN_POSITIVE));
    assert_eq!(Ordering::Greater, 1.0_f16.total_cmp(&0.5));
    assert_eq!(Ordering::Greater, 1.5_f16.total_cmp(&1.0));
    assert_eq!(Ordering::Greater, 2.5_f16.total_cmp(&1.5));
    assert_eq!(Ordering::Greater, f16::MAX.total_cmp(&2.5));
    assert_eq!(Ordering::Greater, f16::INFINITY.total_cmp(&f16::MAX));
    assert_eq!(Ordering::Greater, s_nan().total_cmp(&f16::INFINITY));
    assert_eq!(Ordering::Greater, q_nan().total_cmp(&s_nan()));

    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f16::INFINITY));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f16::MAX));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f16::MIN_POSITIVE));
    // assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-max_subnorm()));
    // assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&0.0));
    // assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&min_subnorm()));
    // assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f16::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&0.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&1.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&1.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&2.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f16::MAX));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f16::INFINITY));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&s_nan()));

    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f16::INFINITY));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f16::MAX));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f16::MIN_POSITIVE));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-max_subnorm()));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&0.0));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&min_subnorm()));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f16::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&0.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&1.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&1.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&2.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f16::MAX));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f16::INFINITY));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&s_nan()));
}
