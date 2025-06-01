// FIXME(f16_f128): only tested on platforms that have symbols and aren't buggy
#![cfg(target_has_reliable_f16)]

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

#[test]
fn test_num_f16() {
    super::test_num(10f16, 2f16);
}

// FIXME(f16_f128,miri): many of these have to be disabled since miri does not yet support
// the intrinsics.

#[test]
#[cfg(any(miri, target_has_reliable_f16_math))]
fn test_min_nan() {
    assert_biteq!(f16::NAN.min(2.0), 2.0);
    assert_biteq!(2.0f16.min(f16::NAN), 2.0);
}

#[test]
#[cfg(any(miri, target_has_reliable_f16_math))]
fn test_max_nan() {
    assert_biteq!(f16::NAN.max(2.0), 2.0);
    assert_biteq!(2.0f16.max(f16::NAN), 2.0);
}

#[test]
#[cfg(any(miri, target_has_reliable_f16_math))]
fn test_minimum() {
    assert!(f16::NAN.minimum(2.0).is_nan());
    assert!(2.0f16.minimum(f16::NAN).is_nan());
}

#[test]
#[cfg(any(miri, target_has_reliable_f16_math))]
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
    // Ensure the quiet bit is set.
    assert!(nan.to_bits() & (1 << (f16::MANTISSA_DIGITS - 2)) != 0);
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
    assert_biteq!(0.0, zero);
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
    assert_biteq!(-0.0, neg_zero);
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
    assert_biteq!(1.0, one);
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
#[cfg(any(miri, target_has_reliable_f16_math))]
fn test_floor() {
    assert_biteq!(1.0f16.floor(), 1.0f16);
    assert_biteq!(1.3f16.floor(), 1.0f16);
    assert_biteq!(1.5f16.floor(), 1.0f16);
    assert_biteq!(1.7f16.floor(), 1.0f16);
    assert_biteq!(0.0f16.floor(), 0.0f16);
    assert_biteq!((-0.0f16).floor(), -0.0f16);
    assert_biteq!((-1.0f16).floor(), -1.0f16);
    assert_biteq!((-1.3f16).floor(), -2.0f16);
    assert_biteq!((-1.5f16).floor(), -2.0f16);
    assert_biteq!((-1.7f16).floor(), -2.0f16);
}

#[test]
#[cfg(any(miri, target_has_reliable_f16_math))]
fn test_ceil() {
    assert_biteq!(1.0f16.ceil(), 1.0f16);
    assert_biteq!(1.3f16.ceil(), 2.0f16);
    assert_biteq!(1.5f16.ceil(), 2.0f16);
    assert_biteq!(1.7f16.ceil(), 2.0f16);
    assert_biteq!(0.0f16.ceil(), 0.0f16);
    assert_biteq!((-0.0f16).ceil(), -0.0f16);
    assert_biteq!((-1.0f16).ceil(), -1.0f16);
    assert_biteq!((-1.3f16).ceil(), -1.0f16);
    assert_biteq!((-1.5f16).ceil(), -1.0f16);
    assert_biteq!((-1.7f16).ceil(), -1.0f16);
}

#[test]
#[cfg(any(miri, target_has_reliable_f16_math))]
fn test_round() {
    assert_biteq!(2.5f16.round(), 3.0f16);
    assert_biteq!(1.0f16.round(), 1.0f16);
    assert_biteq!(1.3f16.round(), 1.0f16);
    assert_biteq!(1.5f16.round(), 2.0f16);
    assert_biteq!(1.7f16.round(), 2.0f16);
    assert_biteq!(0.0f16.round(), 0.0f16);
    assert_biteq!((-0.0f16).round(), -0.0f16);
    assert_biteq!((-1.0f16).round(), -1.0f16);
    assert_biteq!((-1.3f16).round(), -1.0f16);
    assert_biteq!((-1.5f16).round(), -2.0f16);
    assert_biteq!((-1.7f16).round(), -2.0f16);
}

#[test]
#[cfg(any(miri, target_has_reliable_f16_math))]
fn test_round_ties_even() {
    assert_biteq!(2.5f16.round_ties_even(), 2.0f16);
    assert_biteq!(1.0f16.round_ties_even(), 1.0f16);
    assert_biteq!(1.3f16.round_ties_even(), 1.0f16);
    assert_biteq!(1.5f16.round_ties_even(), 2.0f16);
    assert_biteq!(1.7f16.round_ties_even(), 2.0f16);
    assert_biteq!(0.0f16.round_ties_even(), 0.0f16);
    assert_biteq!((-0.0f16).round_ties_even(), -0.0f16);
    assert_biteq!((-1.0f16).round_ties_even(), -1.0f16);
    assert_biteq!((-1.3f16).round_ties_even(), -1.0f16);
    assert_biteq!((-1.5f16).round_ties_even(), -2.0f16);
    assert_biteq!((-1.7f16).round_ties_even(), -2.0f16);
}

#[test]
#[cfg(any(miri, target_has_reliable_f16_math))]
fn test_trunc() {
    assert_biteq!(1.0f16.trunc(), 1.0f16);
    assert_biteq!(1.3f16.trunc(), 1.0f16);
    assert_biteq!(1.5f16.trunc(), 1.0f16);
    assert_biteq!(1.7f16.trunc(), 1.0f16);
    assert_biteq!(0.0f16.trunc(), 0.0f16);
    assert_biteq!((-0.0f16).trunc(), -0.0f16);
    assert_biteq!((-1.0f16).trunc(), -1.0f16);
    assert_biteq!((-1.3f16).trunc(), -1.0f16);
    assert_biteq!((-1.5f16).trunc(), -1.0f16);
    assert_biteq!((-1.7f16).trunc(), -1.0f16);
}

#[test]
#[cfg(any(miri, target_has_reliable_f16_math))]
fn test_fract() {
    assert_biteq!(1.0f16.fract(), 0.0f16);
    assert_biteq!(1.3f16.fract(), 0.2998f16);
    assert_biteq!(1.5f16.fract(), 0.5f16);
    assert_biteq!(1.7f16.fract(), 0.7f16);
    assert_biteq!(0.0f16.fract(), 0.0f16);
    assert_biteq!((-0.0f16).fract(), 0.0f16);
    assert_biteq!((-1.0f16).fract(), 0.0f16);
    assert_biteq!((-1.3f16).fract(), -0.2998f16);
    assert_biteq!((-1.5f16).fract(), -0.5f16);
    assert_biteq!((-1.7f16).fract(), -0.7f16);
}

#[test]
#[cfg(any(miri, target_has_reliable_f16_math))]
fn test_abs() {
    assert_biteq!(f16::INFINITY.abs(), f16::INFINITY);
    assert_biteq!(1f16.abs(), 1f16);
    assert_biteq!(0f16.abs(), 0f16);
    assert_biteq!((-0f16).abs(), 0f16);
    assert_biteq!((-1f16).abs(), 1f16);
    assert_biteq!(f16::NEG_INFINITY.abs(), f16::INFINITY);
    assert_biteq!((1f16 / f16::NEG_INFINITY).abs(), 0f16);
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
    assert_biteq!(f16::NEG_INFINITY.next_up(), f16::MIN);
    assert_biteq!(f16::MIN.next_up(), -max_down);
    assert_biteq!((-1.0 - f16::EPSILON).next_up(), -1.0f16);
    assert_biteq!((-smallest_normal).next_up(), -largest_subnormal);
    assert_biteq!((-tiny_up).next_up(), -tiny);
    assert_biteq!((-tiny).next_up(), -0.0f16);
    assert_biteq!((-0.0f16).next_up(), tiny);
    assert_biteq!(0.0f16.next_up(), tiny);
    assert_biteq!(tiny.next_up(), tiny_up);
    assert_biteq!(largest_subnormal.next_up(), smallest_normal);
    assert_biteq!(1.0f16.next_up(), 1.0 + f16::EPSILON);
    assert_biteq!(f16::MAX.next_up(), f16::INFINITY);
    assert_biteq!(f16::INFINITY.next_up(), f16::INFINITY);

    // Check that NaNs roundtrip.
    let nan0 = f16::NAN;
    let nan1 = f16::from_bits(f16::NAN.to_bits() ^ NAN_MASK1);
    let nan2 = f16::from_bits(f16::NAN.to_bits() ^ NAN_MASK2);
    assert_biteq!(nan0.next_up(), nan0);
    assert_biteq!(nan1.next_up(), nan1);
    assert_biteq!(nan2.next_up(), nan2);
}

#[test]
fn test_next_down() {
    let tiny = f16::from_bits(TINY_BITS);
    let tiny_up = f16::from_bits(TINY_UP_BITS);
    let max_down = f16::from_bits(MAX_DOWN_BITS);
    let largest_subnormal = f16::from_bits(LARGEST_SUBNORMAL_BITS);
    let smallest_normal = f16::from_bits(SMALLEST_NORMAL_BITS);
    assert_biteq!(f16::NEG_INFINITY.next_down(), f16::NEG_INFINITY);
    assert_biteq!(f16::MIN.next_down(), f16::NEG_INFINITY);
    assert_biteq!((-max_down).next_down(), f16::MIN);
    assert_biteq!((-1.0f16).next_down(), -1.0 - f16::EPSILON);
    assert_biteq!((-largest_subnormal).next_down(), -smallest_normal);
    assert_biteq!((-tiny).next_down(), -tiny_up);
    assert_biteq!((-0.0f16).next_down(), -tiny);
    assert_biteq!((0.0f16).next_down(), -tiny);
    assert_biteq!(tiny.next_down(), 0.0f16);
    assert_biteq!(tiny_up.next_down(), tiny);
    assert_biteq!(smallest_normal.next_down(), largest_subnormal);
    assert_biteq!((1.0 + f16::EPSILON).next_down(), 1.0f16);
    assert_biteq!(f16::MAX.next_down(), max_down);
    assert_biteq!(f16::INFINITY.next_down(), f16::MAX);

    // Check that NaNs roundtrip.
    let nan0 = f16::NAN;
    let nan1 = f16::from_bits(f16::NAN.to_bits() ^ NAN_MASK1);
    let nan2 = f16::from_bits(f16::NAN.to_bits() ^ NAN_MASK2);
    assert_biteq!(nan0.next_down(), nan0);
    assert_biteq!(nan1.next_down(), nan1);
    assert_biteq!(nan2.next_down(), nan2);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_mul_add() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert_biteq!(12.3f16.mul_add(4.5, 6.7), 62.031);
    assert_biteq!((-12.3f16).mul_add(-4.5, -6.7), 48.625);
    assert_biteq!(0.0f16.mul_add(8.9, 1.2), 1.2);
    assert_biteq!(3.4f16.mul_add(-0.0, 5.6), 5.6);
    assert!(nan.mul_add(7.8, 9.0).is_nan());
    assert_biteq!(inf.mul_add(7.8, 9.0), inf);
    assert_biteq!(neg_inf.mul_add(7.8, 9.0), neg_inf);
    assert_biteq!(8.9f16.mul_add(inf, 3.2), inf);
    assert_biteq!((-3.2f16).mul_add(2.4, neg_inf), neg_inf);
}

#[test]
#[cfg(any(miri, target_has_reliable_f16_math))]
fn test_recip() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert_biteq!(1.0f16.recip(), 1.0);
    assert_biteq!(2.0f16.recip(), 0.5);
    assert_biteq!((-0.4f16).recip(), -2.5);
    assert_biteq!(0.0f16.recip(), inf);
    assert_approx_eq!(f16::MAX.recip(), 1.526624e-5f16, 1e-4);
    assert!(nan.recip().is_nan());
    assert_biteq!(inf.recip(), 0.0);
    assert_biteq!(neg_inf.recip(), -0.0);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_powi() {
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert_biteq!(1.0f16.powi(1), 1.0);
    assert_approx_eq!((-3.1f16).powi(2), 9.61, TOL_0);
    assert_approx_eq!(5.9f16.powi(-2), 0.028727, TOL_N2);
    assert_biteq!(8.3f16.powi(0), 1.0);
    assert!(nan.powi(2).is_nan());
    assert_biteq!(inf.powi(3), inf);
    assert_biteq!(neg_inf.powi(2), inf);
}

#[test]
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_sqrt_domain() {
    assert!(f16::NAN.sqrt().is_nan());
    assert!(f16::NEG_INFINITY.sqrt().is_nan());
    assert!((-1.0f16).sqrt().is_nan());
    assert_biteq!((-0.0f16).sqrt(), -0.0);
    assert_biteq!(0.0f16.sqrt(), 0.0);
    assert_biteq!(1.0f16.sqrt(), 1.0);
    assert_biteq!(f16::INFINITY.sqrt(), f16::INFINITY);
}

#[test]
fn test_to_degrees() {
    let pi: f16 = consts::PI;
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert_biteq!(0.0f16.to_degrees(), 0.0);
    assert_approx_eq!((-5.8f16).to_degrees(), -332.315521, TOL_P2);
    assert_approx_eq!(pi.to_degrees(), 180.0, TOL_P2);
    assert!(nan.to_degrees().is_nan());
    assert_biteq!(inf.to_degrees(), inf);
    assert_biteq!(neg_inf.to_degrees(), neg_inf);
    assert_biteq!(1_f16.to_degrees(), 57.2957795130823208767981548141051703);
}

#[test]
fn test_to_radians() {
    let pi: f16 = consts::PI;
    let nan: f16 = f16::NAN;
    let inf: f16 = f16::INFINITY;
    let neg_inf: f16 = f16::NEG_INFINITY;
    assert_biteq!(0.0f16.to_radians(), 0.0);
    assert_approx_eq!(154.6f16.to_radians(), 2.698279, TOL_0);
    assert_approx_eq!((-332.31f16).to_radians(), -5.799903, TOL_0);
    assert_approx_eq!(180.0f16.to_radians(), pi, TOL_0);
    assert!(nan.to_radians().is_nan());
    assert_biteq!(inf.to_radians(), inf);
    assert_biteq!(neg_inf.to_radians(), neg_inf);
}

#[test]
fn test_float_bits_conv() {
    assert_eq!((1f16).to_bits(), 0x3c00);
    assert_eq!((12.5f16).to_bits(), 0x4a40);
    assert_eq!((1337f16).to_bits(), 0x6539);
    assert_eq!((-14.25f16).to_bits(), 0xcb20);
    assert_biteq!(f16::from_bits(0x3c00), 1.0);
    assert_biteq!(f16::from_bits(0x4a40), 12.5);
    assert_biteq!(f16::from_bits(0x6539), 1337.0);
    assert_biteq!(f16::from_bits(0xcb20), -14.25);

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
#[cfg(not(miri))]
#[cfg(target_has_reliable_f16_math)]
fn test_total_cmp() {
    use core::cmp::Ordering;

    fn quiet_bit_mask() -> u16 {
        1 << (f16::MANTISSA_DIGITS - 2)
    }

    fn min_subnorm() -> f16 {
        f16::MIN_POSITIVE / f16::powf(2.0, f16::MANTISSA_DIGITS as f16 - 1.0)
    }

    fn max_subnorm() -> f16 {
        f16::MIN_POSITIVE - min_subnorm()
    }

    fn q_nan() -> f16 {
        f16::from_bits(f16::NAN.to_bits() | quiet_bit_mask())
    }

    // FIXME(f16_f128): Tests involving sNaN are disabled because without optimizations,
    // `total_cmp` is getting incorrectly lowered to code that includes a `extend`/`trunc` round
    // trip, which quiets sNaNs. See: https://github.com/llvm/llvm-project/issues/104915
    // fn s_nan() -> f16 {
    //     f16::from_bits((f16::NAN.to_bits() & !quiet_bit_mask()) + 42)
    // }

    assert_eq!(Ordering::Equal, (-q_nan()).total_cmp(&-q_nan()));
    // assert_eq!(Ordering::Equal, (-s_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Equal, (-f16::INFINITY).total_cmp(&-f16::INFINITY));
    assert_eq!(Ordering::Equal, (-f16::MAX).total_cmp(&-f16::MAX));
    assert_eq!(Ordering::Equal, (-2.5_f16).total_cmp(&-2.5));
    assert_eq!(Ordering::Equal, (-1.0_f16).total_cmp(&-1.0));
    assert_eq!(Ordering::Equal, (-1.5_f16).total_cmp(&-1.5));
    assert_eq!(Ordering::Equal, (-0.5_f16).total_cmp(&-0.5));
    assert_eq!(Ordering::Equal, (-f16::MIN_POSITIVE).total_cmp(&-f16::MIN_POSITIVE));
    assert_eq!(Ordering::Equal, (-max_subnorm()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Equal, (-min_subnorm()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Equal, (-0.0_f16).total_cmp(&-0.0));
    assert_eq!(Ordering::Equal, 0.0_f16.total_cmp(&0.0));
    assert_eq!(Ordering::Equal, min_subnorm().total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Equal, max_subnorm().total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Equal, f16::MIN_POSITIVE.total_cmp(&f16::MIN_POSITIVE));
    assert_eq!(Ordering::Equal, 0.5_f16.total_cmp(&0.5));
    assert_eq!(Ordering::Equal, 1.0_f16.total_cmp(&1.0));
    assert_eq!(Ordering::Equal, 1.5_f16.total_cmp(&1.5));
    assert_eq!(Ordering::Equal, 2.5_f16.total_cmp(&2.5));
    assert_eq!(Ordering::Equal, f16::MAX.total_cmp(&f16::MAX));
    assert_eq!(Ordering::Equal, f16::INFINITY.total_cmp(&f16::INFINITY));
    // assert_eq!(Ordering::Equal, s_nan().total_cmp(&s_nan()));
    assert_eq!(Ordering::Equal, q_nan().total_cmp(&q_nan()));

    // assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-s_nan()));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f16::INFINITY));
    assert_eq!(Ordering::Less, (-f16::INFINITY).total_cmp(&-f16::MAX));
    assert_eq!(Ordering::Less, (-f16::MAX).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-2.5_f16).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-1.5_f16).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-1.0_f16).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-0.5_f16).total_cmp(&-f16::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-f16::MIN_POSITIVE).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Less, (-max_subnorm()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-min_subnorm()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-0.0_f16).total_cmp(&0.0));
    assert_eq!(Ordering::Less, 0.0_f16.total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Less, min_subnorm().total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, max_subnorm().total_cmp(&f16::MIN_POSITIVE));
    assert_eq!(Ordering::Less, f16::MIN_POSITIVE.total_cmp(&0.5));
    assert_eq!(Ordering::Less, 0.5_f16.total_cmp(&1.0));
    assert_eq!(Ordering::Less, 1.0_f16.total_cmp(&1.5));
    assert_eq!(Ordering::Less, 1.5_f16.total_cmp(&2.5));
    assert_eq!(Ordering::Less, 2.5_f16.total_cmp(&f16::MAX));
    assert_eq!(Ordering::Less, f16::MAX.total_cmp(&f16::INFINITY));
    // assert_eq!(Ordering::Less, f16::INFINITY.total_cmp(&s_nan()));
    // assert_eq!(Ordering::Less, s_nan().total_cmp(&q_nan()));

    // assert_eq!(Ordering::Greater, (-s_nan()).total_cmp(&-q_nan()));
    // assert_eq!(Ordering::Greater, (-f16::INFINITY).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Greater, (-f16::MAX).total_cmp(&-f16::INFINITY));
    assert_eq!(Ordering::Greater, (-2.5_f16).total_cmp(&-f16::MAX));
    assert_eq!(Ordering::Greater, (-1.5_f16).total_cmp(&-2.5));
    assert_eq!(Ordering::Greater, (-1.0_f16).total_cmp(&-1.5));
    assert_eq!(Ordering::Greater, (-0.5_f16).total_cmp(&-1.0));
    assert_eq!(Ordering::Greater, (-f16::MIN_POSITIVE).total_cmp(&-0.5));
    assert_eq!(Ordering::Greater, (-max_subnorm()).total_cmp(&-f16::MIN_POSITIVE));
    assert_eq!(Ordering::Greater, (-min_subnorm()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Greater, (-0.0_f16).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Greater, 0.0_f16.total_cmp(&-0.0));
    assert_eq!(Ordering::Greater, min_subnorm().total_cmp(&0.0));
    assert_eq!(Ordering::Greater, max_subnorm().total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Greater, f16::MIN_POSITIVE.total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Greater, 0.5_f16.total_cmp(&f16::MIN_POSITIVE));
    assert_eq!(Ordering::Greater, 1.0_f16.total_cmp(&0.5));
    assert_eq!(Ordering::Greater, 1.5_f16.total_cmp(&1.0));
    assert_eq!(Ordering::Greater, 2.5_f16.total_cmp(&1.5));
    assert_eq!(Ordering::Greater, f16::MAX.total_cmp(&2.5));
    assert_eq!(Ordering::Greater, f16::INFINITY.total_cmp(&f16::MAX));
    // assert_eq!(Ordering::Greater, s_nan().total_cmp(&f16::INFINITY));
    // assert_eq!(Ordering::Greater, q_nan().total_cmp(&s_nan()));

    // assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f16::INFINITY));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f16::MAX));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f16::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&0.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f16::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&0.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&1.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&1.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&2.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f16::MAX));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f16::INFINITY));
    // assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&s_nan()));

    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f16::INFINITY));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f16::MAX));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-2.5));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-1.5));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-1.0));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-0.5));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f16::MIN_POSITIVE));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-max_subnorm()));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-min_subnorm()));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-0.0));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&0.0));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&min_subnorm()));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&max_subnorm()));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f16::MIN_POSITIVE));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&0.5));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&1.0));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&1.5));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&2.5));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f16::MAX));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f16::INFINITY));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&s_nan()));
}

#[test]
fn test_algebraic() {
    let a: f16 = 123.0;
    let b: f16 = 456.0;

    // Check that individual operations match their primitive counterparts.
    //
    // This is a check of current implementations and does NOT imply any form of
    // guarantee about future behavior. The compiler reserves the right to make
    // these operations inexact matches in the future.
    let eps_add = if cfg!(miri) { 1e1 } else { 0.0 };
    let eps_mul = if cfg!(miri) { 1e3 } else { 0.0 };
    let eps_div = if cfg!(miri) { 1e0 } else { 0.0 };

    assert_approx_eq!(a.algebraic_add(b), a + b, eps_add);
    assert_approx_eq!(a.algebraic_sub(b), a - b, eps_add);
    assert_approx_eq!(a.algebraic_mul(b), a * b, eps_mul);
    assert_approx_eq!(a.algebraic_div(b), a / b, eps_div);
    assert_approx_eq!(a.algebraic_rem(b), a % b, eps_div);
}

#[test]
fn test_from() {
    assert_biteq!(f16::from(false), 0.0);
    assert_biteq!(f16::from(true), 1.0);
    assert_biteq!(f16::from(u8::MIN), 0.0);
    assert_biteq!(f16::from(42_u8), 42.0);
    assert_biteq!(f16::from(u8::MAX), 255.0);
    assert_biteq!(f16::from(i8::MIN), -128.0);
    assert_biteq!(f16::from(42_i8), 42.0);
    assert_biteq!(f16::from(i8::MAX), 127.0);
}
