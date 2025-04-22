// FIXME(f16_f128): only tested on platforms that have symbols and aren't buggy
#![cfg(reliable_f128)]

use std::f128::consts;
use std::num::FpCategory as Fp;
#[cfg(reliable_f128_math)]
use std::ops::Rem;
use std::ops::{Add, Div, Mul, Sub};

// Note these tolerances make sense around zero, but not for more extreme exponents.

/// For operations that are near exact, usually not involving math of different
/// signs.
const TOL_PRECISE: f128 = 1e-28;

/// Default tolerances. Works for values that should be near precise but not exact. Roughly
/// the precision carried by `100 * 100`.
const TOL: f128 = 1e-12;

/// Tolerances for math that is allowed to be imprecise, usually due to multiple chained
/// operations.
#[cfg(reliable_f128_math)]
const TOL_IMPR: f128 = 1e-10;

/// Smallest number
const TINY_BITS: u128 = 0x1;

/// Next smallest number
const TINY_UP_BITS: u128 = 0x2;

/// Exponent = 0b11...10, Sifnificand 0b1111..10. Min val > 0
const MAX_DOWN_BITS: u128 = 0x7ffefffffffffffffffffffffffffffe;

/// Zeroed exponent, full significant
const LARGEST_SUBNORMAL_BITS: u128 = 0x0000ffffffffffffffffffffffffffff;

/// Exponent = 0b1, zeroed significand
const SMALLEST_NORMAL_BITS: u128 = 0x00010000000000000000000000000000;

/// First pattern over the mantissa
const NAN_MASK1: u128 = 0x0000aaaaaaaaaaaaaaaaaaaaaaaaaaaa;

/// Second pattern over the mantissa
const NAN_MASK2: u128 = 0x00005555555555555555555555555555;

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

#[test]
#[cfg(reliable_f128_math)]
fn test_num_f128_rem() {
    let ten = 10f128;
    let two = 2f128;
    assert_eq!(ten.rem(two), ten % two);
}

#[test]
#[cfg(reliable_f128_math)]
fn test_min_nan() {
    assert_eq!(f128::NAN.min(2.0), 2.0);
    assert_eq!(2.0f128.min(f128::NAN), 2.0);
}

#[test]
#[cfg(reliable_f128_math)]
fn test_max_nan() {
    assert_eq!(f128::NAN.max(2.0), 2.0);
    assert_eq!(2.0f128.max(f128::NAN), 2.0);
}

#[test]
#[cfg(reliable_f128_math)]
fn test_minimum() {
    assert!(f128::NAN.minimum(2.0).is_nan());
    assert!(2.0f128.minimum(f128::NAN).is_nan());
}

#[test]
#[cfg(reliable_f128_math)]
fn test_maximum() {
    assert!(f128::NAN.maximum(2.0).is_nan());
    assert!(2.0f128.maximum(f128::NAN).is_nan());
}

#[test]
fn test_nan() {
    let nan: f128 = f128::NAN;
    assert!(nan.is_nan());
    assert!(!nan.is_infinite());
    assert!(!nan.is_finite());
    assert!(nan.is_sign_positive());
    assert!(!nan.is_sign_negative());
    assert!(!nan.is_normal());
    assert_eq!(Fp::Nan, nan.classify());
    // Ensure the quiet bit is set.
    assert!(nan.to_bits() & (1 << (f128::MANTISSA_DIGITS - 2)) != 0);
}

#[test]
fn test_infinity() {
    let inf: f128 = f128::INFINITY;
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
    let neg_inf: f128 = f128::NEG_INFINITY;
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
    let zero: f128 = 0.0f128;
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
    let neg_zero: f128 = -0.0;
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
    let one: f128 = 1.0f128;
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
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert!(nan.is_nan());
    assert!(!0.0f128.is_nan());
    assert!(!5.3f128.is_nan());
    assert!(!(-10.732f128).is_nan());
    assert!(!inf.is_nan());
    assert!(!neg_inf.is_nan());
}

#[test]
fn test_is_infinite() {
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert!(!nan.is_infinite());
    assert!(inf.is_infinite());
    assert!(neg_inf.is_infinite());
    assert!(!0.0f128.is_infinite());
    assert!(!42.8f128.is_infinite());
    assert!(!(-109.2f128).is_infinite());
}

#[test]
fn test_is_finite() {
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert!(!nan.is_finite());
    assert!(!inf.is_finite());
    assert!(!neg_inf.is_finite());
    assert!(0.0f128.is_finite());
    assert!(42.8f128.is_finite());
    assert!((-109.2f128).is_finite());
}

#[test]
fn test_is_normal() {
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    let zero: f128 = 0.0f128;
    let neg_zero: f128 = -0.0;
    assert!(!nan.is_normal());
    assert!(!inf.is_normal());
    assert!(!neg_inf.is_normal());
    assert!(!zero.is_normal());
    assert!(!neg_zero.is_normal());
    assert!(1f128.is_normal());
    assert!(1e-4931f128.is_normal());
    assert!(!1e-4932f128.is_normal());
}

#[test]
fn test_classify() {
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    let zero: f128 = 0.0f128;
    let neg_zero: f128 = -0.0;
    assert_eq!(nan.classify(), Fp::Nan);
    assert_eq!(inf.classify(), Fp::Infinite);
    assert_eq!(neg_inf.classify(), Fp::Infinite);
    assert_eq!(zero.classify(), Fp::Zero);
    assert_eq!(neg_zero.classify(), Fp::Zero);
    assert_eq!(1f128.classify(), Fp::Normal);
    assert_eq!(1e-4931f128.classify(), Fp::Normal);
    assert_eq!(1e-4932f128.classify(), Fp::Subnormal);
}

#[test]
#[cfg(reliable_f128_math)]
fn test_floor() {
    assert_approx_eq!(1.0f128.floor(), 1.0f128, TOL_PRECISE);
    assert_approx_eq!(1.3f128.floor(), 1.0f128, TOL_PRECISE);
    assert_approx_eq!(1.5f128.floor(), 1.0f128, TOL_PRECISE);
    assert_approx_eq!(1.7f128.floor(), 1.0f128, TOL_PRECISE);
    assert_approx_eq!(0.0f128.floor(), 0.0f128, TOL_PRECISE);
    assert_approx_eq!((-0.0f128).floor(), -0.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.0f128).floor(), -1.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.3f128).floor(), -2.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.5f128).floor(), -2.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.7f128).floor(), -2.0f128, TOL_PRECISE);
}

#[test]
#[cfg(reliable_f128_math)]
fn test_ceil() {
    assert_approx_eq!(1.0f128.ceil(), 1.0f128, TOL_PRECISE);
    assert_approx_eq!(1.3f128.ceil(), 2.0f128, TOL_PRECISE);
    assert_approx_eq!(1.5f128.ceil(), 2.0f128, TOL_PRECISE);
    assert_approx_eq!(1.7f128.ceil(), 2.0f128, TOL_PRECISE);
    assert_approx_eq!(0.0f128.ceil(), 0.0f128, TOL_PRECISE);
    assert_approx_eq!((-0.0f128).ceil(), -0.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.0f128).ceil(), -1.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.3f128).ceil(), -1.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.5f128).ceil(), -1.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.7f128).ceil(), -1.0f128, TOL_PRECISE);
}

#[test]
#[cfg(reliable_f128_math)]
fn test_round() {
    assert_approx_eq!(2.5f128.round(), 3.0f128, TOL_PRECISE);
    assert_approx_eq!(1.0f128.round(), 1.0f128, TOL_PRECISE);
    assert_approx_eq!(1.3f128.round(), 1.0f128, TOL_PRECISE);
    assert_approx_eq!(1.5f128.round(), 2.0f128, TOL_PRECISE);
    assert_approx_eq!(1.7f128.round(), 2.0f128, TOL_PRECISE);
    assert_approx_eq!(0.0f128.round(), 0.0f128, TOL_PRECISE);
    assert_approx_eq!((-0.0f128).round(), -0.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.0f128).round(), -1.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.3f128).round(), -1.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.5f128).round(), -2.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.7f128).round(), -2.0f128, TOL_PRECISE);
}

#[test]
#[cfg(reliable_f128_math)]
fn test_round_ties_even() {
    assert_approx_eq!(2.5f128.round_ties_even(), 2.0f128, TOL_PRECISE);
    assert_approx_eq!(1.0f128.round_ties_even(), 1.0f128, TOL_PRECISE);
    assert_approx_eq!(1.3f128.round_ties_even(), 1.0f128, TOL_PRECISE);
    assert_approx_eq!(1.5f128.round_ties_even(), 2.0f128, TOL_PRECISE);
    assert_approx_eq!(1.7f128.round_ties_even(), 2.0f128, TOL_PRECISE);
    assert_approx_eq!(0.0f128.round_ties_even(), 0.0f128, TOL_PRECISE);
    assert_approx_eq!((-0.0f128).round_ties_even(), -0.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.0f128).round_ties_even(), -1.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.3f128).round_ties_even(), -1.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.5f128).round_ties_even(), -2.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.7f128).round_ties_even(), -2.0f128, TOL_PRECISE);
}

#[test]
#[cfg(reliable_f128_math)]
fn test_trunc() {
    assert_approx_eq!(1.0f128.trunc(), 1.0f128, TOL_PRECISE);
    assert_approx_eq!(1.3f128.trunc(), 1.0f128, TOL_PRECISE);
    assert_approx_eq!(1.5f128.trunc(), 1.0f128, TOL_PRECISE);
    assert_approx_eq!(1.7f128.trunc(), 1.0f128, TOL_PRECISE);
    assert_approx_eq!(0.0f128.trunc(), 0.0f128, TOL_PRECISE);
    assert_approx_eq!((-0.0f128).trunc(), -0.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.0f128).trunc(), -1.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.3f128).trunc(), -1.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.5f128).trunc(), -1.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.7f128).trunc(), -1.0f128, TOL_PRECISE);
}

#[test]
#[cfg(reliable_f128_math)]
fn test_fract() {
    assert_approx_eq!(1.0f128.fract(), 0.0f128, TOL_PRECISE);
    assert_approx_eq!(1.3f128.fract(), 0.3f128, TOL_PRECISE);
    assert_approx_eq!(1.5f128.fract(), 0.5f128, TOL_PRECISE);
    assert_approx_eq!(1.7f128.fract(), 0.7f128, TOL_PRECISE);
    assert_approx_eq!(0.0f128.fract(), 0.0f128, TOL_PRECISE);
    assert_approx_eq!((-0.0f128).fract(), -0.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.0f128).fract(), -0.0f128, TOL_PRECISE);
    assert_approx_eq!((-1.3f128).fract(), -0.3f128, TOL_PRECISE);
    assert_approx_eq!((-1.5f128).fract(), -0.5f128, TOL_PRECISE);
    assert_approx_eq!((-1.7f128).fract(), -0.7f128, TOL_PRECISE);
}

#[test]
#[cfg(reliable_f128_math)]
fn test_abs() {
    assert_eq!(f128::INFINITY.abs(), f128::INFINITY);
    assert_eq!(1f128.abs(), 1f128);
    assert_eq!(0f128.abs(), 0f128);
    assert_eq!((-0f128).abs(), 0f128);
    assert_eq!((-1f128).abs(), 1f128);
    assert_eq!(f128::NEG_INFINITY.abs(), f128::INFINITY);
    assert_eq!((1f128 / f128::NEG_INFINITY).abs(), 0f128);
    assert!(f128::NAN.abs().is_nan());
}

#[test]
fn test_is_sign_positive() {
    assert!(f128::INFINITY.is_sign_positive());
    assert!(1f128.is_sign_positive());
    assert!(0f128.is_sign_positive());
    assert!(!(-0f128).is_sign_positive());
    assert!(!(-1f128).is_sign_positive());
    assert!(!f128::NEG_INFINITY.is_sign_positive());
    assert!(!(1f128 / f128::NEG_INFINITY).is_sign_positive());
    assert!(f128::NAN.is_sign_positive());
    assert!(!(-f128::NAN).is_sign_positive());
}

#[test]
fn test_is_sign_negative() {
    assert!(!f128::INFINITY.is_sign_negative());
    assert!(!1f128.is_sign_negative());
    assert!(!0f128.is_sign_negative());
    assert!((-0f128).is_sign_negative());
    assert!((-1f128).is_sign_negative());
    assert!(f128::NEG_INFINITY.is_sign_negative());
    assert!((1f128 / f128::NEG_INFINITY).is_sign_negative());
    assert!(!f128::NAN.is_sign_negative());
    assert!((-f128::NAN).is_sign_negative());
}

#[test]
fn test_next_up() {
    let tiny = f128::from_bits(TINY_BITS);
    let tiny_up = f128::from_bits(TINY_UP_BITS);
    let max_down = f128::from_bits(MAX_DOWN_BITS);
    let largest_subnormal = f128::from_bits(LARGEST_SUBNORMAL_BITS);
    let smallest_normal = f128::from_bits(SMALLEST_NORMAL_BITS);
    assert_f128_biteq!(f128::NEG_INFINITY.next_up(), f128::MIN);
    assert_f128_biteq!(f128::MIN.next_up(), -max_down);
    assert_f128_biteq!((-1.0 - f128::EPSILON).next_up(), -1.0);
    assert_f128_biteq!((-smallest_normal).next_up(), -largest_subnormal);
    assert_f128_biteq!((-tiny_up).next_up(), -tiny);
    assert_f128_biteq!((-tiny).next_up(), -0.0f128);
    assert_f128_biteq!((-0.0f128).next_up(), tiny);
    assert_f128_biteq!(0.0f128.next_up(), tiny);
    assert_f128_biteq!(tiny.next_up(), tiny_up);
    assert_f128_biteq!(largest_subnormal.next_up(), smallest_normal);
    assert_f128_biteq!(1.0f128.next_up(), 1.0 + f128::EPSILON);
    assert_f128_biteq!(f128::MAX.next_up(), f128::INFINITY);
    assert_f128_biteq!(f128::INFINITY.next_up(), f128::INFINITY);

    // Check that NaNs roundtrip.
    let nan0 = f128::NAN;
    let nan1 = f128::from_bits(f128::NAN.to_bits() ^ 0x002a_aaaa);
    let nan2 = f128::from_bits(f128::NAN.to_bits() ^ 0x0055_5555);
    assert_f128_biteq!(nan0.next_up(), nan0);
    assert_f128_biteq!(nan1.next_up(), nan1);
    assert_f128_biteq!(nan2.next_up(), nan2);
}

#[test]
fn test_next_down() {
    let tiny = f128::from_bits(TINY_BITS);
    let tiny_up = f128::from_bits(TINY_UP_BITS);
    let max_down = f128::from_bits(MAX_DOWN_BITS);
    let largest_subnormal = f128::from_bits(LARGEST_SUBNORMAL_BITS);
    let smallest_normal = f128::from_bits(SMALLEST_NORMAL_BITS);
    assert_f128_biteq!(f128::NEG_INFINITY.next_down(), f128::NEG_INFINITY);
    assert_f128_biteq!(f128::MIN.next_down(), f128::NEG_INFINITY);
    assert_f128_biteq!((-max_down).next_down(), f128::MIN);
    assert_f128_biteq!((-1.0f128).next_down(), -1.0 - f128::EPSILON);
    assert_f128_biteq!((-largest_subnormal).next_down(), -smallest_normal);
    assert_f128_biteq!((-tiny).next_down(), -tiny_up);
    assert_f128_biteq!((-0.0f128).next_down(), -tiny);
    assert_f128_biteq!((0.0f128).next_down(), -tiny);
    assert_f128_biteq!(tiny.next_down(), 0.0f128);
    assert_f128_biteq!(tiny_up.next_down(), tiny);
    assert_f128_biteq!(smallest_normal.next_down(), largest_subnormal);
    assert_f128_biteq!((1.0 + f128::EPSILON).next_down(), 1.0f128);
    assert_f128_biteq!(f128::MAX.next_down(), max_down);
    assert_f128_biteq!(f128::INFINITY.next_down(), f128::MAX);

    // Check that NaNs roundtrip.
    let nan0 = f128::NAN;
    let nan1 = f128::from_bits(f128::NAN.to_bits() ^ 0x002a_aaaa);
    let nan2 = f128::from_bits(f128::NAN.to_bits() ^ 0x0055_5555);
    assert_f128_biteq!(nan0.next_down(), nan0);
    assert_f128_biteq!(nan1.next_down(), nan1);
    assert_f128_biteq!(nan2.next_down(), nan2);
}

#[test]
#[cfg(reliable_f128_math)]
fn test_mul_add() {
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert_approx_eq!(12.3f128.mul_add(4.5, 6.7), 62.05, TOL_PRECISE);
    assert_approx_eq!((-12.3f128).mul_add(-4.5, -6.7), 48.65, TOL_PRECISE);
    assert_approx_eq!(0.0f128.mul_add(8.9, 1.2), 1.2, TOL_PRECISE);
    assert_approx_eq!(3.4f128.mul_add(-0.0, 5.6), 5.6, TOL_PRECISE);
    assert!(nan.mul_add(7.8, 9.0).is_nan());
    assert_eq!(inf.mul_add(7.8, 9.0), inf);
    assert_eq!(neg_inf.mul_add(7.8, 9.0), neg_inf);
    assert_eq!(8.9f128.mul_add(inf, 3.2), inf);
    assert_eq!((-3.2f128).mul_add(2.4, neg_inf), neg_inf);
}

#[test]
#[cfg(reliable_f16_math)]
fn test_recip() {
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert_eq!(1.0f128.recip(), 1.0);
    assert_eq!(2.0f128.recip(), 0.5);
    assert_eq!((-0.4f128).recip(), -2.5);
    assert_eq!(0.0f128.recip(), inf);
    assert_approx_eq!(
        f128::MAX.recip(),
        8.40525785778023376565669454330438228902076605e-4933,
        1e-4900
    );
    assert!(nan.recip().is_nan());
    assert_eq!(inf.recip(), 0.0);
    assert_eq!(neg_inf.recip(), 0.0);
}

// Many math functions allow for less accurate results, so the next tolerance up is used

#[test]
#[cfg(reliable_f128_math)]
fn test_powi() {
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert_eq!(1.0f128.powi(1), 1.0);
    assert_approx_eq!((-3.1f128).powi(2), 9.6100000000000005506706202140776519387, TOL);
    assert_approx_eq!(5.9f128.powi(-2), 0.028727377190462507313100483690639638451, TOL);
    assert_eq!(8.3f128.powi(0), 1.0);
    assert!(nan.powi(2).is_nan());
    assert_eq!(inf.powi(3), inf);
    assert_eq!(neg_inf.powi(2), inf);
}

#[test]
#[cfg(reliable_f128_math)]
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
#[cfg(reliable_f128_math)]
fn test_sqrt_domain() {
    assert!(f128::NAN.sqrt().is_nan());
    assert!(f128::NEG_INFINITY.sqrt().is_nan());
    assert!((-1.0f128).sqrt().is_nan());
    assert_eq!((-0.0f128).sqrt(), -0.0);
    assert_eq!(0.0f128.sqrt(), 0.0);
    assert_eq!(1.0f128.sqrt(), 1.0);
    assert_eq!(f128::INFINITY.sqrt(), f128::INFINITY);
}

#[test]
#[cfg(reliable_f128_math)]
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
#[cfg(reliable_f128_math)]
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
#[cfg(reliable_f128_math)]
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
#[cfg(reliable_f128_math)]
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
#[cfg(reliable_f128_math)]
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
#[cfg(reliable_f128_math)]
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
fn test_to_degrees() {
    let pi: f128 = consts::PI;
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert_eq!(0.0f128.to_degrees(), 0.0);
    assert_approx_eq!((-5.8f128).to_degrees(), -332.31552117587745090765431723855668471, TOL);
    assert_approx_eq!(pi.to_degrees(), 180.0, TOL);
    assert!(nan.to_degrees().is_nan());
    assert_eq!(inf.to_degrees(), inf);
    assert_eq!(neg_inf.to_degrees(), neg_inf);
    assert_eq!(1_f128.to_degrees(), 57.2957795130823208767981548141051703);
}

#[test]
fn test_to_radians() {
    let pi: f128 = consts::PI;
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert_eq!(0.0f128.to_radians(), 0.0);
    assert_approx_eq!(154.6f128.to_radians(), 2.6982790235832334267135442069489767804, TOL);
    assert_approx_eq!((-332.31f128).to_radians(), -5.7999036373023566567593094812182763013, TOL);
    // check approx rather than exact because round trip for pi doesn't fall on an exactly
    // representable value (unlike `f32` and `f64`).
    assert_approx_eq!(180.0f128.to_radians(), pi, TOL_PRECISE);
    assert!(nan.to_radians().is_nan());
    assert_eq!(inf.to_radians(), inf);
    assert_eq!(neg_inf.to_radians(), neg_inf);
}

#[test]
#[cfg(reliable_f128_math)]
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
#[cfg(reliable_f128_math)]
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
#[cfg(reliable_f128_math)]
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
#[cfg(reliable_f128_math)]
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
#[cfg(reliable_f128_math)]
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

    #[cfg(reliable_f128_math)]
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

#[test]
fn test_float_bits_conv() {
    assert_eq!((1f128).to_bits(), 0x3fff0000000000000000000000000000);
    assert_eq!((12.5f128).to_bits(), 0x40029000000000000000000000000000);
    assert_eq!((1337f128).to_bits(), 0x40094e40000000000000000000000000);
    assert_eq!((-14.25f128).to_bits(), 0xc002c800000000000000000000000000);
    assert_approx_eq!(f128::from_bits(0x3fff0000000000000000000000000000), 1.0, TOL_PRECISE);
    assert_approx_eq!(f128::from_bits(0x40029000000000000000000000000000), 12.5, TOL_PRECISE);
    assert_approx_eq!(f128::from_bits(0x40094e40000000000000000000000000), 1337.0, TOL_PRECISE);
    assert_approx_eq!(f128::from_bits(0xc002c800000000000000000000000000), -14.25, TOL_PRECISE);

    // Check that NaNs roundtrip their bits regardless of signaling-ness
    // 0xA is 0b1010; 0x5 is 0b0101 -- so these two together clobbers all the mantissa bits
    let masked_nan1 = f128::NAN.to_bits() ^ NAN_MASK1;
    let masked_nan2 = f128::NAN.to_bits() ^ NAN_MASK2;
    assert!(f128::from_bits(masked_nan1).is_nan());
    assert!(f128::from_bits(masked_nan2).is_nan());

    assert_eq!(f128::from_bits(masked_nan1).to_bits(), masked_nan1);
    assert_eq!(f128::from_bits(masked_nan2).to_bits(), masked_nan2);
}

#[test]
#[should_panic]
fn test_clamp_min_greater_than_max() {
    let _ = 1.0f128.clamp(3.0, 1.0);
}

#[test]
#[should_panic]
fn test_clamp_min_is_nan() {
    let _ = 1.0f128.clamp(f128::NAN, 1.0);
}

#[test]
#[should_panic]
fn test_clamp_max_is_nan() {
    let _ = 1.0f128.clamp(3.0, f128::NAN);
}

#[test]
fn test_total_cmp() {
    use core::cmp::Ordering;

    fn quiet_bit_mask() -> u128 {
        1 << (f128::MANTISSA_DIGITS - 2)
    }

    // FIXME(f16_f128): test subnormals when powf is available
    // fn min_subnorm() -> f128 {
    //     f128::MIN_POSITIVE / f128::powf(2.0, f128::MANTISSA_DIGITS as f128 - 1.0)
    // }

    // fn max_subnorm() -> f128 {
    //     f128::MIN_POSITIVE - min_subnorm()
    // }

    fn q_nan() -> f128 {
        f128::from_bits(f128::NAN.to_bits() | quiet_bit_mask())
    }

    fn s_nan() -> f128 {
        f128::from_bits((f128::NAN.to_bits() & !quiet_bit_mask()) + 42)
    }

    assert_eq!(Ordering::Equal, (-q_nan()).total_cmp(&-q_nan()));
    assert_eq!(Ordering::Equal, (-s_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Equal, (-f128::INFINITY).total_cmp(&-f128::INFINITY));
    assert_eq!(Ordering::Equal, (-f128::MAX).total_cmp(&-f128::MAX));
    assert_eq!(Ordering::Equal, (-2.5_f128).total_cmp(&-2.5));
    assert_eq!(Ordering::Equal, (-1.0_f128).total_cmp(&-1.0));
    assert_eq!(Ordering::Equal, (-1.5_f128).total_cmp(&-1.5));
    assert_eq!(Ordering::Equal, (-0.5_f128).total_cmp(&-0.5));
    assert_eq!(Ordering::Equal, (-f128::MIN_POSITIVE).total_cmp(&-f128::MIN_POSITIVE));
    // assert_eq!(Ordering::Equal, (-max_subnorm()).total_cmp(&-max_subnorm()));
    // assert_eq!(Ordering::Equal, (-min_subnorm()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Equal, (-0.0_f128).total_cmp(&-0.0));
    assert_eq!(Ordering::Equal, 0.0_f128.total_cmp(&0.0));
    // assert_eq!(Ordering::Equal, min_subnorm().total_cmp(&min_subnorm()));
    // assert_eq!(Ordering::Equal, max_subnorm().total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Equal, f128::MIN_POSITIVE.total_cmp(&f128::MIN_POSITIVE));
    assert_eq!(Ordering::Equal, 0.5_f128.total_cmp(&0.5));
    assert_eq!(Ordering::Equal, 1.0_f128.total_cmp(&1.0));
    assert_eq!(Ordering::Equal, 1.5_f128.total_cmp(&1.5));
    assert_eq!(Ordering::Equal, 2.5_f128.total_cmp(&2.5));
    assert_eq!(Ordering::Equal, f128::MAX.total_cmp(&f128::MAX));
    assert_eq!(Ordering::Equal, f128::INFINITY.total_cmp(&f128::INFINITY));
    assert_eq!(Ordering::Equal, s_nan().total_cmp(&s_nan()));
    assert_eq!(Ordering::Equal, q_nan().total_cmp(&q_nan()));

    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f128::INFINITY));
    assert_eq!(Ordering::Less, (-f128::INFINITY).total_cmp(&-f128::MAX));
    assert_eq!(Ordering::Less, (-f128::MAX).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-2.5_f128).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-1.5_f128).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-1.0_f128).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-0.5_f128).total_cmp(&-f128::MIN_POSITIVE));
    // assert_eq!(Ordering::Less, (-f128::MIN_POSITIVE).total_cmp(&-max_subnorm()));
    // assert_eq!(Ordering::Less, (-max_subnorm()).total_cmp(&-min_subnorm()));
    // assert_eq!(Ordering::Less, (-min_subnorm()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-0.0_f128).total_cmp(&0.0));
    // assert_eq!(Ordering::Less, 0.0_f128.total_cmp(&min_subnorm()));
    // assert_eq!(Ordering::Less, min_subnorm().total_cmp(&max_subnorm()));
    // assert_eq!(Ordering::Less, max_subnorm().total_cmp(&f128::MIN_POSITIVE));
    assert_eq!(Ordering::Less, f128::MIN_POSITIVE.total_cmp(&0.5));
    assert_eq!(Ordering::Less, 0.5_f128.total_cmp(&1.0));
    assert_eq!(Ordering::Less, 1.0_f128.total_cmp(&1.5));
    assert_eq!(Ordering::Less, 1.5_f128.total_cmp(&2.5));
    assert_eq!(Ordering::Less, 2.5_f128.total_cmp(&f128::MAX));
    assert_eq!(Ordering::Less, f128::MAX.total_cmp(&f128::INFINITY));
    assert_eq!(Ordering::Less, f128::INFINITY.total_cmp(&s_nan()));
    assert_eq!(Ordering::Less, s_nan().total_cmp(&q_nan()));

    assert_eq!(Ordering::Greater, (-s_nan()).total_cmp(&-q_nan()));
    assert_eq!(Ordering::Greater, (-f128::INFINITY).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Greater, (-f128::MAX).total_cmp(&-f128::INFINITY));
    assert_eq!(Ordering::Greater, (-2.5_f128).total_cmp(&-f128::MAX));
    assert_eq!(Ordering::Greater, (-1.5_f128).total_cmp(&-2.5));
    assert_eq!(Ordering::Greater, (-1.0_f128).total_cmp(&-1.5));
    assert_eq!(Ordering::Greater, (-0.5_f128).total_cmp(&-1.0));
    assert_eq!(Ordering::Greater, (-f128::MIN_POSITIVE).total_cmp(&-0.5));
    // assert_eq!(Ordering::Greater, (-max_subnorm()).total_cmp(&-f128::MIN_POSITIVE));
    // assert_eq!(Ordering::Greater, (-min_subnorm()).total_cmp(&-max_subnorm()));
    // assert_eq!(Ordering::Greater, (-0.0_f128).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Greater, 0.0_f128.total_cmp(&-0.0));
    // assert_eq!(Ordering::Greater, min_subnorm().total_cmp(&0.0));
    // assert_eq!(Ordering::Greater, max_subnorm().total_cmp(&min_subnorm()));
    // assert_eq!(Ordering::Greater, f128::MIN_POSITIVE.total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Greater, 0.5_f128.total_cmp(&f128::MIN_POSITIVE));
    assert_eq!(Ordering::Greater, 1.0_f128.total_cmp(&0.5));
    assert_eq!(Ordering::Greater, 1.5_f128.total_cmp(&1.0));
    assert_eq!(Ordering::Greater, 2.5_f128.total_cmp(&1.5));
    assert_eq!(Ordering::Greater, f128::MAX.total_cmp(&2.5));
    assert_eq!(Ordering::Greater, f128::INFINITY.total_cmp(&f128::MAX));
    assert_eq!(Ordering::Greater, s_nan().total_cmp(&f128::INFINITY));
    assert_eq!(Ordering::Greater, q_nan().total_cmp(&s_nan()));

    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f128::INFINITY));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f128::MAX));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f128::MIN_POSITIVE));
    // assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-max_subnorm()));
    // assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&0.0));
    // assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&min_subnorm()));
    // assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f128::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&0.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&1.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&1.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&2.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f128::MAX));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f128::INFINITY));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&s_nan()));

    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f128::INFINITY));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f128::MAX));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f128::MIN_POSITIVE));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-max_subnorm()));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&0.0));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&min_subnorm()));
    // assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f128::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&0.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&1.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&1.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&2.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f128::MAX));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f128::INFINITY));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&s_nan()));
}

#[test]
fn test_algebraic() {
    let a: f128 = 123.0;
    let b: f128 = 456.0;

    // Check that individual operations match their primitive counterparts.
    //
    // This is a check of current implementations and does NOT imply any form of
    // guarantee about future behavior. The compiler reserves the right to make
    // these operations inexact matches in the future.
    let eps = if cfg!(miri) { 1e-6 } else { 0.0 };

    assert_approx_eq!(a.algebraic_add(b), a + b, eps);
    assert_approx_eq!(a.algebraic_sub(b), a - b, eps);
    assert_approx_eq!(a.algebraic_mul(b), a * b, eps);
    assert_approx_eq!(a.algebraic_div(b), a / b, eps);
    assert_approx_eq!(a.algebraic_rem(b), a % b, eps);
}

#[test]
fn test_from() {
    assert_eq!(f128::from(false), 0.0);
    assert_eq!(f128::from(true), 1.0);
    assert_eq!(f128::from(u8::MIN), 0.0);
    assert_eq!(f128::from(42_u8), 42.0);
    assert_eq!(f128::from(u8::MAX), 255.0);
    assert_eq!(f128::from(i8::MIN), -128.0);
    assert_eq!(f128::from(42_i8), 42.0);
    assert_eq!(f128::from(i8::MAX), 127.0);
    assert_eq!(f128::from(u16::MIN), 0.0);
    assert_eq!(f128::from(42_u16), 42.0);
    assert_eq!(f128::from(u16::MAX), 65535.0);
    assert_eq!(f128::from(i16::MIN), -32768.0);
    assert_eq!(f128::from(42_i16), 42.0);
    assert_eq!(f128::from(i16::MAX), 32767.0);
    assert_eq!(f128::from(u32::MIN), 0.0);
    assert_eq!(f128::from(42_u32), 42.0);
    assert_eq!(f128::from(u32::MAX), 4294967295.0);
    assert_eq!(f128::from(i32::MIN), -2147483648.0);
    assert_eq!(f128::from(42_i32), 42.0);
    assert_eq!(f128::from(i32::MAX), 2147483647.0);
    // FIXME(f16_f128): Uncomment these tests once the From<{u64,i64}> impls are added.
    // assert_eq!(f128::from(u64::MIN), 0.0);
    // assert_eq!(f128::from(42_u64), 42.0);
    // assert_eq!(f128::from(u64::MAX), 18446744073709551615.0);
    // assert_eq!(f128::from(i64::MIN), -9223372036854775808.0);
    // assert_eq!(f128::from(42_i64), 42.0);
    // assert_eq!(f128::from(i64::MAX), 9223372036854775807.0);
}
