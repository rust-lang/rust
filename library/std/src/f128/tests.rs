#![cfg(not(bootstrap))]
// FIXME(f16_f128): only tested on platforms that have symbols and aren't buggy
#![cfg(reliable_f128)]

use crate::f128::consts;
use crate::num::FpCategory as Fp;
use crate::num::*;

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
    test_num(10f128, 2f128);
}

// FIXME(f16_f128): add min and max tests when available

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

// FIXME(f16_f128): add missing math functions when available

#[test]
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
fn test_recip() {
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert_eq!(1.0f128.recip(), 1.0);
    assert_eq!(2.0f128.recip(), 0.5);
    assert_eq!((-0.4f128).recip(), -2.5);
    assert_eq!(0.0f128.recip(), inf);
    assert!(nan.recip().is_nan());
    assert_eq!(inf.recip(), 0.0);
    assert_eq!(neg_inf.recip(), 0.0);
}

#[test]
fn test_to_degrees() {
    let pi: f128 = consts::PI;
    let nan: f128 = f128::NAN;
    let inf: f128 = f128::INFINITY;
    let neg_inf: f128 = f128::NEG_INFINITY;
    assert_eq!(0.0f128.to_degrees(), 0.0);
    assert_approx_eq!((-5.8f128).to_degrees(), -332.315521);
    assert_eq!(pi.to_degrees(), 180.0);
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
    assert_approx_eq!(154.6f128.to_radians(), 2.698279);
    assert_approx_eq!((-332.31f128).to_radians(), -5.799903);
    // check approx rather than exact because round trip for pi doesn't fall on an exactly
    // representable value (unlike `f32` and `f64`).
    assert_approx_eq!(180.0f128.to_radians(), pi);
    assert!(nan.to_radians().is_nan());
    assert_eq!(inf.to_radians(), inf);
    assert_eq!(neg_inf.to_radians(), neg_inf);
}

#[test]
fn test_real_consts() {
    // FIXME(f16_f128): add math tests when available
    use super::consts;

    let pi: f128 = consts::PI;
    let frac_pi_2: f128 = consts::FRAC_PI_2;
    let frac_pi_3: f128 = consts::FRAC_PI_3;
    let frac_pi_4: f128 = consts::FRAC_PI_4;
    let frac_pi_6: f128 = consts::FRAC_PI_6;
    let frac_pi_8: f128 = consts::FRAC_PI_8;
    let frac_1_pi: f128 = consts::FRAC_1_PI;
    let frac_2_pi: f128 = consts::FRAC_2_PI;
    // let frac_2_sqrtpi: f128 = consts::FRAC_2_SQRT_PI;
    // let sqrt2: f128 = consts::SQRT_2;
    // let frac_1_sqrt2: f128 = consts::FRAC_1_SQRT_2;
    // let e: f128 = consts::E;
    // let log2_e: f128 = consts::LOG2_E;
    // let log10_e: f128 = consts::LOG10_E;
    // let ln_2: f128 = consts::LN_2;
    // let ln_10: f128 = consts::LN_10;

    assert_approx_eq!(frac_pi_2, pi / 2f128);
    assert_approx_eq!(frac_pi_3, pi / 3f128);
    assert_approx_eq!(frac_pi_4, pi / 4f128);
    assert_approx_eq!(frac_pi_6, pi / 6f128);
    assert_approx_eq!(frac_pi_8, pi / 8f128);
    assert_approx_eq!(frac_1_pi, 1f128 / pi);
    assert_approx_eq!(frac_2_pi, 2f128 / pi);
    // assert_approx_eq!(frac_2_sqrtpi, 2f128 / pi.sqrt());
    // assert_approx_eq!(sqrt2, 2f128.sqrt());
    // assert_approx_eq!(frac_1_sqrt2, 1f128 / 2f128.sqrt());
    // assert_approx_eq!(log2_e, e.log2());
    // assert_approx_eq!(log10_e, e.log10());
    // assert_approx_eq!(ln_2, 2f128.ln());
    // assert_approx_eq!(ln_10, 10f128.ln());
}

#[test]
fn test_float_bits_conv() {
    assert_eq!((1f128).to_bits(), 0x3fff0000000000000000000000000000);
    assert_eq!((12.5f128).to_bits(), 0x40029000000000000000000000000000);
    assert_eq!((1337f128).to_bits(), 0x40094e40000000000000000000000000);
    assert_eq!((-14.25f128).to_bits(), 0xc002c800000000000000000000000000);
    assert_approx_eq!(f128::from_bits(0x3fff0000000000000000000000000000), 1.0);
    assert_approx_eq!(f128::from_bits(0x40029000000000000000000000000000), 12.5);
    assert_approx_eq!(f128::from_bits(0x40094e40000000000000000000000000), 1337.0);
    assert_approx_eq!(f128::from_bits(0xc002c800000000000000000000000000), -14.25);

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
