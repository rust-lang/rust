use core::f32;
use core::f32::consts;
use core::num::FpCategory as Fp;

/// Smallest number
const TINY_BITS: u32 = 0x1;

/// Next smallest number
const TINY_UP_BITS: u32 = 0x2;

/// Exponent = 0b11...10, Sifnificand 0b1111..10. Min val > 0
const MAX_DOWN_BITS: u32 = 0x7f7f_fffe;

/// Zeroed exponent, full significant
const LARGEST_SUBNORMAL_BITS: u32 = 0x007f_ffff;

/// Exponent = 0b1, zeroed significand
const SMALLEST_NORMAL_BITS: u32 = 0x0080_0000;

/// First pattern over the mantissa
const NAN_MASK1: u32 = 0x002a_aaaa;

/// Second pattern over the mantissa
const NAN_MASK2: u32 = 0x0055_5555;

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
fn test_num_f32() {
    super::test_num(10f32, 2f32);
}

#[test]
fn test_min_nan() {
    assert_eq!(f32::NAN.min(2.0), 2.0);
    assert_eq!(2.0f32.min(f32::NAN), 2.0);
}

#[test]
fn test_max_nan() {
    assert_eq!(f32::NAN.max(2.0), 2.0);
    assert_eq!(2.0f32.max(f32::NAN), 2.0);
}

#[test]
fn test_minimum() {
    assert!(f32::NAN.minimum(2.0).is_nan());
    assert!(2.0f32.minimum(f32::NAN).is_nan());
}

#[test]
fn test_maximum() {
    assert!(f32::NAN.maximum(2.0).is_nan());
    assert!(2.0f32.maximum(f32::NAN).is_nan());
}

#[test]
fn test_nan() {
    let nan: f32 = f32::NAN;
    assert!(nan.is_nan());
    assert!(!nan.is_infinite());
    assert!(!nan.is_finite());
    assert!(!nan.is_normal());
    assert!(nan.is_sign_positive());
    assert!(!nan.is_sign_negative());
    assert_eq!(Fp::Nan, nan.classify());
    // Ensure the quiet bit is set.
    assert!(nan.to_bits() & (1 << (f32::MANTISSA_DIGITS - 2)) != 0);
}

#[test]
fn test_infinity() {
    let inf: f32 = f32::INFINITY;
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
    let neg_inf: f32 = f32::NEG_INFINITY;
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
    let zero: f32 = 0.0f32;
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
    let neg_zero: f32 = -0.0;
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
    let one: f32 = 1.0f32;
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
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert!(nan.is_nan());
    assert!(!0.0f32.is_nan());
    assert!(!5.3f32.is_nan());
    assert!(!(-10.732f32).is_nan());
    assert!(!inf.is_nan());
    assert!(!neg_inf.is_nan());
}

#[test]
fn test_is_infinite() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert!(!nan.is_infinite());
    assert!(inf.is_infinite());
    assert!(neg_inf.is_infinite());
    assert!(!0.0f32.is_infinite());
    assert!(!42.8f32.is_infinite());
    assert!(!(-109.2f32).is_infinite());
}

#[test]
fn test_is_finite() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert!(!nan.is_finite());
    assert!(!inf.is_finite());
    assert!(!neg_inf.is_finite());
    assert!(0.0f32.is_finite());
    assert!(42.8f32.is_finite());
    assert!((-109.2f32).is_finite());
}

#[test]
fn test_is_normal() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    let zero: f32 = 0.0f32;
    let neg_zero: f32 = -0.0;
    assert!(!nan.is_normal());
    assert!(!inf.is_normal());
    assert!(!neg_inf.is_normal());
    assert!(!zero.is_normal());
    assert!(!neg_zero.is_normal());
    assert!(1f32.is_normal());
    assert!(1e-37f32.is_normal());
    assert!(!1e-38f32.is_normal());
}

#[test]
fn test_classify() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    let zero: f32 = 0.0f32;
    let neg_zero: f32 = -0.0;
    assert_eq!(nan.classify(), Fp::Nan);
    assert_eq!(inf.classify(), Fp::Infinite);
    assert_eq!(neg_inf.classify(), Fp::Infinite);
    assert_eq!(zero.classify(), Fp::Zero);
    assert_eq!(neg_zero.classify(), Fp::Zero);
    assert_eq!(1f32.classify(), Fp::Normal);
    assert_eq!(1e-37f32.classify(), Fp::Normal);
    assert_eq!(1e-38f32.classify(), Fp::Subnormal);
}

#[test]
fn test_floor() {
    assert_approx_eq!(f32::floor(1.0f32), 1.0f32);
    assert_approx_eq!(f32::floor(1.3f32), 1.0f32);
    assert_approx_eq!(f32::floor(1.5f32), 1.0f32);
    assert_approx_eq!(f32::floor(1.7f32), 1.0f32);
    assert_approx_eq!(f32::floor(0.0f32), 0.0f32);
    assert_approx_eq!(f32::floor(-0.0f32), -0.0f32);
    assert_approx_eq!(f32::floor(-1.0f32), -1.0f32);
    assert_approx_eq!(f32::floor(-1.3f32), -2.0f32);
    assert_approx_eq!(f32::floor(-1.5f32), -2.0f32);
    assert_approx_eq!(f32::floor(-1.7f32), -2.0f32);
}

#[test]
fn test_ceil() {
    assert_approx_eq!(f32::ceil(1.0f32), 1.0f32);
    assert_approx_eq!(f32::ceil(1.3f32), 2.0f32);
    assert_approx_eq!(f32::ceil(1.5f32), 2.0f32);
    assert_approx_eq!(f32::ceil(1.7f32), 2.0f32);
    assert_approx_eq!(f32::ceil(0.0f32), 0.0f32);
    assert_approx_eq!(f32::ceil(-0.0f32), -0.0f32);
    assert_approx_eq!(f32::ceil(-1.0f32), -1.0f32);
    assert_approx_eq!(f32::ceil(-1.3f32), -1.0f32);
    assert_approx_eq!(f32::ceil(-1.5f32), -1.0f32);
    assert_approx_eq!(f32::ceil(-1.7f32), -1.0f32);
}

#[test]
fn test_round() {
    assert_approx_eq!(f32::round(2.5f32), 3.0f32);
    assert_approx_eq!(f32::round(1.0f32), 1.0f32);
    assert_approx_eq!(f32::round(1.3f32), 1.0f32);
    assert_approx_eq!(f32::round(1.5f32), 2.0f32);
    assert_approx_eq!(f32::round(1.7f32), 2.0f32);
    assert_approx_eq!(f32::round(0.0f32), 0.0f32);
    assert_approx_eq!(f32::round(-0.0f32), -0.0f32);
    assert_approx_eq!(f32::round(-1.0f32), -1.0f32);
    assert_approx_eq!(f32::round(-1.3f32), -1.0f32);
    assert_approx_eq!(f32::round(-1.5f32), -2.0f32);
    assert_approx_eq!(f32::round(-1.7f32), -2.0f32);
}

#[test]
fn test_round_ties_even() {
    assert_approx_eq!(f32::round_ties_even(2.5f32), 2.0f32);
    assert_approx_eq!(f32::round_ties_even(1.0f32), 1.0f32);
    assert_approx_eq!(f32::round_ties_even(1.3f32), 1.0f32);
    assert_approx_eq!(f32::round_ties_even(1.5f32), 2.0f32);
    assert_approx_eq!(f32::round_ties_even(1.7f32), 2.0f32);
    assert_approx_eq!(f32::round_ties_even(0.0f32), 0.0f32);
    assert_approx_eq!(f32::round_ties_even(-0.0f32), -0.0f32);
    assert_approx_eq!(f32::round_ties_even(-1.0f32), -1.0f32);
    assert_approx_eq!(f32::round_ties_even(-1.3f32), -1.0f32);
    assert_approx_eq!(f32::round_ties_even(-1.5f32), -2.0f32);
    assert_approx_eq!(f32::round_ties_even(-1.7f32), -2.0f32);
}

#[test]
fn test_trunc() {
    assert_approx_eq!(f32::trunc(1.0f32), 1.0f32);
    assert_approx_eq!(f32::trunc(1.3f32), 1.0f32);
    assert_approx_eq!(f32::trunc(1.5f32), 1.0f32);
    assert_approx_eq!(f32::trunc(1.7f32), 1.0f32);
    assert_approx_eq!(f32::trunc(0.0f32), 0.0f32);
    assert_approx_eq!(f32::trunc(-0.0f32), -0.0f32);
    assert_approx_eq!(f32::trunc(-1.0f32), -1.0f32);
    assert_approx_eq!(f32::trunc(-1.3f32), -1.0f32);
    assert_approx_eq!(f32::trunc(-1.5f32), -1.0f32);
    assert_approx_eq!(f32::trunc(-1.7f32), -1.0f32);
}

#[test]
fn test_fract() {
    assert_approx_eq!(f32::fract(1.0f32), 0.0f32);
    assert_approx_eq!(f32::fract(1.3f32), 0.3f32);
    assert_approx_eq!(f32::fract(1.5f32), 0.5f32);
    assert_approx_eq!(f32::fract(1.7f32), 0.7f32);
    assert_approx_eq!(f32::fract(0.0f32), 0.0f32);
    assert_approx_eq!(f32::fract(-0.0f32), -0.0f32);
    assert_approx_eq!(f32::fract(-1.0f32), -0.0f32);
    assert_approx_eq!(f32::fract(-1.3f32), -0.3f32);
    assert_approx_eq!(f32::fract(-1.5f32), -0.5f32);
    assert_approx_eq!(f32::fract(-1.7f32), -0.7f32);
}

#[test]
fn test_abs() {
    assert_eq!(f32::INFINITY.abs(), f32::INFINITY);
    assert_eq!(1f32.abs(), 1f32);
    assert_eq!(0f32.abs(), 0f32);
    assert_eq!((-0f32).abs(), 0f32);
    assert_eq!((-1f32).abs(), 1f32);
    assert_eq!(f32::NEG_INFINITY.abs(), f32::INFINITY);
    assert_eq!((1f32 / f32::NEG_INFINITY).abs(), 0f32);
    assert!(f32::NAN.abs().is_nan());
}

#[test]
fn test_signum() {
    assert_eq!(f32::INFINITY.signum(), 1f32);
    assert_eq!(1f32.signum(), 1f32);
    assert_eq!(0f32.signum(), 1f32);
    assert_eq!((-0f32).signum(), -1f32);
    assert_eq!((-1f32).signum(), -1f32);
    assert_eq!(f32::NEG_INFINITY.signum(), -1f32);
    assert_eq!((1f32 / f32::NEG_INFINITY).signum(), -1f32);
    assert!(f32::NAN.signum().is_nan());
}

#[test]
fn test_is_sign_positive() {
    assert!(f32::INFINITY.is_sign_positive());
    assert!(1f32.is_sign_positive());
    assert!(0f32.is_sign_positive());
    assert!(!(-0f32).is_sign_positive());
    assert!(!(-1f32).is_sign_positive());
    assert!(!f32::NEG_INFINITY.is_sign_positive());
    assert!(!(1f32 / f32::NEG_INFINITY).is_sign_positive());
    assert!(f32::NAN.is_sign_positive());
    assert!(!(-f32::NAN).is_sign_positive());
}

#[test]
fn test_is_sign_negative() {
    assert!(!f32::INFINITY.is_sign_negative());
    assert!(!1f32.is_sign_negative());
    assert!(!0f32.is_sign_negative());
    assert!((-0f32).is_sign_negative());
    assert!((-1f32).is_sign_negative());
    assert!(f32::NEG_INFINITY.is_sign_negative());
    assert!((1f32 / f32::NEG_INFINITY).is_sign_negative());
    assert!(!f32::NAN.is_sign_negative());
    assert!((-f32::NAN).is_sign_negative());
}

#[test]
fn test_next_up() {
    let tiny = f32::from_bits(TINY_BITS);
    let tiny_up = f32::from_bits(TINY_UP_BITS);
    let max_down = f32::from_bits(MAX_DOWN_BITS);
    let largest_subnormal = f32::from_bits(LARGEST_SUBNORMAL_BITS);
    let smallest_normal = f32::from_bits(SMALLEST_NORMAL_BITS);
    assert_f32_biteq!(f32::NEG_INFINITY.next_up(), f32::MIN);
    assert_f32_biteq!(f32::MIN.next_up(), -max_down);
    assert_f32_biteq!((-1.0 - f32::EPSILON).next_up(), -1.0);
    assert_f32_biteq!((-smallest_normal).next_up(), -largest_subnormal);
    assert_f32_biteq!((-tiny_up).next_up(), -tiny);
    assert_f32_biteq!((-tiny).next_up(), -0.0f32);
    assert_f32_biteq!((-0.0f32).next_up(), tiny);
    assert_f32_biteq!(0.0f32.next_up(), tiny);
    assert_f32_biteq!(tiny.next_up(), tiny_up);
    assert_f32_biteq!(largest_subnormal.next_up(), smallest_normal);
    assert_f32_biteq!(1.0f32.next_up(), 1.0 + f32::EPSILON);
    assert_f32_biteq!(f32::MAX.next_up(), f32::INFINITY);
    assert_f32_biteq!(f32::INFINITY.next_up(), f32::INFINITY);

    // Check that NaNs roundtrip.
    let nan0 = f32::NAN;
    let nan1 = f32::from_bits(f32::NAN.to_bits() ^ NAN_MASK1);
    let nan2 = f32::from_bits(f32::NAN.to_bits() ^ NAN_MASK2);
    assert_f32_biteq!(nan0.next_up(), nan0);
    assert_f32_biteq!(nan1.next_up(), nan1);
    assert_f32_biteq!(nan2.next_up(), nan2);
}

#[test]
fn test_next_down() {
    let tiny = f32::from_bits(TINY_BITS);
    let tiny_up = f32::from_bits(TINY_UP_BITS);
    let max_down = f32::from_bits(MAX_DOWN_BITS);
    let largest_subnormal = f32::from_bits(LARGEST_SUBNORMAL_BITS);
    let smallest_normal = f32::from_bits(SMALLEST_NORMAL_BITS);
    assert_f32_biteq!(f32::NEG_INFINITY.next_down(), f32::NEG_INFINITY);
    assert_f32_biteq!(f32::MIN.next_down(), f32::NEG_INFINITY);
    assert_f32_biteq!((-max_down).next_down(), f32::MIN);
    assert_f32_biteq!((-1.0f32).next_down(), -1.0 - f32::EPSILON);
    assert_f32_biteq!((-largest_subnormal).next_down(), -smallest_normal);
    assert_f32_biteq!((-tiny).next_down(), -tiny_up);
    assert_f32_biteq!((-0.0f32).next_down(), -tiny);
    assert_f32_biteq!((0.0f32).next_down(), -tiny);
    assert_f32_biteq!(tiny.next_down(), 0.0f32);
    assert_f32_biteq!(tiny_up.next_down(), tiny);
    assert_f32_biteq!(smallest_normal.next_down(), largest_subnormal);
    assert_f32_biteq!((1.0 + f32::EPSILON).next_down(), 1.0f32);
    assert_f32_biteq!(f32::MAX.next_down(), max_down);
    assert_f32_biteq!(f32::INFINITY.next_down(), f32::MAX);

    // Check that NaNs roundtrip.
    let nan0 = f32::NAN;
    let nan1 = f32::from_bits(f32::NAN.to_bits() ^ NAN_MASK1);
    let nan2 = f32::from_bits(f32::NAN.to_bits() ^ NAN_MASK2);
    assert_f32_biteq!(nan0.next_down(), nan0);
    assert_f32_biteq!(nan1.next_down(), nan1);
    assert_f32_biteq!(nan2.next_down(), nan2);
}

#[test]
fn test_mul_add() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_approx_eq!(f32::mul_add(12.3f32, 4.5, 6.7), 62.05);
    assert_approx_eq!(f32::mul_add(-12.3f32, -4.5, -6.7), 48.65);
    assert_approx_eq!(f32::mul_add(0.0f32, 8.9, 1.2), 1.2);
    assert_approx_eq!(f32::mul_add(3.4f32, -0.0, 5.6), 5.6);
    assert!(f32::mul_add(nan, 7.8, 9.0).is_nan());
    assert_eq!(f32::mul_add(inf, 7.8, 9.0), inf);
    assert_eq!(f32::mul_add(neg_inf, 7.8, 9.0), neg_inf);
    assert_eq!(f32::mul_add(8.9f32, inf, 3.2), inf);
    assert_eq!(f32::mul_add(-3.2f32, 2.4, neg_inf), neg_inf);
}

#[test]
fn test_recip() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_eq!(1.0f32.recip(), 1.0);
    assert_eq!(2.0f32.recip(), 0.5);
    assert_eq!((-0.4f32).recip(), -2.5);
    assert_eq!(0.0f32.recip(), inf);
    assert!(nan.recip().is_nan());
    assert_eq!(inf.recip(), 0.0);
    assert_eq!(neg_inf.recip(), 0.0);
}

#[test]
fn test_powi() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_eq!(1.0f32.powi(1), 1.0);
    assert_approx_eq!((-3.1f32).powi(2), 9.61);
    assert_approx_eq!(5.9f32.powi(-2), 0.028727);
    assert_eq!(8.3f32.powi(0), 1.0);
    assert!(nan.powi(2).is_nan());
    assert_eq!(inf.powi(3), inf);
    assert_eq!(neg_inf.powi(2), inf);
}

#[test]
fn test_sqrt_domain() {
    assert!(f32::NAN.sqrt().is_nan());
    assert!(f32::NEG_INFINITY.sqrt().is_nan());
    assert!((-1.0f32).sqrt().is_nan());
    assert_eq!((-0.0f32).sqrt(), -0.0);
    assert_eq!(0.0f32.sqrt(), 0.0);
    assert_eq!(1.0f32.sqrt(), 1.0);
    assert_eq!(f32::INFINITY.sqrt(), f32::INFINITY);
}

#[test]
fn test_to_degrees() {
    let pi: f32 = consts::PI;
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_eq!(0.0f32.to_degrees(), 0.0);
    assert_approx_eq!((-5.8f32).to_degrees(), -332.315521);
    assert_eq!(pi.to_degrees(), 180.0);
    assert!(nan.to_degrees().is_nan());
    assert_eq!(inf.to_degrees(), inf);
    assert_eq!(neg_inf.to_degrees(), neg_inf);
    assert_eq!(1_f32.to_degrees(), 57.2957795130823208767981548141051703);
}

#[test]
fn test_to_radians() {
    let pi: f32 = consts::PI;
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_eq!(0.0f32.to_radians(), 0.0);
    assert_approx_eq!(154.6f32.to_radians(), 2.698279);
    assert_approx_eq!((-332.31f32).to_radians(), -5.799903);
    assert_eq!(180.0f32.to_radians(), pi);
    assert!(nan.to_radians().is_nan());
    assert_eq!(inf.to_radians(), inf);
    assert_eq!(neg_inf.to_radians(), neg_inf);
}

#[test]
fn test_float_bits_conv() {
    assert_eq!((1f32).to_bits(), 0x3f800000);
    assert_eq!((12.5f32).to_bits(), 0x41480000);
    assert_eq!((1337f32).to_bits(), 0x44a72000);
    assert_eq!((-14.25f32).to_bits(), 0xc1640000);
    assert_approx_eq!(f32::from_bits(0x3f800000), 1.0);
    assert_approx_eq!(f32::from_bits(0x41480000), 12.5);
    assert_approx_eq!(f32::from_bits(0x44a72000), 1337.0);
    assert_approx_eq!(f32::from_bits(0xc1640000), -14.25);

    // Check that NaNs roundtrip their bits regardless of signaling-ness
    // 0xA is 0b1010; 0x5 is 0b0101 -- so these two together clobbers all the mantissa bits
    let masked_nan1 = f32::NAN.to_bits() ^ NAN_MASK1;
    let masked_nan2 = f32::NAN.to_bits() ^ NAN_MASK2;
    assert!(f32::from_bits(masked_nan1).is_nan());
    assert!(f32::from_bits(masked_nan2).is_nan());

    assert_eq!(f32::from_bits(masked_nan1).to_bits(), masked_nan1);
    assert_eq!(f32::from_bits(masked_nan2).to_bits(), masked_nan2);
}

#[test]
#[should_panic]
fn test_clamp_min_greater_than_max() {
    let _ = 1.0f32.clamp(3.0, 1.0);
}

#[test]
#[should_panic]
fn test_clamp_min_is_nan() {
    let _ = 1.0f32.clamp(f32::NAN, 1.0);
}

#[test]
#[should_panic]
fn test_clamp_max_is_nan() {
    let _ = 1.0f32.clamp(3.0, f32::NAN);
}

#[test]
fn test_total_cmp() {
    use core::cmp::Ordering;

    fn quiet_bit_mask() -> u32 {
        1 << (f32::MANTISSA_DIGITS - 2)
    }

    fn min_subnorm() -> f32 {
        f32::MIN_POSITIVE / f32::powf(2.0, f32::MANTISSA_DIGITS as f32 - 1.0)
    }

    fn max_subnorm() -> f32 {
        f32::MIN_POSITIVE - min_subnorm()
    }

    fn q_nan() -> f32 {
        f32::from_bits(f32::NAN.to_bits() | quiet_bit_mask())
    }

    fn s_nan() -> f32 {
        f32::from_bits((f32::NAN.to_bits() & !quiet_bit_mask()) + 42)
    }

    assert_eq!(Ordering::Equal, (-q_nan()).total_cmp(&-q_nan()));
    assert_eq!(Ordering::Equal, (-s_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Equal, (-f32::INFINITY).total_cmp(&-f32::INFINITY));
    assert_eq!(Ordering::Equal, (-f32::MAX).total_cmp(&-f32::MAX));
    assert_eq!(Ordering::Equal, (-2.5_f32).total_cmp(&-2.5));
    assert_eq!(Ordering::Equal, (-1.0_f32).total_cmp(&-1.0));
    assert_eq!(Ordering::Equal, (-1.5_f32).total_cmp(&-1.5));
    assert_eq!(Ordering::Equal, (-0.5_f32).total_cmp(&-0.5));
    assert_eq!(Ordering::Equal, (-f32::MIN_POSITIVE).total_cmp(&-f32::MIN_POSITIVE));
    assert_eq!(Ordering::Equal, (-max_subnorm()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Equal, (-min_subnorm()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Equal, (-0.0_f32).total_cmp(&-0.0));
    assert_eq!(Ordering::Equal, 0.0_f32.total_cmp(&0.0));
    assert_eq!(Ordering::Equal, min_subnorm().total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Equal, max_subnorm().total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Equal, f32::MIN_POSITIVE.total_cmp(&f32::MIN_POSITIVE));
    assert_eq!(Ordering::Equal, 0.5_f32.total_cmp(&0.5));
    assert_eq!(Ordering::Equal, 1.0_f32.total_cmp(&1.0));
    assert_eq!(Ordering::Equal, 1.5_f32.total_cmp(&1.5));
    assert_eq!(Ordering::Equal, 2.5_f32.total_cmp(&2.5));
    assert_eq!(Ordering::Equal, f32::MAX.total_cmp(&f32::MAX));
    assert_eq!(Ordering::Equal, f32::INFINITY.total_cmp(&f32::INFINITY));
    assert_eq!(Ordering::Equal, s_nan().total_cmp(&s_nan()));
    assert_eq!(Ordering::Equal, q_nan().total_cmp(&q_nan()));

    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f32::INFINITY));
    assert_eq!(Ordering::Less, (-f32::INFINITY).total_cmp(&-f32::MAX));
    assert_eq!(Ordering::Less, (-f32::MAX).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-2.5_f32).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-1.5_f32).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-1.0_f32).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-0.5_f32).total_cmp(&-f32::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-f32::MIN_POSITIVE).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Less, (-max_subnorm()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-min_subnorm()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-0.0_f32).total_cmp(&0.0));
    assert_eq!(Ordering::Less, 0.0_f32.total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Less, min_subnorm().total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, max_subnorm().total_cmp(&f32::MIN_POSITIVE));
    assert_eq!(Ordering::Less, f32::MIN_POSITIVE.total_cmp(&0.5));
    assert_eq!(Ordering::Less, 0.5_f32.total_cmp(&1.0));
    assert_eq!(Ordering::Less, 1.0_f32.total_cmp(&1.5));
    assert_eq!(Ordering::Less, 1.5_f32.total_cmp(&2.5));
    assert_eq!(Ordering::Less, 2.5_f32.total_cmp(&f32::MAX));
    assert_eq!(Ordering::Less, f32::MAX.total_cmp(&f32::INFINITY));
    assert_eq!(Ordering::Less, f32::INFINITY.total_cmp(&s_nan()));
    assert_eq!(Ordering::Less, s_nan().total_cmp(&q_nan()));

    assert_eq!(Ordering::Greater, (-s_nan()).total_cmp(&-q_nan()));
    assert_eq!(Ordering::Greater, (-f32::INFINITY).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Greater, (-f32::MAX).total_cmp(&-f32::INFINITY));
    assert_eq!(Ordering::Greater, (-2.5_f32).total_cmp(&-f32::MAX));
    assert_eq!(Ordering::Greater, (-1.5_f32).total_cmp(&-2.5));
    assert_eq!(Ordering::Greater, (-1.0_f32).total_cmp(&-1.5));
    assert_eq!(Ordering::Greater, (-0.5_f32).total_cmp(&-1.0));
    assert_eq!(Ordering::Greater, (-f32::MIN_POSITIVE).total_cmp(&-0.5));
    assert_eq!(Ordering::Greater, (-max_subnorm()).total_cmp(&-f32::MIN_POSITIVE));
    assert_eq!(Ordering::Greater, (-min_subnorm()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Greater, (-0.0_f32).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Greater, 0.0_f32.total_cmp(&-0.0));
    assert_eq!(Ordering::Greater, min_subnorm().total_cmp(&0.0));
    assert_eq!(Ordering::Greater, max_subnorm().total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Greater, f32::MIN_POSITIVE.total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Greater, 0.5_f32.total_cmp(&f32::MIN_POSITIVE));
    assert_eq!(Ordering::Greater, 1.0_f32.total_cmp(&0.5));
    assert_eq!(Ordering::Greater, 1.5_f32.total_cmp(&1.0));
    assert_eq!(Ordering::Greater, 2.5_f32.total_cmp(&1.5));
    assert_eq!(Ordering::Greater, f32::MAX.total_cmp(&2.5));
    assert_eq!(Ordering::Greater, f32::INFINITY.total_cmp(&f32::MAX));
    assert_eq!(Ordering::Greater, s_nan().total_cmp(&f32::INFINITY));
    assert_eq!(Ordering::Greater, q_nan().total_cmp(&s_nan()));

    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f32::INFINITY));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f32::MAX));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f32::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&0.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f32::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&0.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&1.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&1.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&2.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f32::MAX));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f32::INFINITY));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&s_nan()));

    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f32::INFINITY));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f32::MAX));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f32::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&0.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f32::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&0.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&1.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&1.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&2.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f32::MAX));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f32::INFINITY));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&s_nan()));
}

#[test]
fn test_algebraic() {
    let a: f32 = 123.0;
    let b: f32 = 456.0;

    // Check that individual operations match their primitive counterparts.
    //
    // This is a check of current implementations and does NOT imply any form of
    // guarantee about future behavior. The compiler reserves the right to make
    // these operations inexact matches in the future.
    let eps_add = if cfg!(miri) { 1e-3 } else { 0.0 };
    let eps_mul = if cfg!(miri) { 1e-1 } else { 0.0 };
    let eps_div = if cfg!(miri) { 1e-4 } else { 0.0 };

    assert_approx_eq!(a.algebraic_add(b), a + b, eps_add);
    assert_approx_eq!(a.algebraic_sub(b), a - b, eps_add);
    assert_approx_eq!(a.algebraic_mul(b), a * b, eps_mul);
    assert_approx_eq!(a.algebraic_div(b), a / b, eps_div);
    assert_approx_eq!(a.algebraic_rem(b), a % b, eps_div);
}
