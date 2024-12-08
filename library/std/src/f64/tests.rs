use crate::f64::consts;
use crate::num::{FpCategory as Fp, *};

/// Smallest number
const TINY_BITS: u64 = 0x1;

/// Next smallest number
const TINY_UP_BITS: u64 = 0x2;

/// Exponent = 0b11...10, Sifnificand 0b1111..10. Min val > 0
const MAX_DOWN_BITS: u64 = 0x7fef_ffff_ffff_fffe;

/// Zeroed exponent, full significant
const LARGEST_SUBNORMAL_BITS: u64 = 0x000f_ffff_ffff_ffff;

/// Exponent = 0b1, zeroed significand
const SMALLEST_NORMAL_BITS: u64 = 0x0010_0000_0000_0000;

/// First pattern over the mantissa
const NAN_MASK1: u64 = 0x000a_aaaa_aaaa_aaaa;

/// Second pattern over the mantissa
const NAN_MASK2: u64 = 0x0005_5555_5555_5555;

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
fn test_num_f64() {
    test_num(10f64, 2f64);
}

#[test]
fn test_min_nan() {
    assert_eq!(f64::NAN.min(2.0), 2.0);
    assert_eq!(2.0f64.min(f64::NAN), 2.0);
}

#[test]
fn test_max_nan() {
    assert_eq!(f64::NAN.max(2.0), 2.0);
    assert_eq!(2.0f64.max(f64::NAN), 2.0);
}

#[test]
fn test_nan() {
    let nan: f64 = f64::NAN;
    assert!(nan.is_nan());
    assert!(!nan.is_infinite());
    assert!(!nan.is_finite());
    assert!(!nan.is_normal());
    assert!(nan.is_sign_positive());
    assert!(!nan.is_sign_negative());
    assert_eq!(Fp::Nan, nan.classify());
}

#[test]
fn test_infinity() {
    let inf: f64 = f64::INFINITY;
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
    let neg_inf: f64 = f64::NEG_INFINITY;
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
    let zero: f64 = 0.0f64;
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
    let neg_zero: f64 = -0.0;
    assert_eq!(0.0, neg_zero);
    assert!(!neg_zero.is_infinite());
    assert!(neg_zero.is_finite());
    assert!(!neg_zero.is_sign_positive());
    assert!(neg_zero.is_sign_negative());
    assert!(!neg_zero.is_nan());
    assert!(!neg_zero.is_normal());
    assert_eq!(Fp::Zero, neg_zero.classify());
}

#[cfg_attr(all(target_arch = "wasm32", target_os = "emscripten"), ignore)] // issue 42630
#[test]
fn test_one() {
    let one: f64 = 1.0f64;
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
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert!(nan.is_nan());
    assert!(!0.0f64.is_nan());
    assert!(!5.3f64.is_nan());
    assert!(!(-10.732f64).is_nan());
    assert!(!inf.is_nan());
    assert!(!neg_inf.is_nan());
}

#[test]
fn test_is_infinite() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert!(!nan.is_infinite());
    assert!(inf.is_infinite());
    assert!(neg_inf.is_infinite());
    assert!(!0.0f64.is_infinite());
    assert!(!42.8f64.is_infinite());
    assert!(!(-109.2f64).is_infinite());
}

#[test]
fn test_is_finite() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert!(!nan.is_finite());
    assert!(!inf.is_finite());
    assert!(!neg_inf.is_finite());
    assert!(0.0f64.is_finite());
    assert!(42.8f64.is_finite());
    assert!((-109.2f64).is_finite());
}

#[cfg_attr(all(target_arch = "wasm32", target_os = "emscripten"), ignore)] // issue 42630
#[test]
fn test_is_normal() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    let zero: f64 = 0.0f64;
    let neg_zero: f64 = -0.0;
    assert!(!nan.is_normal());
    assert!(!inf.is_normal());
    assert!(!neg_inf.is_normal());
    assert!(!zero.is_normal());
    assert!(!neg_zero.is_normal());
    assert!(1f64.is_normal());
    assert!(1e-307f64.is_normal());
    assert!(!1e-308f64.is_normal());
}

#[cfg_attr(all(target_arch = "wasm32", target_os = "emscripten"), ignore)] // issue 42630
#[test]
fn test_classify() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    let zero: f64 = 0.0f64;
    let neg_zero: f64 = -0.0;
    assert_eq!(nan.classify(), Fp::Nan);
    assert_eq!(inf.classify(), Fp::Infinite);
    assert_eq!(neg_inf.classify(), Fp::Infinite);
    assert_eq!(zero.classify(), Fp::Zero);
    assert_eq!(neg_zero.classify(), Fp::Zero);
    assert_eq!(1e-307f64.classify(), Fp::Normal);
    assert_eq!(1e-308f64.classify(), Fp::Subnormal);
}

#[test]
fn test_floor() {
    assert_approx_eq!(1.0f64.floor(), 1.0f64);
    assert_approx_eq!(1.3f64.floor(), 1.0f64);
    assert_approx_eq!(1.5f64.floor(), 1.0f64);
    assert_approx_eq!(1.7f64.floor(), 1.0f64);
    assert_approx_eq!(0.0f64.floor(), 0.0f64);
    assert_approx_eq!((-0.0f64).floor(), -0.0f64);
    assert_approx_eq!((-1.0f64).floor(), -1.0f64);
    assert_approx_eq!((-1.3f64).floor(), -2.0f64);
    assert_approx_eq!((-1.5f64).floor(), -2.0f64);
    assert_approx_eq!((-1.7f64).floor(), -2.0f64);
}

#[test]
fn test_ceil() {
    assert_approx_eq!(1.0f64.ceil(), 1.0f64);
    assert_approx_eq!(1.3f64.ceil(), 2.0f64);
    assert_approx_eq!(1.5f64.ceil(), 2.0f64);
    assert_approx_eq!(1.7f64.ceil(), 2.0f64);
    assert_approx_eq!(0.0f64.ceil(), 0.0f64);
    assert_approx_eq!((-0.0f64).ceil(), -0.0f64);
    assert_approx_eq!((-1.0f64).ceil(), -1.0f64);
    assert_approx_eq!((-1.3f64).ceil(), -1.0f64);
    assert_approx_eq!((-1.5f64).ceil(), -1.0f64);
    assert_approx_eq!((-1.7f64).ceil(), -1.0f64);
}

#[test]
fn test_round() {
    assert_approx_eq!(2.5f64.round(), 3.0f64);
    assert_approx_eq!(1.0f64.round(), 1.0f64);
    assert_approx_eq!(1.3f64.round(), 1.0f64);
    assert_approx_eq!(1.5f64.round(), 2.0f64);
    assert_approx_eq!(1.7f64.round(), 2.0f64);
    assert_approx_eq!(0.0f64.round(), 0.0f64);
    assert_approx_eq!((-0.0f64).round(), -0.0f64);
    assert_approx_eq!((-1.0f64).round(), -1.0f64);
    assert_approx_eq!((-1.3f64).round(), -1.0f64);
    assert_approx_eq!((-1.5f64).round(), -2.0f64);
    assert_approx_eq!((-1.7f64).round(), -2.0f64);
}

#[test]
fn test_round_ties_even() {
    assert_approx_eq!(2.5f64.round_ties_even(), 2.0f64);
    assert_approx_eq!(1.0f64.round_ties_even(), 1.0f64);
    assert_approx_eq!(1.3f64.round_ties_even(), 1.0f64);
    assert_approx_eq!(1.5f64.round_ties_even(), 2.0f64);
    assert_approx_eq!(1.7f64.round_ties_even(), 2.0f64);
    assert_approx_eq!(0.0f64.round_ties_even(), 0.0f64);
    assert_approx_eq!((-0.0f64).round_ties_even(), -0.0f64);
    assert_approx_eq!((-1.0f64).round_ties_even(), -1.0f64);
    assert_approx_eq!((-1.3f64).round_ties_even(), -1.0f64);
    assert_approx_eq!((-1.5f64).round_ties_even(), -2.0f64);
    assert_approx_eq!((-1.7f64).round_ties_even(), -2.0f64);
}

#[test]
fn test_trunc() {
    assert_approx_eq!(1.0f64.trunc(), 1.0f64);
    assert_approx_eq!(1.3f64.trunc(), 1.0f64);
    assert_approx_eq!(1.5f64.trunc(), 1.0f64);
    assert_approx_eq!(1.7f64.trunc(), 1.0f64);
    assert_approx_eq!(0.0f64.trunc(), 0.0f64);
    assert_approx_eq!((-0.0f64).trunc(), -0.0f64);
    assert_approx_eq!((-1.0f64).trunc(), -1.0f64);
    assert_approx_eq!((-1.3f64).trunc(), -1.0f64);
    assert_approx_eq!((-1.5f64).trunc(), -1.0f64);
    assert_approx_eq!((-1.7f64).trunc(), -1.0f64);
}

#[test]
fn test_fract() {
    assert_approx_eq!(1.0f64.fract(), 0.0f64);
    assert_approx_eq!(1.3f64.fract(), 0.3f64);
    assert_approx_eq!(1.5f64.fract(), 0.5f64);
    assert_approx_eq!(1.7f64.fract(), 0.7f64);
    assert_approx_eq!(0.0f64.fract(), 0.0f64);
    assert_approx_eq!((-0.0f64).fract(), -0.0f64);
    assert_approx_eq!((-1.0f64).fract(), -0.0f64);
    assert_approx_eq!((-1.3f64).fract(), -0.3f64);
    assert_approx_eq!((-1.5f64).fract(), -0.5f64);
    assert_approx_eq!((-1.7f64).fract(), -0.7f64);
}

#[test]
fn test_abs() {
    assert_eq!(f64::INFINITY.abs(), f64::INFINITY);
    assert_eq!(1f64.abs(), 1f64);
    assert_eq!(0f64.abs(), 0f64);
    assert_eq!((-0f64).abs(), 0f64);
    assert_eq!((-1f64).abs(), 1f64);
    assert_eq!(f64::NEG_INFINITY.abs(), f64::INFINITY);
    assert_eq!((1f64 / f64::NEG_INFINITY).abs(), 0f64);
    assert!(f64::NAN.abs().is_nan());
}

#[test]
fn test_signum() {
    assert_eq!(f64::INFINITY.signum(), 1f64);
    assert_eq!(1f64.signum(), 1f64);
    assert_eq!(0f64.signum(), 1f64);
    assert_eq!((-0f64).signum(), -1f64);
    assert_eq!((-1f64).signum(), -1f64);
    assert_eq!(f64::NEG_INFINITY.signum(), -1f64);
    assert_eq!((1f64 / f64::NEG_INFINITY).signum(), -1f64);
    assert!(f64::NAN.signum().is_nan());
}

#[test]
fn test_is_sign_positive() {
    assert!(f64::INFINITY.is_sign_positive());
    assert!(1f64.is_sign_positive());
    assert!(0f64.is_sign_positive());
    assert!(!(-0f64).is_sign_positive());
    assert!(!(-1f64).is_sign_positive());
    assert!(!f64::NEG_INFINITY.is_sign_positive());
    assert!(!(1f64 / f64::NEG_INFINITY).is_sign_positive());
    assert!(f64::NAN.is_sign_positive());
    assert!(!(-f64::NAN).is_sign_positive());
}

#[test]
fn test_is_sign_negative() {
    assert!(!f64::INFINITY.is_sign_negative());
    assert!(!1f64.is_sign_negative());
    assert!(!0f64.is_sign_negative());
    assert!((-0f64).is_sign_negative());
    assert!((-1f64).is_sign_negative());
    assert!(f64::NEG_INFINITY.is_sign_negative());
    assert!((1f64 / f64::NEG_INFINITY).is_sign_negative());
    assert!(!f64::NAN.is_sign_negative());
    assert!((-f64::NAN).is_sign_negative());
}

#[test]
fn test_next_up() {
    let tiny = f64::from_bits(TINY_BITS);
    let tiny_up = f64::from_bits(TINY_UP_BITS);
    let max_down = f64::from_bits(MAX_DOWN_BITS);
    let largest_subnormal = f64::from_bits(LARGEST_SUBNORMAL_BITS);
    let smallest_normal = f64::from_bits(SMALLEST_NORMAL_BITS);
    assert_f64_biteq!(f64::NEG_INFINITY.next_up(), f64::MIN);
    assert_f64_biteq!(f64::MIN.next_up(), -max_down);
    assert_f64_biteq!((-1.0 - f64::EPSILON).next_up(), -1.0);
    assert_f64_biteq!((-smallest_normal).next_up(), -largest_subnormal);
    assert_f64_biteq!((-tiny_up).next_up(), -tiny);
    assert_f64_biteq!((-tiny).next_up(), -0.0f64);
    assert_f64_biteq!((-0.0f64).next_up(), tiny);
    assert_f64_biteq!(0.0f64.next_up(), tiny);
    assert_f64_biteq!(tiny.next_up(), tiny_up);
    assert_f64_biteq!(largest_subnormal.next_up(), smallest_normal);
    assert_f64_biteq!(1.0f64.next_up(), 1.0 + f64::EPSILON);
    assert_f64_biteq!(f64::MAX.next_up(), f64::INFINITY);
    assert_f64_biteq!(f64::INFINITY.next_up(), f64::INFINITY);

    let nan0 = f64::NAN;
    let nan1 = f64::from_bits(f64::NAN.to_bits() ^ NAN_MASK1);
    let nan2 = f64::from_bits(f64::NAN.to_bits() ^ NAN_MASK2);
    assert_f64_biteq!(nan0.next_up(), nan0);
    assert_f64_biteq!(nan1.next_up(), nan1);
    assert_f64_biteq!(nan2.next_up(), nan2);
}

#[test]
fn test_next_down() {
    let tiny = f64::from_bits(TINY_BITS);
    let tiny_up = f64::from_bits(TINY_UP_BITS);
    let max_down = f64::from_bits(MAX_DOWN_BITS);
    let largest_subnormal = f64::from_bits(LARGEST_SUBNORMAL_BITS);
    let smallest_normal = f64::from_bits(SMALLEST_NORMAL_BITS);
    assert_f64_biteq!(f64::NEG_INFINITY.next_down(), f64::NEG_INFINITY);
    assert_f64_biteq!(f64::MIN.next_down(), f64::NEG_INFINITY);
    assert_f64_biteq!((-max_down).next_down(), f64::MIN);
    assert_f64_biteq!((-1.0f64).next_down(), -1.0 - f64::EPSILON);
    assert_f64_biteq!((-largest_subnormal).next_down(), -smallest_normal);
    assert_f64_biteq!((-tiny).next_down(), -tiny_up);
    assert_f64_biteq!((-0.0f64).next_down(), -tiny);
    assert_f64_biteq!((0.0f64).next_down(), -tiny);
    assert_f64_biteq!(tiny.next_down(), 0.0f64);
    assert_f64_biteq!(tiny_up.next_down(), tiny);
    assert_f64_biteq!(smallest_normal.next_down(), largest_subnormal);
    assert_f64_biteq!((1.0 + f64::EPSILON).next_down(), 1.0f64);
    assert_f64_biteq!(f64::MAX.next_down(), max_down);
    assert_f64_biteq!(f64::INFINITY.next_down(), f64::MAX);

    let nan0 = f64::NAN;
    let nan1 = f64::from_bits(f64::NAN.to_bits() ^ NAN_MASK1);
    let nan2 = f64::from_bits(f64::NAN.to_bits() ^ NAN_MASK2);
    assert_f64_biteq!(nan0.next_down(), nan0);
    assert_f64_biteq!(nan1.next_down(), nan1);
    assert_f64_biteq!(nan2.next_down(), nan2);
}

#[test]
fn test_mul_add() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_approx_eq!(12.3f64.mul_add(4.5, 6.7), 62.05);
    assert_approx_eq!((-12.3f64).mul_add(-4.5, -6.7), 48.65);
    assert_approx_eq!(0.0f64.mul_add(8.9, 1.2), 1.2);
    assert_approx_eq!(3.4f64.mul_add(-0.0, 5.6), 5.6);
    assert!(nan.mul_add(7.8, 9.0).is_nan());
    assert_eq!(inf.mul_add(7.8, 9.0), inf);
    assert_eq!(neg_inf.mul_add(7.8, 9.0), neg_inf);
    assert_eq!(8.9f64.mul_add(inf, 3.2), inf);
    assert_eq!((-3.2f64).mul_add(2.4, neg_inf), neg_inf);
}

#[test]
fn test_recip() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_eq!(1.0f64.recip(), 1.0);
    assert_eq!(2.0f64.recip(), 0.5);
    assert_eq!((-0.4f64).recip(), -2.5);
    assert_eq!(0.0f64.recip(), inf);
    assert!(nan.recip().is_nan());
    assert_eq!(inf.recip(), 0.0);
    assert_eq!(neg_inf.recip(), 0.0);
}

#[test]
fn test_powi() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_eq!(1.0f64.powi(1), 1.0);
    assert_approx_eq!((-3.1f64).powi(2), 9.61);
    assert_approx_eq!(5.9f64.powi(-2), 0.028727);
    assert_eq!(8.3f64.powi(0), 1.0);
    assert!(nan.powi(2).is_nan());
    assert_eq!(inf.powi(3), inf);
    assert_eq!(neg_inf.powi(2), inf);
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
fn test_sqrt_domain() {
    assert!(f64::NAN.sqrt().is_nan());
    assert!(f64::NEG_INFINITY.sqrt().is_nan());
    assert!((-1.0f64).sqrt().is_nan());
    assert_eq!((-0.0f64).sqrt(), -0.0);
    assert_eq!(0.0f64.sqrt(), 0.0);
    assert_eq!(1.0f64.sqrt(), 1.0);
    assert_eq!(f64::INFINITY.sqrt(), f64::INFINITY);
}

#[test]
fn test_exp() {
    assert_eq!(1.0, 0.0f64.exp());
    assert_approx_eq!(2.718282, 1.0f64.exp());
    assert_approx_eq!(148.413159, 5.0f64.exp());

    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    let nan: f64 = f64::NAN;
    assert_eq!(inf, inf.exp());
    assert_eq!(0.0, neg_inf.exp());
    assert!(nan.exp().is_nan());
}

#[test]
fn test_exp2() {
    assert_eq!(32.0, 5.0f64.exp2());
    assert_eq!(1.0, 0.0f64.exp2());

    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    let nan: f64 = f64::NAN;
    assert_eq!(inf, inf.exp2());
    assert_eq!(0.0, neg_inf.exp2());
    assert!(nan.exp2().is_nan());
}

#[test]
fn test_ln() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_approx_eq!(1.0f64.exp().ln(), 1.0);
    assert!(nan.ln().is_nan());
    assert_eq!(inf.ln(), inf);
    assert!(neg_inf.ln().is_nan());
    assert!((-2.3f64).ln().is_nan());
    assert_eq!((-0.0f64).ln(), neg_inf);
    assert_eq!(0.0f64.ln(), neg_inf);
    assert_approx_eq!(4.0f64.ln(), 1.386294);
}

#[test]
fn test_log() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_eq!(10.0f64.log(10.0), 1.0);
    assert_approx_eq!(2.3f64.log(3.5), 0.664858);
    assert_eq!(1.0f64.exp().log(1.0f64.exp()), 1.0);
    assert!(1.0f64.log(1.0).is_nan());
    assert!(1.0f64.log(-13.9).is_nan());
    assert!(nan.log(2.3).is_nan());
    assert_eq!(inf.log(10.0), inf);
    assert!(neg_inf.log(8.8).is_nan());
    assert!((-2.3f64).log(0.1).is_nan());
    assert_eq!((-0.0f64).log(2.0), neg_inf);
    assert_eq!(0.0f64.log(7.0), neg_inf);
}

#[test]
fn test_log2() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_approx_eq!(10.0f64.log2(), 3.321928);
    assert_approx_eq!(2.3f64.log2(), 1.201634);
    assert_approx_eq!(1.0f64.exp().log2(), 1.442695);
    assert!(nan.log2().is_nan());
    assert_eq!(inf.log2(), inf);
    assert!(neg_inf.log2().is_nan());
    assert!((-2.3f64).log2().is_nan());
    assert_eq!((-0.0f64).log2(), neg_inf);
    assert_eq!(0.0f64.log2(), neg_inf);
}

#[test]
fn test_log10() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_eq!(10.0f64.log10(), 1.0);
    assert_approx_eq!(2.3f64.log10(), 0.361728);
    assert_approx_eq!(1.0f64.exp().log10(), 0.434294);
    assert_eq!(1.0f64.log10(), 0.0);
    assert!(nan.log10().is_nan());
    assert_eq!(inf.log10(), inf);
    assert!(neg_inf.log10().is_nan());
    assert!((-2.3f64).log10().is_nan());
    assert_eq!((-0.0f64).log10(), neg_inf);
    assert_eq!(0.0f64.log10(), neg_inf);
}

#[test]
fn test_to_degrees() {
    let pi: f64 = consts::PI;
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_eq!(0.0f64.to_degrees(), 0.0);
    assert_approx_eq!((-5.8f64).to_degrees(), -332.315521);
    assert_eq!(pi.to_degrees(), 180.0);
    assert!(nan.to_degrees().is_nan());
    assert_eq!(inf.to_degrees(), inf);
    assert_eq!(neg_inf.to_degrees(), neg_inf);
}

#[test]
fn test_to_radians() {
    let pi: f64 = consts::PI;
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_eq!(0.0f64.to_radians(), 0.0);
    assert_approx_eq!(154.6f64.to_radians(), 2.698279);
    assert_approx_eq!((-332.31f64).to_radians(), -5.799903);
    assert_eq!(180.0f64.to_radians(), pi);
    assert!(nan.to_radians().is_nan());
    assert_eq!(inf.to_radians(), inf);
    assert_eq!(neg_inf.to_radians(), neg_inf);
}

#[test]
fn test_asinh() {
    assert_eq!(0.0f64.asinh(), 0.0f64);
    assert_eq!((-0.0f64).asinh(), -0.0f64);

    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    let nan: f64 = f64::NAN;
    assert_eq!(inf.asinh(), inf);
    assert_eq!(neg_inf.asinh(), neg_inf);
    assert!(nan.asinh().is_nan());
    assert!((-0.0f64).asinh().is_sign_negative());
    // issue 63271
    assert_approx_eq!(2.0f64.asinh(), 1.443635475178810342493276740273105f64);
    assert_approx_eq!((-2.0f64).asinh(), -1.443635475178810342493276740273105f64);
    // regression test for the catastrophic cancellation fixed in 72486
    assert_approx_eq!((-67452098.07139316f64).asinh(), -18.72007542627454439398548429400083);

    // test for low accuracy from issue 104548
    assert_approx_eq!(60.0f64, 60.0f64.sinh().asinh());
    // mul needed for approximate comparison to be meaningful
    assert_approx_eq!(1.0f64, 1e-15f64.sinh().asinh() * 1e15f64);
}

#[test]
fn test_acosh() {
    assert_eq!(1.0f64.acosh(), 0.0f64);
    assert!(0.999f64.acosh().is_nan());

    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    let nan: f64 = f64::NAN;
    assert_eq!(inf.acosh(), inf);
    assert!(neg_inf.acosh().is_nan());
    assert!(nan.acosh().is_nan());
    assert_approx_eq!(2.0f64.acosh(), 1.31695789692481670862504634730796844f64);
    assert_approx_eq!(3.0f64.acosh(), 1.76274717403908605046521864995958461f64);

    // test for low accuracy from issue 104548
    assert_approx_eq!(60.0f64, 60.0f64.cosh().acosh());
}

#[test]
fn test_atanh() {
    assert_eq!(0.0f64.atanh(), 0.0f64);
    assert_eq!((-0.0f64).atanh(), -0.0f64);

    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    let nan: f64 = f64::NAN;
    assert_eq!(1.0f64.atanh(), inf);
    assert_eq!((-1.0f64).atanh(), neg_inf);
    assert!(2f64.atanh().atanh().is_nan());
    assert!((-2f64).atanh().atanh().is_nan());
    assert!(inf.atanh().is_nan());
    assert!(neg_inf.atanh().is_nan());
    assert!(nan.atanh().is_nan());
    assert_approx_eq!(0.5f64.atanh(), 0.54930614433405484569762261846126285f64);
    assert_approx_eq!((-0.5f64).atanh(), -0.54930614433405484569762261846126285f64);
}

#[test]
fn test_gamma() {
    // precision can differ between platforms
    assert_approx_eq!(1.0f64.gamma(), 1.0f64);
    assert_approx_eq!(2.0f64.gamma(), 1.0f64);
    assert_approx_eq!(3.0f64.gamma(), 2.0f64);
    assert_approx_eq!(4.0f64.gamma(), 6.0f64);
    assert_approx_eq!(5.0f64.gamma(), 24.0f64);
    assert_approx_eq!(0.5f64.gamma(), consts::PI.sqrt());
    assert_approx_eq!((-0.5f64).gamma(), -2.0 * consts::PI.sqrt());
    assert_eq!(0.0f64.gamma(), f64::INFINITY);
    assert_eq!((-0.0f64).gamma(), f64::NEG_INFINITY);
    assert!((-1.0f64).gamma().is_nan());
    assert!((-2.0f64).gamma().is_nan());
    assert!(f64::NAN.gamma().is_nan());
    assert!(f64::NEG_INFINITY.gamma().is_nan());
    assert_eq!(f64::INFINITY.gamma(), f64::INFINITY);
    assert_eq!(171.71f64.gamma(), f64::INFINITY);
}

#[test]
fn test_ln_gamma() {
    assert_approx_eq!(1.0f64.ln_gamma().0, 0.0f64);
    assert_eq!(1.0f64.ln_gamma().1, 1);
    assert_approx_eq!(2.0f64.ln_gamma().0, 0.0f64);
    assert_eq!(2.0f64.ln_gamma().1, 1);
    assert_approx_eq!(3.0f64.ln_gamma().0, 2.0f64.ln());
    assert_eq!(3.0f64.ln_gamma().1, 1);
    assert_approx_eq!((-0.5f64).ln_gamma().0, (2.0 * consts::PI.sqrt()).ln());
    assert_eq!((-0.5f64).ln_gamma().1, -1);
}

#[test]
fn test_real_consts() {
    use super::consts;
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

#[test]
fn test_float_bits_conv() {
    assert_eq!((1f64).to_bits(), 0x3ff0000000000000);
    assert_eq!((12.5f64).to_bits(), 0x4029000000000000);
    assert_eq!((1337f64).to_bits(), 0x4094e40000000000);
    assert_eq!((-14.25f64).to_bits(), 0xc02c800000000000);
    assert_approx_eq!(f64::from_bits(0x3ff0000000000000), 1.0);
    assert_approx_eq!(f64::from_bits(0x4029000000000000), 12.5);
    assert_approx_eq!(f64::from_bits(0x4094e40000000000), 1337.0);
    assert_approx_eq!(f64::from_bits(0xc02c800000000000), -14.25);

    // Check that NaNs roundtrip their bits regardless of signaling-ness
    let masked_nan1 = f64::NAN.to_bits() ^ NAN_MASK1;
    let masked_nan2 = f64::NAN.to_bits() ^ NAN_MASK2;
    assert!(f64::from_bits(masked_nan1).is_nan());
    assert!(f64::from_bits(masked_nan2).is_nan());

    assert_eq!(f64::from_bits(masked_nan1).to_bits(), masked_nan1);
    assert_eq!(f64::from_bits(masked_nan2).to_bits(), masked_nan2);
}

#[test]
#[should_panic]
fn test_clamp_min_greater_than_max() {
    let _ = 1.0f64.clamp(3.0, 1.0);
}

#[test]
#[should_panic]
fn test_clamp_min_is_nan() {
    let _ = 1.0f64.clamp(f64::NAN, 1.0);
}

#[test]
#[should_panic]
fn test_clamp_max_is_nan() {
    let _ = 1.0f64.clamp(3.0, f64::NAN);
}

#[test]
fn test_total_cmp() {
    use core::cmp::Ordering;

    fn quiet_bit_mask() -> u64 {
        1 << (f64::MANTISSA_DIGITS - 2)
    }

    fn min_subnorm() -> f64 {
        f64::MIN_POSITIVE / f64::powf(2.0, f64::MANTISSA_DIGITS as f64 - 1.0)
    }

    fn max_subnorm() -> f64 {
        f64::MIN_POSITIVE - min_subnorm()
    }

    fn q_nan() -> f64 {
        f64::from_bits(f64::NAN.to_bits() | quiet_bit_mask())
    }

    fn s_nan() -> f64 {
        f64::from_bits((f64::NAN.to_bits() & !quiet_bit_mask()) + 42)
    }

    assert_eq!(Ordering::Equal, (-q_nan()).total_cmp(&-q_nan()));
    assert_eq!(Ordering::Equal, (-s_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Equal, (-f64::INFINITY).total_cmp(&-f64::INFINITY));
    assert_eq!(Ordering::Equal, (-f64::MAX).total_cmp(&-f64::MAX));
    assert_eq!(Ordering::Equal, (-2.5_f64).total_cmp(&-2.5));
    assert_eq!(Ordering::Equal, (-1.0_f64).total_cmp(&-1.0));
    assert_eq!(Ordering::Equal, (-1.5_f64).total_cmp(&-1.5));
    assert_eq!(Ordering::Equal, (-0.5_f64).total_cmp(&-0.5));
    assert_eq!(Ordering::Equal, (-f64::MIN_POSITIVE).total_cmp(&-f64::MIN_POSITIVE));
    assert_eq!(Ordering::Equal, (-max_subnorm()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Equal, (-min_subnorm()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Equal, (-0.0_f64).total_cmp(&-0.0));
    assert_eq!(Ordering::Equal, 0.0_f64.total_cmp(&0.0));
    assert_eq!(Ordering::Equal, min_subnorm().total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Equal, max_subnorm().total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Equal, f64::MIN_POSITIVE.total_cmp(&f64::MIN_POSITIVE));
    assert_eq!(Ordering::Equal, 0.5_f64.total_cmp(&0.5));
    assert_eq!(Ordering::Equal, 1.0_f64.total_cmp(&1.0));
    assert_eq!(Ordering::Equal, 1.5_f64.total_cmp(&1.5));
    assert_eq!(Ordering::Equal, 2.5_f64.total_cmp(&2.5));
    assert_eq!(Ordering::Equal, f64::MAX.total_cmp(&f64::MAX));
    assert_eq!(Ordering::Equal, f64::INFINITY.total_cmp(&f64::INFINITY));
    assert_eq!(Ordering::Equal, s_nan().total_cmp(&s_nan()));
    assert_eq!(Ordering::Equal, q_nan().total_cmp(&q_nan()));

    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f64::INFINITY));
    assert_eq!(Ordering::Less, (-f64::INFINITY).total_cmp(&-f64::MAX));
    assert_eq!(Ordering::Less, (-f64::MAX).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-2.5_f64).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-1.5_f64).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-1.0_f64).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-0.5_f64).total_cmp(&-f64::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-f64::MIN_POSITIVE).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Less, (-max_subnorm()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-min_subnorm()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-0.0_f64).total_cmp(&0.0));
    assert_eq!(Ordering::Less, 0.0_f64.total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Less, min_subnorm().total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, max_subnorm().total_cmp(&f64::MIN_POSITIVE));
    assert_eq!(Ordering::Less, f64::MIN_POSITIVE.total_cmp(&0.5));
    assert_eq!(Ordering::Less, 0.5_f64.total_cmp(&1.0));
    assert_eq!(Ordering::Less, 1.0_f64.total_cmp(&1.5));
    assert_eq!(Ordering::Less, 1.5_f64.total_cmp(&2.5));
    assert_eq!(Ordering::Less, 2.5_f64.total_cmp(&f64::MAX));
    assert_eq!(Ordering::Less, f64::MAX.total_cmp(&f64::INFINITY));
    assert_eq!(Ordering::Less, f64::INFINITY.total_cmp(&s_nan()));
    assert_eq!(Ordering::Less, s_nan().total_cmp(&q_nan()));

    assert_eq!(Ordering::Greater, (-s_nan()).total_cmp(&-q_nan()));
    assert_eq!(Ordering::Greater, (-f64::INFINITY).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Greater, (-f64::MAX).total_cmp(&-f64::INFINITY));
    assert_eq!(Ordering::Greater, (-2.5_f64).total_cmp(&-f64::MAX));
    assert_eq!(Ordering::Greater, (-1.5_f64).total_cmp(&-2.5));
    assert_eq!(Ordering::Greater, (-1.0_f64).total_cmp(&-1.5));
    assert_eq!(Ordering::Greater, (-0.5_f64).total_cmp(&-1.0));
    assert_eq!(Ordering::Greater, (-f64::MIN_POSITIVE).total_cmp(&-0.5));
    assert_eq!(Ordering::Greater, (-max_subnorm()).total_cmp(&-f64::MIN_POSITIVE));
    assert_eq!(Ordering::Greater, (-min_subnorm()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Greater, (-0.0_f64).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Greater, 0.0_f64.total_cmp(&-0.0));
    assert_eq!(Ordering::Greater, min_subnorm().total_cmp(&0.0));
    assert_eq!(Ordering::Greater, max_subnorm().total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Greater, f64::MIN_POSITIVE.total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Greater, 0.5_f64.total_cmp(&f64::MIN_POSITIVE));
    assert_eq!(Ordering::Greater, 1.0_f64.total_cmp(&0.5));
    assert_eq!(Ordering::Greater, 1.5_f64.total_cmp(&1.0));
    assert_eq!(Ordering::Greater, 2.5_f64.total_cmp(&1.5));
    assert_eq!(Ordering::Greater, f64::MAX.total_cmp(&2.5));
    assert_eq!(Ordering::Greater, f64::INFINITY.total_cmp(&f64::MAX));
    assert_eq!(Ordering::Greater, s_nan().total_cmp(&f64::INFINITY));
    assert_eq!(Ordering::Greater, q_nan().total_cmp(&s_nan()));

    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-s_nan()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f64::INFINITY));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f64::MAX));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-f64::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&0.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f64::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&0.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&1.0));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&1.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&2.5));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f64::MAX));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&f64::INFINITY));
    assert_eq!(Ordering::Less, (-q_nan()).total_cmp(&s_nan()));

    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f64::INFINITY));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f64::MAX));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-2.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-1.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-1.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-0.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-f64::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-max_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-min_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&-0.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&0.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&min_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&max_subnorm()));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f64::MIN_POSITIVE));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&0.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&1.0));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&1.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&2.5));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f64::MAX));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&f64::INFINITY));
    assert_eq!(Ordering::Less, (-s_nan()).total_cmp(&s_nan()));
}
