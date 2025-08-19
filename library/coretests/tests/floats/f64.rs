use core::f64;
use core::f64::consts;

use super::{assert_approx_eq, assert_biteq};

/// First pattern over the mantissa
const NAN_MASK1: u64 = 0x000a_aaaa_aaaa_aaaa;

/// Second pattern over the mantissa
const NAN_MASK2: u64 = 0x0005_5555_5555_5555;

// FIXME(#140515): mingw has an incorrect fma https://sourceforge.net/p/mingw-w64/bugs/848/
#[cfg_attr(all(target_os = "windows", target_env = "gnu", not(target_abi = "llvm")), ignore)]
#[test]
fn test_mul_add() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_biteq!(12.3f64.mul_add(4.5, 6.7), 62.050000000000004);
    assert_biteq!((-12.3f64).mul_add(-4.5, -6.7), 48.650000000000006);
    assert_biteq!(0.0f64.mul_add(8.9, 1.2), 1.2);
    assert_biteq!(3.4f64.mul_add(-0.0, 5.6), 5.6);
    assert!(nan.mul_add(7.8, 9.0).is_nan());
    assert_biteq!(inf.mul_add(7.8, 9.0), inf);
    assert_biteq!(neg_inf.mul_add(7.8, 9.0), neg_inf);
    assert_biteq!(8.9f64.mul_add(inf, 3.2), inf);
    assert_biteq!((-3.2f64).mul_add(2.4, neg_inf), neg_inf);
}

#[test]
fn test_recip() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_biteq!(1.0f64.recip(), 1.0);
    assert_biteq!(2.0f64.recip(), 0.5);
    assert_biteq!((-0.4f64).recip(), -2.5);
    assert_biteq!(0.0f64.recip(), inf);
    assert!(nan.recip().is_nan());
    assert_biteq!(inf.recip(), 0.0);
    assert_biteq!(neg_inf.recip(), -0.0);
}

#[test]
fn test_powi() {
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_approx_eq!(1.0f64.powi(1), 1.0);
    assert_approx_eq!((-3.1f64).powi(2), 9.61);
    assert_approx_eq!(5.9f64.powi(-2), 0.028727);
    assert_biteq!(8.3f64.powi(0), 1.0);
    assert!(nan.powi(2).is_nan());
    assert_biteq!(inf.powi(3), inf);
    assert_biteq!(neg_inf.powi(2), inf);
}

#[test]
fn test_to_degrees() {
    let pi: f64 = consts::PI;
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_biteq!(0.0f64.to_degrees(), 0.0);
    assert_approx_eq!((-5.8f64).to_degrees(), -332.315521);
    assert_biteq!(pi.to_degrees(), 180.0);
    assert!(nan.to_degrees().is_nan());
    assert_biteq!(inf.to_degrees(), inf);
    assert_biteq!(neg_inf.to_degrees(), neg_inf);
}

#[test]
fn test_to_radians() {
    let pi: f64 = consts::PI;
    let nan: f64 = f64::NAN;
    let inf: f64 = f64::INFINITY;
    let neg_inf: f64 = f64::NEG_INFINITY;
    assert_biteq!(0.0f64.to_radians(), 0.0);
    assert_approx_eq!(154.6f64.to_radians(), 2.698279);
    assert_approx_eq!((-332.31f64).to_radians(), -5.799903);
    assert_biteq!(180.0f64.to_radians(), pi);
    assert!(nan.to_radians().is_nan());
    assert_biteq!(inf.to_radians(), inf);
    assert_biteq!(neg_inf.to_radians(), neg_inf);
}

#[test]
fn test_float_bits_conv() {
    assert_eq!((1f64).to_bits(), 0x3ff0000000000000);
    assert_eq!((12.5f64).to_bits(), 0x4029000000000000);
    assert_eq!((1337f64).to_bits(), 0x4094e40000000000);
    assert_eq!((-14.25f64).to_bits(), 0xc02c800000000000);
    assert_biteq!(f64::from_bits(0x3ff0000000000000), 1.0);
    assert_biteq!(f64::from_bits(0x4029000000000000), 12.5);
    assert_biteq!(f64::from_bits(0x4094e40000000000), 1337.0);
    assert_biteq!(f64::from_bits(0xc02c800000000000), -14.25);

    // Check that NaNs roundtrip their bits regardless of signaling-ness
    let masked_nan1 = f64::NAN.to_bits() ^ NAN_MASK1;
    let masked_nan2 = f64::NAN.to_bits() ^ NAN_MASK2;
    assert!(f64::from_bits(masked_nan1).is_nan());
    assert!(f64::from_bits(masked_nan2).is_nan());

    assert_eq!(f64::from_bits(masked_nan1).to_bits(), masked_nan1);
    assert_eq!(f64::from_bits(masked_nan2).to_bits(), masked_nan2);
}

#[test]
fn test_algebraic() {
    let a: f64 = 123.0;
    let b: f64 = 456.0;

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
