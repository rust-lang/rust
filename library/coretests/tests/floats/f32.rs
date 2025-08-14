use core::f32;
use core::f32::consts;

use super::{assert_approx_eq, assert_biteq};

/// First pattern over the mantissa
const NAN_MASK1: u32 = 0x002a_aaaa;

/// Second pattern over the mantissa
const NAN_MASK2: u32 = 0x0055_5555;

/// Miri adds some extra errors to float functions; make sure the tests still pass.
/// These values are purely used as a canary to test against and are thus not a stable guarantee Rust provides.
/// They serve as a way to get an idea of the real precision of floating point operations on different platforms.
const APPROX_DELTA: f32 = if cfg!(miri) { 1e-4 } else { 1e-6 };

// FIXME(#140515): mingw has an incorrect fma https://sourceforge.net/p/mingw-w64/bugs/848/
#[cfg_attr(all(target_os = "windows", target_env = "gnu", not(target_abi = "llvm")), ignore)]
#[test]
fn test_mul_add() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_biteq!(f32::math::mul_add(12.3f32, 4.5, 6.7), 62.05);
    assert_biteq!(f32::math::mul_add(-12.3f32, -4.5, -6.7), 48.65);
    assert_biteq!(f32::math::mul_add(0.0f32, 8.9, 1.2), 1.2);
    assert_biteq!(f32::math::mul_add(3.4f32, -0.0, 5.6), 5.6);
    assert!(f32::math::mul_add(nan, 7.8, 9.0).is_nan());
    assert_biteq!(f32::math::mul_add(inf, 7.8, 9.0), inf);
    assert_biteq!(f32::math::mul_add(neg_inf, 7.8, 9.0), neg_inf);
    assert_biteq!(f32::math::mul_add(8.9f32, inf, 3.2), inf);
    assert_biteq!(f32::math::mul_add(-3.2f32, 2.4, neg_inf), neg_inf);
}

#[test]
fn test_recip() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_biteq!(1.0f32.recip(), 1.0);
    assert_biteq!(2.0f32.recip(), 0.5);
    assert_biteq!((-0.4f32).recip(), -2.5);
    assert_biteq!(0.0f32.recip(), inf);
    assert!(nan.recip().is_nan());
    assert_biteq!(inf.recip(), 0.0);
    assert_biteq!(neg_inf.recip(), -0.0);
}

#[test]
fn test_powi() {
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_approx_eq!(1.0f32.powi(1), 1.0);
    assert_approx_eq!((-3.1f32).powi(2), 9.61, APPROX_DELTA);
    assert_approx_eq!(5.9f32.powi(-2), 0.028727);
    assert_biteq!(8.3f32.powi(0), 1.0);
    assert!(nan.powi(2).is_nan());
    assert_biteq!(inf.powi(3), inf);
    assert_biteq!(neg_inf.powi(2), inf);
}

#[test]
fn test_to_degrees() {
    let pi: f32 = consts::PI;
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_biteq!(0.0f32.to_degrees(), 0.0);
    assert_approx_eq!((-5.8f32).to_degrees(), -332.315521);
    assert_biteq!(pi.to_degrees(), 180.0);
    assert!(nan.to_degrees().is_nan());
    assert_biteq!(inf.to_degrees(), inf);
    assert_biteq!(neg_inf.to_degrees(), neg_inf);
    assert_biteq!(1_f32.to_degrees(), 57.2957795130823208767981548141051703);
}

#[test]
fn test_to_radians() {
    let pi: f32 = consts::PI;
    let nan: f32 = f32::NAN;
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;
    assert_biteq!(0.0f32.to_radians(), 0.0);
    assert_approx_eq!(154.6f32.to_radians(), 2.698279);
    assert_approx_eq!((-332.31f32).to_radians(), -5.799903);
    assert_biteq!(180.0f32.to_radians(), pi);
    assert!(nan.to_radians().is_nan());
    assert_biteq!(inf.to_radians(), inf);
    assert_biteq!(neg_inf.to_radians(), neg_inf);
}

#[test]
fn test_float_bits_conv() {
    assert_eq!((1f32).to_bits(), 0x3f800000);
    assert_eq!((12.5f32).to_bits(), 0x41480000);
    assert_eq!((1337f32).to_bits(), 0x44a72000);
    assert_eq!((-14.25f32).to_bits(), 0xc1640000);
    assert_biteq!(f32::from_bits(0x3f800000), 1.0);
    assert_biteq!(f32::from_bits(0x41480000), 12.5);
    assert_biteq!(f32::from_bits(0x44a72000), 1337.0);
    assert_biteq!(f32::from_bits(0xc1640000), -14.25);

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
