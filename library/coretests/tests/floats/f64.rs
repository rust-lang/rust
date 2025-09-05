use core::f64;

use super::assert_biteq;

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
