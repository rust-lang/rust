use core::f32;

use super::assert_biteq;

/// First pattern over the mantissa
const NAN_MASK1: u32 = 0x002a_aaaa;

/// Second pattern over the mantissa
const NAN_MASK2: u32 = 0x0055_5555;

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
