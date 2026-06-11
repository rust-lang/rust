#![feature(f128)]
#![feature(portable_simd)]

use core_simd::simd::{
    Select, Simd,
    cmp::{SimdPartialEq, SimdPartialOrd},
    f128x2, mask128x2, u128x2,
    num::SimdFloat,
};

#[test]
fn f128_vectors_support_basic_ops() {
    let a = f128x2::from_array([1.0f128, 4.0]);
    let b = f128x2::from_array([2.0f128, 8.0]);

    assert_eq!((a + b).to_array(), [3.0f128, 12.0]);
    assert_eq!((b - a).to_array(), [1.0f128, 4.0]);
    assert_eq!((a * b).to_array(), [2.0f128, 32.0]);
    assert_eq!((b / a).to_array(), [2.0f128, 2.0]);
    assert_eq!((-a).to_array(), [-1.0f128, -4.0]);

    let mask = a.simd_lt(b);
    assert_eq!(mask.to_array(), [true, true]);
    assert_eq!(mask.select(a, b).to_array(), a.to_array());
    assert_eq!(a.simd_min(b).to_array(), a.to_array());
    assert_eq!(b.simd_max(a).to_array(), b.to_array());
    assert!(a.simd_ne(b).all());
}

#[test]
fn f128_vectors_expose_u128_bits() {
    let one = f128x2::splat(1.0);
    let bits: Simd<u128, 2> = one.to_bits();
    assert_eq!(f128x2::from_bits(bits).to_array(), one.to_array());
    assert_eq!(bits & u128x2::splat(u128::MAX), bits);
}

#[test]
fn i128_masks_support_public_mask_api() {
    let mask = mask128x2::from_array([true, false]);
    assert_eq!(mask.to_array(), [true, false]);
    assert_eq!(mask.to_bitmask(), 0b01);
    assert_eq!(mask128x2::from_bitmask(0b10).to_array(), [false, true]);
}
