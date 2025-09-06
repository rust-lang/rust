//@ run-pass

#![feature(const_cmp, f16, f128)]

use std::cmp::Ordering;

fn main() {
    // Test f32::total_cmp in const context
    const F32_EQUAL: Ordering = (1.0_f32).total_cmp(&1.0_f32);
    assert_eq!(F32_EQUAL, Ordering::Equal);

    const F32_LESS: Ordering = (1.0_f32).total_cmp(&2.0_f32);
    assert_eq!(F32_LESS, Ordering::Less);

    const F32_GREATER: Ordering = (2.0_f32).total_cmp(&1.0_f32);
    assert_eq!(F32_GREATER, Ordering::Greater);

    // Test special values for f32
    const F32_NEG_ZERO_VS_POS_ZERO: Ordering = (-0.0_f32).total_cmp(&0.0_f32);
    assert_eq!(F32_NEG_ZERO_VS_POS_ZERO, Ordering::Less);

    const F32_NAN_VS_FINITE: Ordering = f32::NAN.total_cmp(&1.0_f32);
    assert_eq!(F32_NAN_VS_FINITE, Ordering::Greater);

    const F32_INFINITY_VS_FINITE: Ordering = f32::INFINITY.total_cmp(&1.0_f32);
    assert_eq!(F32_INFINITY_VS_FINITE, Ordering::Greater);

    const F32_NEG_INFINITY_VS_FINITE: Ordering = f32::NEG_INFINITY.total_cmp(&1.0_f32);
    assert_eq!(F32_NEG_INFINITY_VS_FINITE, Ordering::Less);

    // Test f64::total_cmp in const context
    const F64_EQUAL: Ordering = (1.0_f64).total_cmp(&1.0_f64);
    assert_eq!(F64_EQUAL, Ordering::Equal);

    const F64_LESS: Ordering = (1.0_f64).total_cmp(&2.0_f64);
    assert_eq!(F64_LESS, Ordering::Less);

    const F64_GREATER: Ordering = (2.0_f64).total_cmp(&1.0_f64);
    assert_eq!(F64_GREATER, Ordering::Greater);

    // Test special values for f64
    const F64_NEG_ZERO_VS_POS_ZERO: Ordering = (-0.0_f64).total_cmp(&0.0_f64);
    assert_eq!(F64_NEG_ZERO_VS_POS_ZERO, Ordering::Less);

    const F64_NAN_VS_FINITE: Ordering = f64::NAN.total_cmp(&1.0_f64);
    assert_eq!(F64_NAN_VS_FINITE, Ordering::Greater);

    const F64_INFINITY_VS_FINITE: Ordering = f64::INFINITY.total_cmp(&1.0_f64);
    assert_eq!(F64_INFINITY_VS_FINITE, Ordering::Greater);

    const F64_NEG_INFINITY_VS_FINITE: Ordering = f64::NEG_INFINITY.total_cmp(&1.0_f64);
    assert_eq!(F64_NEG_INFINITY_VS_FINITE, Ordering::Less);

    // Test edge cases: comparing NaNs with each other
    const F32_NAN_VS_NAN: Ordering = f32::NAN.total_cmp(&f32::NAN);
    assert_eq!(F32_NAN_VS_NAN, Ordering::Equal);

    const F64_NAN_VS_NAN: Ordering = f64::NAN.total_cmp(&f64::NAN);
    assert_eq!(F64_NAN_VS_NAN, Ordering::Equal);

    // Test subnormal numbers
    const F32_SUBNORMAL_CMP: Ordering = f32::MIN_POSITIVE.total_cmp(&(f32::MIN_POSITIVE / 2.0));
    assert_eq!(F32_SUBNORMAL_CMP, Ordering::Greater);

    const F64_SUBNORMAL_CMP: Ordering = f64::MIN_POSITIVE.total_cmp(&(f64::MIN_POSITIVE / 2.0));
    assert_eq!(F64_SUBNORMAL_CMP, Ordering::Greater);

    // Test f16::total_cmp in const context
    const F16_EQUAL: Ordering = (1.0_f16).total_cmp(&1.0_f16);
    assert_eq!(F16_EQUAL, Ordering::Equal);

    const F16_LESS: Ordering = (1.0_f16).total_cmp(&2.0_f16);
    assert_eq!(F16_LESS, Ordering::Less);

    const F16_GREATER: Ordering = (2.0_f16).total_cmp(&1.0_f16);
    assert_eq!(F16_GREATER, Ordering::Greater);

    // Test special values for f16
    const F16_NEG_ZERO_VS_POS_ZERO: Ordering = (-0.0_f16).total_cmp(&0.0_f16);
    assert_eq!(F16_NEG_ZERO_VS_POS_ZERO, Ordering::Less);

    const F16_NAN_VS_FINITE: Ordering = f16::NAN.total_cmp(&1.0_f16);
    assert_eq!(F16_NAN_VS_FINITE, Ordering::Greater);

    const F16_INFINITY_VS_FINITE: Ordering = f16::INFINITY.total_cmp(&1.0_f16);
    assert_eq!(F16_INFINITY_VS_FINITE, Ordering::Greater);

    const F16_NAN_VS_NAN: Ordering = f16::NAN.total_cmp(&f16::NAN);
    assert_eq!(F16_NAN_VS_NAN, Ordering::Equal);

    // Test f128::total_cmp in const context
    const F128_EQUAL: Ordering = (1.0_f128).total_cmp(&1.0_f128);
    assert_eq!(F128_EQUAL, Ordering::Equal);

    const F128_LESS: Ordering = (1.0_f128).total_cmp(&2.0_f128);
    assert_eq!(F128_LESS, Ordering::Less);

    const F128_GREATER: Ordering = (2.0_f128).total_cmp(&1.0_f128);
    assert_eq!(F128_GREATER, Ordering::Greater);

    // Test special values for f128
    const F128_NEG_ZERO_VS_POS_ZERO: Ordering = (-0.0_f128).total_cmp(&0.0_f128);
    assert_eq!(F128_NEG_ZERO_VS_POS_ZERO, Ordering::Less);

    const F128_NAN_VS_FINITE: Ordering = f128::NAN.total_cmp(&1.0_f128);
    assert_eq!(F128_NAN_VS_FINITE, Ordering::Greater);

    const F128_INFINITY_VS_FINITE: Ordering = f128::INFINITY.total_cmp(&1.0_f128);
    assert_eq!(F128_INFINITY_VS_FINITE, Ordering::Greater);

    const F128_NAN_VS_NAN: Ordering = f128::NAN.total_cmp(&f128::NAN);
    assert_eq!(F128_NAN_VS_NAN, Ordering::Equal);
}