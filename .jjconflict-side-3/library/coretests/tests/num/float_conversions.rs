// Tests for the `float_conversions` methods (ACP rust-lang/libs-team#810):
// `cast`, `to_int_saturating`, `to_int_checked`, and `to_int_strict`.

#[test]
fn cast_widen_narrow() {
    assert_eq!(1.5_f32.cast::<f64>(), 1.5_f64);
    assert_eq!(0.5_f64.cast::<f32>(), 0.5_f32);
    // Narrowing a value out of the target range produces an infinity.
    assert_eq!(1e300_f64.cast::<f32>(), f32::INFINITY);
    assert_eq!((-1e300_f64).cast::<f32>(), f32::NEG_INFINITY);
    // Same-type cast is the identity.
    assert_eq!(3.25_f64.cast::<f64>(), 3.25_f64);
    // Narrowing rounds to the nearest representable value.
    assert_eq!(0.1_f64.cast::<f32>(), 0.1_f32);
}

#[test]
fn saturating_basics() {
    assert_eq!(255.9_f32.to_int_saturating::<u8>(), 255);
    assert_eq!(300.0_f32.to_int_saturating::<u8>(), 255);
    assert_eq!((-1.0_f32).to_int_saturating::<u8>(), 0);
    assert_eq!(f32::NAN.to_int_saturating::<u8>(), 0);
    assert_eq!(f32::INFINITY.to_int_saturating::<u8>(), 255);
    assert_eq!(f32::NEG_INFINITY.to_int_saturating::<i8>(), i8::MIN);
    assert_eq!(f64::NAN.to_int_saturating::<i32>(), 0);
    // Large finite values saturate at the target boundary.
    assert_eq!(1e18_f64.to_int_saturating::<i32>(), i32::MAX);
    assert_eq!((-1e18_f64).to_int_saturating::<i32>(), i32::MIN);
}

#[test]
fn checked_truncates_then_bounds() {
    // Truncation toward zero happens before the bounds check.
    assert_eq!(255.5_f64.to_int_checked::<u8>(), Some(255));
    assert_eq!(255.9_f64.to_int_checked::<u8>(), Some(255));
    assert_eq!(256.0_f64.to_int_checked::<u8>(), None);
    // A negative fraction truncates toward zero and fits.
    assert_eq!((-0.5_f64).to_int_checked::<u8>(), Some(0));
    assert_eq!((-1.0_f64).to_int_checked::<u8>(), None);
    // Non-finite is always None.
    assert_eq!(f64::NAN.to_int_checked::<u8>(), None);
    assert_eq!(f64::INFINITY.to_int_checked::<u8>(), None);
    assert_eq!(f64::NEG_INFINITY.to_int_checked::<i32>(), None);
}

#[test]
fn checked_signed_boundaries() {
    assert_eq!((-128.0_f64).to_int_checked::<i8>(), Some(-128));
    assert_eq!((-128.9_f64).to_int_checked::<i8>(), Some(-128));
    assert_eq!((-129.0_f64).to_int_checked::<i8>(), None);
    assert_eq!(127.0_f64.to_int_checked::<i8>(), Some(127));
    assert_eq!(127.9_f64.to_int_checked::<i8>(), Some(127));
    assert_eq!(128.0_f64.to_int_checked::<i8>(), None);
}

#[test]
fn checked_exact_power_of_two_bounds() {
    // i32::MIN is exactly representable and must be accepted.
    assert_eq!((i32::MIN as f64).to_int_checked::<i32>(), Some(i32::MIN));
    // 2^31 as f32 is exact and one past i32::MAX, so it is rejected...
    assert_eq!((2147483648.0_f32).to_int_checked::<i32>(), None);
    // ...while the largest f32 below 2^31 is accepted.
    assert_eq!((2147483520.0_f32).to_int_checked::<i32>(), Some(2147483520));
}

#[test]
fn strict_matches_checked() {
    assert_eq!(255.5_f64.to_int_strict::<u8>(), 255);
    assert_eq!((-128.0_f64).to_int_strict::<i8>(), -128);
}

#[test]
#[should_panic]
fn strict_panics_on_nan() {
    let _ = f64::NAN.to_int_strict::<u8>();
}

#[test]
#[should_panic]
fn strict_panics_on_overflow() {
    let _ = 256.0_f64.to_int_strict::<u8>();
}

#[cfg(target_has_reliable_f16)]
#[test]
fn f16_into_wide_int_accepts_all_finite() {
    // Every finite f16 fits in i128, so the bounds are +/-inf and accept all
    // finite values; only non-finite is rejected.
    assert_eq!(f16::MAX.to_int_checked::<i128>(), Some(f16::MAX as i128));
    assert_eq!((-f16::MAX).to_int_checked::<i128>(), Some(-f16::MAX as i128));
    assert_eq!(f16::INFINITY.to_int_checked::<u128>(), None);
    assert_eq!(f16::NAN.to_int_checked::<u128>(), None);
    assert_eq!(4.6_f16.to_int_saturating::<u8>(), 4);
    assert_eq!(1.5_f16.cast::<f32>(), 1.5_f32);
}

// Truncation-then-check must not depend on a libm `trunc`, which is unreliable
// for f16/f128 on some targets. These exercise the fractional checked path.
#[cfg(target_has_reliable_f16)]
#[test]
fn f16_checked_fractional() {
    assert_eq!(4.6_f16.to_int_checked::<u8>(), Some(4));
    assert_eq!(255.5_f16.to_int_checked::<u8>(), Some(255));
    assert_eq!(256.0_f16.to_int_checked::<u8>(), None);
    assert_eq!((-0.5_f16).to_int_checked::<u8>(), Some(0));
    assert_eq!(4.6_f16.to_int_strict::<u8>(), 4);
}

#[cfg(target_has_reliable_f128)]
#[test]
fn f128_checked_fractional() {
    assert_eq!(4.6_f128.to_int_checked::<u8>(), Some(4));
    assert_eq!(255.5_f128.to_int_checked::<u8>(), Some(255));
    assert_eq!(256.0_f128.to_int_checked::<u8>(), None);
    assert_eq!((-0.5_f128).to_int_checked::<u8>(), Some(0));
    assert_eq!(4.6_f128.to_int_strict::<u8>(), 4);
}
