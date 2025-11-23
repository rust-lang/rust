#![feature(clamp_magnitude)]

#[test]
fn test_clamp_magnitude_i8() {
    // Basic clamping
    assert_eq!(100i8.clamp_magnitude(50), 50);
    assert_eq!(-100i8.clamp_magnitude(50), -50);
    assert_eq!(30i8.clamp_magnitude(50), 30);
    assert_eq!(-30i8.clamp_magnitude(50), -30);

    // Exact boundary
    assert_eq!(50i8.clamp_magnitude(50), 50);
    assert_eq!(-50i8.clamp_magnitude(50), -50);

    // Zero cases
    assert_eq!(0i8.clamp_magnitude(100), 0);
    assert_eq!(0i8.clamp_magnitude(0), 0);

    // Limit is zero
    assert_eq!(100i8.clamp_magnitude(0), 0);
    assert_eq!(-100i8.clamp_magnitude(0), 0);

    // MIN/MAX values
    assert_eq!(i8::MAX.clamp_magnitude(100), i8::MAX);
    assert_eq!(i8::MAX.clamp_magnitude(200), i8::MAX); // limit > MAX
    assert_eq!(i8::MIN.clamp_magnitude(200), -i8::MAX); // Symmetric range
    assert_eq!(i8::MIN.clamp_magnitude(128), -i8::MAX);
    assert_eq!(i8::MIN.clamp_magnitude(127), -127);
    assert_eq!(i8::MIN.clamp_magnitude(10), -10);

    // Limit larger than type max (u8 > i8::MAX)
    assert_eq!(127i8.clamp_magnitude(255), 127);
    assert_eq!(-128i8.clamp_magnitude(255), -127);
}

#[test]
fn test_clamp_magnitude_i16() {
    // Basic clamping
    assert_eq!(1000i16.clamp_magnitude(500), 500);
    assert_eq!(-1000i16.clamp_magnitude(500), -500);
    assert_eq!(300i16.clamp_magnitude(500), 300);
    assert_eq!(-300i16.clamp_magnitude(500), -300);

    // Exact boundary
    assert_eq!(500i16.clamp_magnitude(500), 500);
    assert_eq!(-500i16.clamp_magnitude(500), -500);

    // Zero cases
    assert_eq!(0i16.clamp_magnitude(1000), 0);
    assert_eq!(0i16.clamp_magnitude(0), 0);

    // MIN/MAX values
    assert_eq!(i16::MAX.clamp_magnitude(20000), i16::MAX);
    assert_eq!(i16::MAX.clamp_magnitude(40000), i16::MAX);
    assert_eq!(i16::MIN.clamp_magnitude(40000), -i16::MAX);
    assert_eq!(i16::MIN.clamp_magnitude(32768), -i16::MAX);
    assert_eq!(i16::MIN.clamp_magnitude(32767), -32767);

    // Limit larger than type max
    assert_eq!(32767i16.clamp_magnitude(65535), 32767);
    assert_eq!(-32768i16.clamp_magnitude(65535), -32767);
}

#[test]
fn test_clamp_magnitude_i32() {
    // Basic clamping
    assert_eq!(150i32.clamp_magnitude(100), 100);
    assert_eq!(-150i32.clamp_magnitude(100), -100);
    assert_eq!(80i32.clamp_magnitude(100), 80);
    assert_eq!(-80i32.clamp_magnitude(100), -80);

    // Exact boundary
    assert_eq!(100i32.clamp_magnitude(100), 100);
    assert_eq!(-100i32.clamp_magnitude(100), -100);

    // Zero cases
    assert_eq!(0i32.clamp_magnitude(1000), 0);
    assert_eq!(0i32.clamp_magnitude(0), 0);

    // Large values
    assert_eq!(1_000_000i32.clamp_magnitude(500_000), 500_000);
    assert_eq!(-1_000_000i32.clamp_magnitude(500_000), -500_000);

    // MIN/MAX values
    assert_eq!(i32::MAX.clamp_magnitude(2_000_000_000), i32::MAX);
    assert_eq!(i32::MAX.clamp_magnitude(3_000_000_000), i32::MAX);
    assert_eq!(i32::MIN.clamp_magnitude(3_000_000_000), -i32::MAX);
    assert_eq!(i32::MIN.clamp_magnitude(2_147_483_648), -i32::MAX);
    assert_eq!(i32::MIN.clamp_magnitude(2_147_483_647), -2_147_483_647);

    // Limit larger than type max
    assert_eq!(i32::MAX.clamp_magnitude(u32::MAX), i32::MAX);
    assert_eq!(i32::MIN.clamp_magnitude(u32::MAX), -i32::MAX);
}

#[test]
fn test_clamp_magnitude_i64() {
    // Basic clamping
    assert_eq!(150i64.clamp_magnitude(100), 100);
    assert_eq!(-150i64.clamp_magnitude(100), -100);
    assert_eq!(80i64.clamp_magnitude(100), 80);
    assert_eq!(-80i64.clamp_magnitude(100), -80);

    // Exact boundary
    assert_eq!(100i64.clamp_magnitude(100), 100);
    assert_eq!(-100i64.clamp_magnitude(100), -100);

    // Zero cases
    assert_eq!(0i64.clamp_magnitude(1000), 0);
    assert_eq!(0i64.clamp_magnitude(0), 0);

    // Large values
    assert_eq!(1_000_000_000_000i64.clamp_magnitude(500_000_000_000), 500_000_000_000);
    assert_eq!(-1_000_000_000_000i64.clamp_magnitude(500_000_000_000), -500_000_000_000);

    // MIN/MAX values
    assert_eq!(i64::MAX.clamp_magnitude(9_000_000_000_000_000_000), i64::MAX);
    assert_eq!(i64::MAX.clamp_magnitude(10_000_000_000_000_000_000), i64::MAX);
    assert_eq!(i64::MIN.clamp_magnitude(10_000_000_000_000_000_000), -i64::MAX);
    assert_eq!(i64::MIN.clamp_magnitude(9_223_372_036_854_775_808), -i64::MAX);
    assert_eq!(i64::MIN.clamp_magnitude(9_223_372_036_854_775_807), -9_223_372_036_854_775_807);

    // Limit larger than type max
    assert_eq!(i64::MAX.clamp_magnitude(u64::MAX), i64::MAX);
    assert_eq!(i64::MIN.clamp_magnitude(u64::MAX), -i64::MAX);
}

#[test]
fn test_clamp_magnitude_i128() {
    // Basic clamping
    assert_eq!(150i128.clamp_magnitude(100), 100);
    assert_eq!(-150i128.clamp_magnitude(100), -100);
    assert_eq!(80i128.clamp_magnitude(100), 80);
    assert_eq!(-80i128.clamp_magnitude(100), -80);

    // Exact boundary
    assert_eq!(100i128.clamp_magnitude(100), 100);
    assert_eq!(-100i128.clamp_magnitude(100), -100);

    // Zero cases
    assert_eq!(0i128.clamp_magnitude(1000), 0);
    assert_eq!(0i128.clamp_magnitude(0), 0);

    // Very large values
    let large = 123_456_789_012_345_678_901_234_567_890i128;
    let limit = 100_000_000_000_000_000_000_000_000_000u128;
    assert_eq!(large.clamp_magnitude(limit), limit as i128);
    assert_eq!((-large).clamp_magnitude(limit), -(limit as i128));

    // MIN/MAX values
    assert_eq!(
        i128::MAX.clamp_magnitude(170_000_000_000_000_000_000_000_000_000_000_000_000),
        i128::MAX
    );
    assert_eq!(
        i128::MIN.clamp_magnitude(170_141_183_460_469_231_731_687_303_715_884_105_728),
        -i128::MAX
    );
    assert_eq!(
        i128::MIN.clamp_magnitude(170_141_183_460_469_231_731_687_303_715_884_105_727),
        -170_141_183_460_469_231_731_687_303_715_884_105_727
    );

    // Limit larger than type max
    assert_eq!(i128::MAX.clamp_magnitude(u128::MAX), i128::MAX);
    assert_eq!(i128::MIN.clamp_magnitude(u128::MAX), -i128::MAX);
}

#[test]
fn test_clamp_magnitude_isize() {
    // Basic clamping
    assert_eq!(150isize.clamp_magnitude(100), 100);
    assert_eq!(-150isize.clamp_magnitude(100), -100);
    assert_eq!(80isize.clamp_magnitude(100), 80);
    assert_eq!(-80isize.clamp_magnitude(100), -80);

    // Exact boundary
    assert_eq!(100isize.clamp_magnitude(100), 100);
    assert_eq!(-100isize.clamp_magnitude(100), -100);

    // Zero cases
    assert_eq!(0isize.clamp_magnitude(1000), 0);
    assert_eq!(0isize.clamp_magnitude(0), 0);

    // MIN/MAX values (architecture-dependent)
    assert_eq!(isize::MAX.clamp_magnitude(usize::MAX / 2), isize::MAX);
    assert_eq!(isize::MAX.clamp_magnitude(usize::MAX), isize::MAX);
    assert_eq!(isize::MIN.clamp_magnitude(usize::MAX), -isize::MAX);

    // Test that it works across different architectures
    if isize::MAX == i32::MAX as isize {
        // 32-bit
        assert_eq!(isize::MIN.clamp_magnitude(2_147_483_648), -isize::MAX);
    } else if isize::MAX == i64::MAX as isize {
        // 64-bit
        assert_eq!(isize::MIN.clamp_magnitude(9_223_372_036_854_775_808), -isize::MAX);
    }
}

#[test]
fn test_clamp_magnitude_f32() {
    // Basic clamping
    assert_eq!(5.0f32.clamp_magnitude(3.0), 3.0);
    assert_eq!(-5.0f32.clamp_magnitude(3.0), -3.0);
    assert_eq!(2.0f32.clamp_magnitude(3.0), 2.0);
    assert_eq!(-2.0f32.clamp_magnitude(3.0), -2.0);

    // Exact boundary
    assert_eq!(3.0f32.clamp_magnitude(3.0), 3.0);
    assert_eq!(-3.0f32.clamp_magnitude(3.0), -3.0);

    // Zero cases
    assert_eq!(0.0f32.clamp_magnitude(1.0), 0.0);
    assert_eq!(-0.0f32.clamp_magnitude(1.0), -0.0);
    assert_eq!(5.0f32.clamp_magnitude(0.0), 0.0);
    assert_eq!(-5.0f32.clamp_magnitude(0.0), -0.0);

    // Fractional values
    assert_eq!(1.5f32.clamp_magnitude(1.0), 1.0);
    assert_eq!(-1.5f32.clamp_magnitude(1.0), -1.0);
    assert_eq!(0.5f32.clamp_magnitude(1.0), 0.5);
    assert_eq!(-0.5f32.clamp_magnitude(1.0), -0.5);

    // Very small values
    assert_eq!(0.001f32.clamp_magnitude(0.01), 0.001);
    assert_eq!(-0.001f32.clamp_magnitude(0.01), -0.001);
    assert_eq!(0.1f32.clamp_magnitude(0.01), 0.01);
    assert_eq!(-0.1f32.clamp_magnitude(0.01), -0.01);

    // Large values
    assert_eq!(1e10f32.clamp_magnitude(1e5), 1e5);
    assert_eq!(-1e10f32.clamp_magnitude(1e5), -1e5);
    assert_eq!(1e3f32.clamp_magnitude(1e5), 1e3);
    assert_eq!(-1e3f32.clamp_magnitude(1e5), -1e3);

    // Special values - NaN
    assert!(f32::NAN.clamp_magnitude(1.0).is_nan());
    assert!(f32::NAN.clamp_magnitude(0.0).is_nan());
    assert!(f32::NAN.clamp_magnitude(f32::INFINITY).is_nan());

    // Special values - Infinity
    assert_eq!(f32::INFINITY.clamp_magnitude(100.0), 100.0);
    assert_eq!(f32::NEG_INFINITY.clamp_magnitude(100.0), -100.0);
    assert_eq!(f32::INFINITY.clamp_magnitude(1e10), 1e10);
    assert_eq!(f32::NEG_INFINITY.clamp_magnitude(1e10), -1e10);

    // Normal value with infinite limit
    assert_eq!(1.0f32.clamp_magnitude(f32::INFINITY), 1.0);
    assert_eq!(-1.0f32.clamp_magnitude(f32::INFINITY), -1.0);
    assert_eq!(1e10f32.clamp_magnitude(f32::INFINITY), 1e10);

    // MIN and MAX
    assert_eq!(f32::MAX.clamp_magnitude(1e10), 1e10);
    assert_eq!(f32::MIN.clamp_magnitude(1e10), -1e10);
}

#[test]
fn test_clamp_magnitude_f64() {
    // Basic clamping
    assert_eq!(5.0f64.clamp_magnitude(3.0), 3.0);
    assert_eq!(-5.0f64.clamp_magnitude(3.0), -3.0);
    assert_eq!(2.0f64.clamp_magnitude(3.0), 2.0);
    assert_eq!(-2.0f64.clamp_magnitude(3.0), -2.0);

    // Exact boundary
    assert_eq!(3.0f64.clamp_magnitude(3.0), 3.0);
    assert_eq!(-3.0f64.clamp_magnitude(3.0), -3.0);

    // Zero cases
    assert_eq!(0.0f64.clamp_magnitude(1.0), 0.0);
    assert_eq!(-0.0f64.clamp_magnitude(1.0), -0.0);
    assert_eq!(5.0f64.clamp_magnitude(0.0), 0.0);
    assert_eq!(-5.0f64.clamp_magnitude(0.0), -0.0);

    // Fractional values
    assert_eq!(1.5f64.clamp_magnitude(1.0), 1.0);
    assert_eq!(-1.5f64.clamp_magnitude(1.0), -1.0);
    assert_eq!(0.5f64.clamp_magnitude(1.0), 0.5);
    assert_eq!(-0.5f64.clamp_magnitude(1.0), -0.5);

    // Very small values (higher precision than f32)
    assert_eq!(1e-100f64.clamp_magnitude(1e-50), 1e-100);
    assert_eq!(-1e-100f64.clamp_magnitude(1e-50), -1e-100);
    assert_eq!(1e-10f64.clamp_magnitude(1e-50), 1e-10);
    assert_eq!(-1e-10f64.clamp_magnitude(1e-50), -1e-10);

    // Very large values
    assert_eq!(1e100f64.clamp_magnitude(1e50), 1e50);
    assert_eq!(-1e100f64.clamp_magnitude(1e50), -1e50);
    assert_eq!(1e20f64.clamp_magnitude(1e50), 1e20);
    assert_eq!(-1e20f64.clamp_magnitude(1e50), -1e20);

    // Special values - NaN
    assert!(f64::NAN.clamp_magnitude(1.0).is_nan());
    assert!(f64::NAN.clamp_magnitude(0.0).is_nan());
    assert!(f64::NAN.clamp_magnitude(f64::INFINITY).is_nan());

    // Special values - Infinity
    assert_eq!(f64::INFINITY.clamp_magnitude(100.0), 100.0);
    assert_eq!(f64::NEG_INFINITY.clamp_magnitude(100.0), -100.0);
    assert_eq!(f64::INFINITY.clamp_magnitude(1e100), 1e100);
    assert_eq!(f64::NEG_INFINITY.clamp_magnitude(1e100), -1e100);

    // Normal value with infinite limit
    assert_eq!(1.0f64.clamp_magnitude(f64::INFINITY), 1.0);
    assert_eq!(-1.0f64.clamp_magnitude(f64::INFINITY), -1.0);
    assert_eq!(1e100f64.clamp_magnitude(f64::INFINITY), 1e100);

    // MIN and MAX
    assert_eq!(f64::MAX.clamp_magnitude(1e100), 1e100);
    assert_eq!(f64::MIN.clamp_magnitude(1e100), -1e100);
}

#[test]
#[should_panic(expected = "limit must be non-negative")]
fn test_clamp_magnitude_f32_panic_negative_limit() {
    let _ = 1.0f32.clamp_magnitude(-1.0);
}

#[test]
#[should_panic(expected = "limit must be non-negative")]
fn test_clamp_magnitude_f64_panic_negative_limit() {
    let _ = 1.0f64.clamp_magnitude(-1.0);
}

#[test]
#[should_panic]
fn test_clamp_magnitude_f32_panic_nan_limit() {
    let _ = 1.0f32.clamp_magnitude(f32::NAN);
}

#[test]
#[should_panic]
fn test_clamp_magnitude_f64_panic_nan_limit() {
    let _ = 1.0f64.clamp_magnitude(f64::NAN);
}

#[test]
fn test_clamp_magnitude_symmetry() {
    // Test that clamp_magnitude(x, limit) == -clamp_magnitude(-x, limit) for integers
    for val in [-100i32, -50, -10, -1, 0, 1, 10, 50, 100] {
        for limit in [0u32, 1, 10, 25, 50, 75, 100, 200] {
            assert_eq!(val.clamp_magnitude(limit), -(-val).clamp_magnitude(limit));
        }
    }

    // Test for floats (approximately)
    for val in [-100.0f32, -50.5, -10.1, -1.0, 0.0, 1.0, 10.1, 50.5, 100.0] {
        for limit in [0.0, 1.0, 10.0, 25.0, 50.0, 75.0, 100.0, 200.0] {
            assert_eq!(val.clamp_magnitude(limit), -(-val).clamp_magnitude(limit));
        }
    }
}

#[test]
fn test_clamp_magnitude_equivalent_to_clamp() {
    // For integers
    for val in [-100i32, -50, -10, -1, 0, 1, 10, 50, 100] {
        for limit in [0u32, 1, 10, 25, 50, 75, 100] {
            let limit_signed = limit as i32;
            assert_eq!(val.clamp_magnitude(limit), val.clamp(-limit_signed, limit_signed));
        }
    }

    // For floats
    for val in [-100.0f64, -50.5, -10.1, -1.0, 0.0, 1.0, 10.1, 50.5, 100.0] {
        for limit in [0.0, 1.0, 10.0, 25.0, 50.0, 75.0, 100.0] {
            assert_eq!(val.clamp_magnitude(limit), val.clamp(-limit, limit));
        }
    }
}
