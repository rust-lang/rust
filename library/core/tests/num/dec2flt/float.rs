use core::num::dec2flt::float::RawFloat;

#[test]
fn test_f16_integer_decode() {
    assert_eq!(3.14159265359f16.integer_decode(), (1608, -9, 1));
    assert_eq!((-8573.5918555f16).integer_decode(), (1072, 3, -1));
    assert_eq!(2f16.powf(4.0).integer_decode(), (1024, -6, 1));
    assert_eq!(0f16.integer_decode(), (0, -25, 1));
    assert_eq!((-0f16).integer_decode(), (0, -25, -1));
    assert_eq!(f16::INFINITY.integer_decode(), (1024, 6, 1));
    assert_eq!(f16::NEG_INFINITY.integer_decode(), (1024, 6, -1));

    // Ignore the "sign" (quiet / signalling flag) of NAN.
    // It can vary between runtime operations and LLVM folding.
    let (nan_m, nan_p, _nan_s) = f16::NAN.integer_decode();
    assert_eq!((nan_m, nan_p), (1536, 6));
}

#[test]
fn test_f32_integer_decode() {
    assert_eq!(3.14159265359f32.integer_decode(), (13176795, -22, 1));
    assert_eq!((-8573.5918555f32).integer_decode(), (8779358, -10, -1));
    assert_eq!(2f32.powf(100.0).integer_decode(), (8388608, 77, 1));
    assert_eq!(0f32.integer_decode(), (0, -150, 1));
    assert_eq!((-0f32).integer_decode(), (0, -150, -1));
    assert_eq!(f32::INFINITY.integer_decode(), (8388608, 105, 1));
    assert_eq!(f32::NEG_INFINITY.integer_decode(), (8388608, 105, -1));

    // Ignore the "sign" (quiet / signalling flag) of NAN.
    // It can vary between runtime operations and LLVM folding.
    let (nan_m, nan_p, _nan_s) = f32::NAN.integer_decode();
    assert_eq!((nan_m, nan_p), (12582912, 105));
}

#[test]
fn test_f64_integer_decode() {
    assert_eq!(3.14159265359f64.integer_decode(), (7074237752028906, -51, 1));
    assert_eq!((-8573.5918555f64).integer_decode(), (4713381968463931, -39, -1));
    assert_eq!(2f64.powf(100.0).integer_decode(), (4503599627370496, 48, 1));
    assert_eq!(0f64.integer_decode(), (0, -1075, 1));
    assert_eq!((-0f64).integer_decode(), (0, -1075, -1));
    assert_eq!(f64::INFINITY.integer_decode(), (4503599627370496, 972, 1));
    assert_eq!(f64::NEG_INFINITY.integer_decode(), (4503599627370496, 972, -1));

    // Ignore the "sign" (quiet / signalling flag) of NAN.
    // It can vary between runtime operations and LLVM folding.
    let (nan_m, nan_p, _nan_s) = f64::NAN.integer_decode();
    assert_eq!((nan_m, nan_p), (6755399441055744, 972));
}
