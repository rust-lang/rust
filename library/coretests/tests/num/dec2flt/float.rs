use core::num::dec2flt::float::RawFloat;

// FIXME(f16_f128): enable on all targets once possible.
#[test]
#[cfg(target_has_reliable_f16)]
fn test_f16_integer_decode() {
    assert_eq!(3.14159265359f16.integer_decode(), (1608, -9, 1));
    assert_eq!((-8573.5918555f16).integer_decode(), (1072, 3, -1));
    #[cfg(not(miri))] // miri doesn't have powf16
    assert_eq!(2f16.powf(14.0).integer_decode(), (1 << 10, 4, 1));
    assert_eq!(0f16.integer_decode(), (0, -25, 1));
    assert_eq!((-0f16).integer_decode(), (0, -25, -1));
    assert_eq!(f16::INFINITY.integer_decode(), (1 << 10, 6, 1));
    assert_eq!(f16::NEG_INFINITY.integer_decode(), (1 << 10, 6, -1));

    // Ignore the "sign" (quiet / signalling flag) of NAN.
    // It can vary between runtime operations and LLVM folding.
    let (nan_m, nan_p, _nan_s) = f16::NAN.integer_decode();
    assert_eq!((nan_m, nan_p), (1536, 6));
}

#[test]
fn test_f32_integer_decode() {
    assert_eq!(3.14159265359f32.integer_decode(), (13176795, -22, 1));
    assert_eq!((-8573.5918555f32).integer_decode(), (8779358, -10, -1));
    // Set 2^100 directly instead of using powf, because it doesn't guarentee precision
    assert_eq!(1.2676506e30_f32.integer_decode(), (8388608, 77, 1));
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
    // Set 2^100 directly instead of using powf, because it doesn't guarentee precision
    assert_eq!(1.2676506002282294e30_f64.integer_decode(), (4503599627370496, 48, 1));
    assert_eq!(0f64.integer_decode(), (0, -1075, 1));
    assert_eq!((-0f64).integer_decode(), (0, -1075, -1));
    assert_eq!(f64::INFINITY.integer_decode(), (4503599627370496, 972, 1));
    assert_eq!(f64::NEG_INFINITY.integer_decode(), (4503599627370496, 972, -1));

    // Ignore the "sign" (quiet / signalling flag) of NAN.
    // It can vary between runtime operations and LLVM folding.
    let (nan_m, nan_p, _nan_s) = f64::NAN.integer_decode();
    assert_eq!((nan_m, nan_p), (6755399441055744, 972));
}

/* Sanity checks of computed magic numbers */

// FIXME(f16_f128): enable on all targets once possible.
#[test]
#[cfg(target_has_reliable_f16)]
fn test_f16_consts() {
    assert_eq!(<f16 as RawFloat>::INFINITY, f16::INFINITY);
    assert_eq!(<f16 as RawFloat>::NEG_INFINITY, -f16::INFINITY);
    assert_eq!(<f16 as RawFloat>::NAN.to_bits(), f16::NAN.to_bits());
    assert_eq!(<f16 as RawFloat>::NEG_NAN.to_bits(), (-f16::NAN).to_bits());
    assert_eq!(<f16 as RawFloat>::SIG_BITS, 10);
    assert_eq!(<f16 as RawFloat>::MIN_EXPONENT_ROUND_TO_EVEN, -22);
    assert_eq!(<f16 as RawFloat>::MAX_EXPONENT_ROUND_TO_EVEN, 5);
    assert_eq!(<f16 as RawFloat>::MIN_EXPONENT_FAST_PATH, -4);
    assert_eq!(<f16 as RawFloat>::MAX_EXPONENT_FAST_PATH, 4);
    assert_eq!(<f16 as RawFloat>::MAX_EXPONENT_DISGUISED_FAST_PATH, 7);
    assert_eq!(<f16 as RawFloat>::EXP_MIN, -14);
    assert_eq!(<f16 as RawFloat>::EXP_SAT, 0x1f);
    assert_eq!(<f16 as RawFloat>::SMALLEST_POWER_OF_TEN, -27);
    assert_eq!(<f16 as RawFloat>::LARGEST_POWER_OF_TEN, 4);
    assert_eq!(<f16 as RawFloat>::MAX_MANTISSA_FAST_PATH, 2048);
}

#[test]
fn test_f32_consts() {
    assert_eq!(<f32 as RawFloat>::INFINITY, f32::INFINITY);
    assert_eq!(<f32 as RawFloat>::NEG_INFINITY, -f32::INFINITY);
    assert_eq!(<f32 as RawFloat>::NAN.to_bits(), f32::NAN.to_bits());
    assert_eq!(<f32 as RawFloat>::NEG_NAN.to_bits(), (-f32::NAN).to_bits());
    assert_eq!(<f32 as RawFloat>::SIG_BITS, 23);
    assert_eq!(<f32 as RawFloat>::MIN_EXPONENT_ROUND_TO_EVEN, -17);
    assert_eq!(<f32 as RawFloat>::MAX_EXPONENT_ROUND_TO_EVEN, 10);
    assert_eq!(<f32 as RawFloat>::MIN_EXPONENT_FAST_PATH, -10);
    assert_eq!(<f32 as RawFloat>::MAX_EXPONENT_FAST_PATH, 10);
    assert_eq!(<f32 as RawFloat>::MAX_EXPONENT_DISGUISED_FAST_PATH, 17);
    assert_eq!(<f32 as RawFloat>::EXP_MIN, -126);
    assert_eq!(<f32 as RawFloat>::EXP_SAT, 0xff);
    assert_eq!(<f32 as RawFloat>::SMALLEST_POWER_OF_TEN, -65);
    assert_eq!(<f32 as RawFloat>::LARGEST_POWER_OF_TEN, 38);
    assert_eq!(<f32 as RawFloat>::MAX_MANTISSA_FAST_PATH, 16777216);
}

#[test]
fn test_f64_consts() {
    assert_eq!(<f64 as RawFloat>::INFINITY, f64::INFINITY);
    assert_eq!(<f64 as RawFloat>::NEG_INFINITY, -f64::INFINITY);
    assert_eq!(<f64 as RawFloat>::NAN.to_bits(), f64::NAN.to_bits());
    assert_eq!(<f64 as RawFloat>::NEG_NAN.to_bits(), (-f64::NAN).to_bits());
    assert_eq!(<f64 as RawFloat>::SIG_BITS, 52);
    assert_eq!(<f64 as RawFloat>::MIN_EXPONENT_ROUND_TO_EVEN, -4);
    assert_eq!(<f64 as RawFloat>::MAX_EXPONENT_ROUND_TO_EVEN, 23);
    assert_eq!(<f64 as RawFloat>::MIN_EXPONENT_FAST_PATH, -22);
    assert_eq!(<f64 as RawFloat>::MAX_EXPONENT_FAST_PATH, 22);
    assert_eq!(<f64 as RawFloat>::MAX_EXPONENT_DISGUISED_FAST_PATH, 37);
    assert_eq!(<f64 as RawFloat>::EXP_MIN, -1022);
    assert_eq!(<f64 as RawFloat>::EXP_SAT, 0x7ff);
    assert_eq!(<f64 as RawFloat>::SMALLEST_POWER_OF_TEN, -342);
    assert_eq!(<f64 as RawFloat>::LARGEST_POWER_OF_TEN, 308);
    assert_eq!(<f64 as RawFloat>::MAX_MANTISSA_FAST_PATH, 9007199254740992);
}
