use core::num::dec2flt::decimal::Decimal;

type FPath<F> = ((i64, u64, bool, bool), Option<F>);

const FPATHS_F32: &[FPath<f32>] =
    &[((0, 0, false, false), Some(0.0)), ((0, 0, false, false), Some(0.0))];
const FPATHS_F64: &[FPath<f64>] =
    &[((0, 0, false, false), Some(0.0)), ((0, 0, false, false), Some(0.0))];

// FIXME(f16_f128): enable on all targets once possible.
#[test]
#[cfg(target_has_reliable_f16)]
fn check_fast_path_f16() {
    const FPATHS_F16: &[FPath<f16>] =
        &[((0, 0, false, false), Some(0.0)), ((0, 0, false, false), Some(0.0))];
    for ((exponent, mantissa, negative, many_digits), expected) in FPATHS_F16.iter().copied() {
        let dec = Decimal { exponent, mantissa, negative, many_digits };
        let actual = dec.try_fast_path::<f16>();

        assert_eq!(actual, expected);
    }
}

#[test]
fn check_fast_path_f32() {
    for ((exponent, mantissa, negative, many_digits), expected) in FPATHS_F32.iter().copied() {
        let dec = Decimal { exponent, mantissa, negative, many_digits };
        let actual = dec.try_fast_path::<f32>();

        assert_eq!(actual, expected);
    }
}

#[test]
fn check_fast_path_f64() {
    for ((exponent, mantissa, negative, many_digits), expected) in FPATHS_F64.iter().copied() {
        let dec = Decimal { exponent, mantissa, negative, many_digits };
        let actual = dec.try_fast_path::<f64>();

        assert_eq!(actual, expected);
    }
}
