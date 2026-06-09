use super::generic;

/// Rounds the number toward 0 to the closest integral value (f16).
///
/// This effectively removes the decimal part of the number, leaving the integral part.
#[cfg(f16_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn truncf16(x: f16) -> f16 {
    generic::trunc_status(x).val
}

/// Rounds the number toward 0 to the closest integral value (f32).
///
/// This effectively removes the decimal part of the number, leaving the integral part.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn truncf(x: f32) -> f32 {
    select_implementation! {
        name: truncf,
        use_arch: all(target_arch = "wasm32", intrinsics_enabled),
        args: x,
    }

    generic::trunc_status(x).val
}

/// Rounds the number toward 0 to the closest integral value (f64).
///
/// This effectively removes the decimal part of the number, leaving the integral part.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn trunc(x: f64) -> f64 {
    select_implementation! {
        name: trunc,
        use_arch: all(target_arch = "wasm32", intrinsics_enabled),
        args: x,
    }

    generic::trunc_status(x).val
}

/// Rounds the number toward 0 to the closest integral value (f128).
///
/// This effectively removes the decimal part of the number, leaving the integral part.
#[cfg(f128_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn truncf128(x: f128) -> f128 {
    generic::trunc_status(x).val
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::{Float, FpResult, Hex, Status};

    macro_rules! cases {
        ($f:ty) => {
            [
                // roundtrip
                (0.0, 0.0, Status::OK),
                (-0.0, -0.0, Status::OK),
                (1.0, 1.0, Status::OK),
                (-1.0, -1.0, Status::OK),
                (<$f>::INFINITY, <$f>::INFINITY, Status::OK),
                (<$f>::NEG_INFINITY, <$f>::NEG_INFINITY, Status::OK),
                // with rounding
                (0.1, 0.0, Status::INEXACT),
                (-0.1, -0.0, Status::INEXACT),
                (0.5, 0.0, Status::INEXACT),
                (-0.5, -0.0, Status::INEXACT),
                (0.9, 0.0, Status::INEXACT),
                (-0.9, -0.0, Status::INEXACT),
                (1.1, 1.0, Status::INEXACT),
                (-1.1, -1.0, Status::INEXACT),
                (1.5, 1.0, Status::INEXACT),
                (-1.5, -1.0, Status::INEXACT),
                (1.9, 1.0, Status::INEXACT),
                (-1.9, -1.0, Status::INEXACT),
            ]
        };
    }

    #[track_caller]
    fn check<F: Float>(f: fn(F) -> F, cases: &[(F, F, Status)]) {
        for &(x, exp_res, exp_stat) in cases {
            let FpResult { val, status } = generic::trunc_status(x);
            assert_biteq!(val, exp_res, "generic::trunc_status({x:?}) ({})", Hex(x));
            assert_eq!(
                status,
                exp_stat,
                "{x:?} {} -> {exp_res:?} {}",
                Hex(x),
                Hex(exp_res)
            );
            let val = f(x);
            assert_biteq!(val, exp_res, "trunc({x:?}) ({})", Hex(x));
        }
    }

    #[test]
    #[cfg(f16_enabled)]
    fn check_f16() {
        check::<f16>(truncf16, &cases!(f16));
        check::<f16>(
            truncf16,
            &[
                (hf16!("0x1p10"), hf16!("0x1p10"), Status::OK),
                (hf16!("-0x1p10"), hf16!("-0x1p10"), Status::OK),
            ],
        );
    }

    #[test]
    fn check_f32() {
        check::<f32>(truncf, &cases!(f32));
        check::<f32>(
            truncf,
            &[
                (hf32!("0x1p23"), hf32!("0x1p23"), Status::OK),
                (hf32!("-0x1p23"), hf32!("-0x1p23"), Status::OK),
            ],
        );
    }

    #[test]
    fn check_f64() {
        check::<f64>(trunc, &cases!(f64));
        check::<f64>(
            trunc,
            &[
                (hf64!("0x1p52"), hf64!("0x1p52"), Status::OK),
                (hf64!("-0x1p52"), hf64!("-0x1p52"), Status::OK),
            ],
        );
    }

    #[test]
    #[cfg(f128_enabled)]
    fn check_f128() {
        check::<f128>(truncf128, &cases!(f128));
        check::<f128>(
            truncf128,
            &[
                (hf128!("0x1p112"), hf128!("0x1p112"), Status::OK),
                (hf128!("-0x1p112"), hf128!("-0x1p112"), Status::OK),
            ],
        );
    }
}
