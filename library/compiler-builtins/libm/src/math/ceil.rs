use super::generic;

/// Ceil (f16)
///
/// Finds the nearest integer greater than or equal to `x`.
#[cfg(f16_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn ceilf16(x: f16) -> f16 {
    generic::ceil_status(x).val
}

/// Ceil (f32)
///
/// Finds the nearest integer greater than or equal to `x`.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn ceilf(x: f32) -> f32 {
    select_implementation! {
        name: ceilf,
        use_arch: all(target_arch = "wasm32", intrinsics_enabled),
        args: x,
    }

    generic::ceil_status(x).val
}

/// Ceil (f64)
///
/// Finds the nearest integer greater than or equal to `x`.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn ceil(x: f64) -> f64 {
    select_implementation! {
        name: ceil,
        use_arch: all(target_arch = "wasm32", intrinsics_enabled),
        use_arch_required: all(target_arch = "x86", not(target_feature = "sse2")),
        args: x,
    }

    generic::ceil_status(x).val
}

/// Ceil (f128)
///
/// Finds the nearest integer greater than or equal to `x`.
#[cfg(f128_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn ceilf128(x: f128) -> f128 {
    generic::ceil_status(x).val
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
                (0.1, 1.0, Status::INEXACT),
                (-0.1, -0.0, Status::INEXACT),
                (0.5, 1.0, Status::INEXACT),
                (-0.5, -0.0, Status::INEXACT),
                (0.9, 1.0, Status::INEXACT),
                (-0.9, -0.0, Status::INEXACT),
                (1.1, 2.0, Status::INEXACT),
                (-1.1, -1.0, Status::INEXACT),
                (1.5, 2.0, Status::INEXACT),
                (-1.5, -1.0, Status::INEXACT),
                (1.9, 2.0, Status::INEXACT),
                (-1.9, -1.0, Status::INEXACT),
            ]
        };
    }

    #[track_caller]
    fn check<F: Float>(cases: &[(F, F, Status)]) {
        for &(x, exp_res, exp_stat) in cases {
            let FpResult { val, status } = generic::ceil_status(x);
            assert_biteq!(val, exp_res, "{x:?} {}", Hex(x));
            assert_eq!(
                status,
                exp_stat,
                "{x:?} {} -> {exp_res:?} {}",
                Hex(x),
                Hex(exp_res)
            );
        }
    }

    #[test]
    #[cfg(f16_enabled)]
    fn check_f16() {
        check::<f16>(&cases!(f16));
        check::<f16>(&[
            (hf16!("0x1p10"), hf16!("0x1p10"), Status::OK),
            (hf16!("-0x1p10"), hf16!("-0x1p10"), Status::OK),
        ]);
    }

    #[test]
    fn check_f32() {
        check::<f32>(&cases!(f32));
        check::<f32>(&[
            (hf32!("0x1p23"), hf32!("0x1p23"), Status::OK),
            (hf32!("-0x1p23"), hf32!("-0x1p23"), Status::OK),
        ]);
    }

    #[test]
    fn check_f64() {
        check::<f64>(&cases!(f64));
        check::<f64>(&[
            (hf64!("0x1p52"), hf64!("0x1p52"), Status::OK),
            (hf64!("-0x1p52"), hf64!("-0x1p52"), Status::OK),
        ]);
    }

    #[test]
    #[cfg(f128_enabled)]
    fn spec_tests_f128() {
        check::<f128>(&cases!(f128));
        check::<f128>(&[
            (hf128!("0x1p112"), hf128!("0x1p112"), Status::OK),
            (hf128!("-0x1p112"), hf128!("-0x1p112"), Status::OK),
        ]);
    }
}
