use super::generic;

/// Round `x` to the nearest integer, breaking ties toward even.
#[cfg(f16_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn rintf16(x: f16) -> f16 {
    select_implementation! {
        name: rintf16,
        use_arch: all(target_arch = "aarch64", target_feature = "fp16"),
        args: x,
    }

    generic::rint_status(x).val
}

/// Round `x` to the nearest integer, breaking ties toward even.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn rintf(x: f32) -> f32 {
    select_implementation! {
        name: rintf,
        use_arch: any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "wasm32", intrinsics_enabled),
        ),
        args: x,
    }

    generic::rint_status(x).val
}

/// Round `x` to the nearest integer, breaking ties toward even.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn rint(x: f64) -> f64 {
    select_implementation! {
        name: rint,
        use_arch: any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "wasm32", intrinsics_enabled),
        ),
        args: x,
    }

    generic::rint_status(x).val
}

/// Round `x` to the nearest integer, breaking ties toward even.
#[cfg(f128_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn rintf128(x: f128) -> f128 {
    generic::rint_status(x).val
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::{Float, FpResult, Hex, Status};

    fn spec_test<F: Float>(cases: &[(F, F, Status)]) {
        let roundtrip = [
            F::ZERO,
            F::ONE,
            F::NEG_ONE,
            F::NEG_ZERO,
            F::INFINITY,
            F::NEG_INFINITY,
        ];

        for x in roundtrip {
            let FpResult { val, status } = generic::rint_status(x);
            assert_biteq!(val, x, "rint_status({})", Hex(x));
            assert_eq!(status, Status::OK, "{}", Hex(x));
        }

        for &(x, res, res_stat) in cases {
            let FpResult { val, status } = generic::rint_status(x);
            assert_biteq!(val, res, "rint_status({})", Hex(x));
            assert_eq!(status, res_stat, "{}", Hex(x));
        }
    }

    #[test]
    #[cfg(f16_enabled)]
    fn spec_tests_f16() {
        let cases = [];
        spec_test::<f16>(&cases);
    }

    #[test]
    fn spec_tests_f32() {
        let cases = [
            (0.1, 0.0, Status::INEXACT),
            (-0.1, -0.0, Status::INEXACT),
            (0.5, 0.0, Status::INEXACT),
            (-0.5, -0.0, Status::INEXACT),
            (0.9, 1.0, Status::INEXACT),
            (-0.9, -1.0, Status::INEXACT),
            (1.1, 1.0, Status::INEXACT),
            (-1.1, -1.0, Status::INEXACT),
            (1.5, 2.0, Status::INEXACT),
            (-1.5, -2.0, Status::INEXACT),
            (1.9, 2.0, Status::INEXACT),
            (-1.9, -2.0, Status::INEXACT),
            (2.8, 3.0, Status::INEXACT),
            (-2.8, -3.0, Status::INEXACT),
        ];
        spec_test::<f32>(&cases);
    }

    #[test]
    fn spec_tests_f64() {
        let cases = [
            (0.1, 0.0, Status::INEXACT),
            (-0.1, -0.0, Status::INEXACT),
            (0.5, 0.0, Status::INEXACT),
            (-0.5, -0.0, Status::INEXACT),
            (0.9, 1.0, Status::INEXACT),
            (-0.9, -1.0, Status::INEXACT),
            (1.1, 1.0, Status::INEXACT),
            (-1.1, -1.0, Status::INEXACT),
            (1.5, 2.0, Status::INEXACT),
            (-1.5, -2.0, Status::INEXACT),
            (1.9, 2.0, Status::INEXACT),
            (-1.9, -2.0, Status::INEXACT),
            (2.8, 3.0, Status::INEXACT),
            (-2.8, -3.0, Status::INEXACT),
        ];
        spec_test::<f64>(&cases);
    }

    #[test]
    #[cfg(f128_enabled)]
    fn spec_tests_f128() {
        let cases = [];
        spec_test::<f128>(&cases);
    }
}
