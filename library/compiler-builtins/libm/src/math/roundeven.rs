use super::generic;
use super::support::Round;

/// Round `x` to the nearest integer, breaking ties toward even. This is IEEE 754
/// `roundToIntegralTiesToEven`.
#[cfg(f16_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn roundevenf16(x: f16) -> f16 {
    generic::rint_round(x, Round::Nearest).val
}

/// Round `x` to the nearest integer, breaking ties toward even. This is IEEE 754
/// `roundToIntegralTiesToEven`.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn roundevenf(x: f32) -> f32 {
    generic::rint_round(x, Round::Nearest).val
}

/// Round `x` to the nearest integer, breaking ties toward even. This is IEEE 754
/// `roundToIntegralTiesToEven`.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn roundeven(x: f64) -> f64 {
    generic::rint_round(x, Round::Nearest).val
}

/// Round `x` to the nearest integer, breaking ties toward even. This is IEEE 754
/// `roundToIntegralTiesToEven`.
#[cfg(f128_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn roundevenf128(x: f128) -> f128 {
    generic::rint_round(x, Round::Nearest).val
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
            let FpResult { val, status } = generic::rint_round(x, Round::Nearest);
            assert_biteq!(val, x, "rint_round({})", Hex(x));
            assert_eq!(status, Status::OK, "{}", Hex(x));
        }

        for &(x, res, res_stat) in cases {
            let FpResult { val, status } = generic::rint_round(x, Round::Nearest);
            assert_biteq!(val, res, "rint_round({})", Hex(x));
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
            (0.1, 0.0, Status::OK),
            (-0.1, -0.0, Status::OK),
            (0.5, 0.0, Status::OK),
            (-0.5, -0.0, Status::OK),
            (0.9, 1.0, Status::OK),
            (-0.9, -1.0, Status::OK),
            (1.1, 1.0, Status::OK),
            (-1.1, -1.0, Status::OK),
            (1.5, 2.0, Status::OK),
            (-1.5, -2.0, Status::OK),
            (1.9, 2.0, Status::OK),
            (-1.9, -2.0, Status::OK),
            (2.8, 3.0, Status::OK),
            (-2.8, -3.0, Status::OK),
        ];
        spec_test::<f32>(&cases);
    }

    #[test]
    fn spec_tests_f64() {
        let cases = [
            (0.1, 0.0, Status::OK),
            (-0.1, -0.0, Status::OK),
            (0.5, 0.0, Status::OK),
            (-0.5, -0.0, Status::OK),
            (0.9, 1.0, Status::OK),
            (-0.9, -1.0, Status::OK),
            (1.1, 1.0, Status::OK),
            (-1.1, -1.0, Status::OK),
            (1.5, 2.0, Status::OK),
            (-1.5, -2.0, Status::OK),
            (1.9, 2.0, Status::OK),
            (-1.9, -2.0, Status::OK),
            (2.8, 3.0, Status::OK),
            (-2.8, -3.0, Status::OK),
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
