/* SPDX-License-Identifier: MIT OR Apache-2.0 */
//! IEEE 754-2019 `minimum`.
//!
//! Per the spec, returns the canonicalized result of:
//! - `x` if `x < y`
//! - `y` if `y < x`
//! - qNaN if either operation is NaN
//! - Logic following +0.0 > -0.0
//!
//! Excluded from our implementation is sNaN handling.

use super::super::Float;

pub fn fminimum<F: Float>(x: F, y: F) -> F {
    let res = if x.is_nan() {
        x
    } else if y.is_nan() {
        y
    } else if x < y || (x.to_bits() == F::NEG_ZERO.to_bits() && y.is_sign_positive()) {
        x
    } else {
        y
    };

    // Canonicalize
    res * F::ONE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::Hexf;

    fn spec_test<F: Float>() {
        let cases = [
            (F::ZERO, F::ZERO, F::ZERO),
            (F::ONE, F::ONE, F::ONE),
            (F::ZERO, F::ONE, F::ZERO),
            (F::ONE, F::ZERO, F::ZERO),
            (F::ZERO, F::NEG_ONE, F::NEG_ONE),
            (F::NEG_ONE, F::ZERO, F::NEG_ONE),
            (F::INFINITY, F::ZERO, F::ZERO),
            (F::NEG_INFINITY, F::ZERO, F::NEG_INFINITY),
            (F::NAN, F::ZERO, F::NAN),
            (F::ZERO, F::NAN, F::NAN),
            (F::NAN, F::NAN, F::NAN),
            (F::ZERO, F::NEG_ZERO, F::NEG_ZERO),
            (F::NEG_ZERO, F::ZERO, F::NEG_ZERO),
        ];

        for (x, y, res) in cases {
            let val = fminimum(x, y);
            assert_biteq!(val, res, "fminimum({}, {})", Hexf(x), Hexf(y));
        }
    }

    #[test]
    #[cfg(f16_enabled)]
    fn spec_tests_f16() {
        spec_test::<f16>();
    }

    #[test]
    fn spec_tests_f32() {
        spec_test::<f32>();
    }

    #[test]
    fn spec_tests_f64() {
        spec_test::<f64>();
    }

    #[test]
    #[cfg(f128_enabled)]
    fn spec_tests_f128() {
        spec_test::<f128>();
    }
}
