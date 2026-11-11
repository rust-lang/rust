/// Return the lesser of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `minimum`. The result orders -0.0 < 0.0.
#[cfg(f16_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn fminimumf16(x: f16, y: f16) -> f16 {
    super::generic::fminimum(x, y)
}

/// Return the lesser of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `minimum`. The result orders -0.0 < 0.0.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn fminimum(x: f64, y: f64) -> f64 {
    super::generic::fminimum(x, y)
}

/// Return the lesser of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `minimum`. The result orders -0.0 < 0.0.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn fminimumf(x: f32, y: f32) -> f32 {
    super::generic::fminimum(x, y)
}

/// Return the lesser of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `minimum`. The result orders -0.0 < 0.0.
#[cfg(f128_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn fminimumf128(x: f128, y: f128) -> f128 {
    super::generic::fminimum(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `maximum`. The result orders -0.0 < 0.0.
#[cfg(f16_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn fmaximumf16(x: f16, y: f16) -> f16 {
    super::generic::fmaximum(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `maximum`. The result orders -0.0 < 0.0.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn fmaximumf(x: f32, y: f32) -> f32 {
    super::generic::fmaximum(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `maximum`. The result orders -0.0 < 0.0.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn fmaximum(x: f64, y: f64) -> f64 {
    super::generic::fmaximum(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `maximum`. The result orders -0.0 < 0.0.
#[cfg(f128_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn fmaximumf128(x: f128, y: f128) -> f128 {
    super::generic::fmaximum(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::{Float, Hex};

    fn fminimum_spec_test<F: Float>(f: impl Fn(F, F) -> F) {
        let cases = [
            (F::ZERO, F::ZERO, F::ZERO),
            (F::ZERO, F::NEG_ZERO, F::NEG_ZERO),
            (F::ZERO, F::ONE, F::ZERO),
            (F::ZERO, F::NEG_ONE, F::NEG_ONE),
            (F::ZERO, F::INFINITY, F::ZERO),
            (F::ZERO, F::NEG_INFINITY, F::NEG_INFINITY),
            (F::ZERO, F::NAN, F::NAN),
            (F::NEG_ZERO, F::ZERO, F::NEG_ZERO),
            (F::NEG_ZERO, F::NEG_ZERO, F::NEG_ZERO),
            (F::NEG_ZERO, F::ONE, F::NEG_ZERO),
            (F::NEG_ZERO, F::NEG_ONE, F::NEG_ONE),
            (F::NEG_ZERO, F::INFINITY, F::NEG_ZERO),
            (F::NEG_ZERO, F::NEG_INFINITY, F::NEG_INFINITY),
            (F::NEG_ZERO, F::NAN, F::NAN),
            (F::ONE, F::ZERO, F::ZERO),
            (F::ONE, F::NEG_ZERO, F::NEG_ZERO),
            (F::ONE, F::ONE, F::ONE),
            (F::ONE, F::NEG_ONE, F::NEG_ONE),
            (F::ONE, F::INFINITY, F::ONE),
            (F::ONE, F::NEG_INFINITY, F::NEG_INFINITY),
            (F::ONE, F::NAN, F::NAN),
            (F::NEG_ONE, F::ZERO, F::NEG_ONE),
            (F::NEG_ONE, F::NEG_ZERO, F::NEG_ONE),
            (F::NEG_ONE, F::ONE, F::NEG_ONE),
            (F::NEG_ONE, F::NEG_ONE, F::NEG_ONE),
            (F::NEG_ONE, F::INFINITY, F::NEG_ONE),
            (F::NEG_ONE, F::NEG_INFINITY, F::NEG_INFINITY),
            (F::NEG_ONE, F::NAN, F::NAN),
            (F::INFINITY, F::ZERO, F::ZERO),
            (F::INFINITY, F::NEG_ZERO, F::NEG_ZERO),
            (F::INFINITY, F::ONE, F::ONE),
            (F::INFINITY, F::NEG_ONE, F::NEG_ONE),
            (F::INFINITY, F::INFINITY, F::INFINITY),
            (F::INFINITY, F::NEG_INFINITY, F::NEG_INFINITY),
            (F::INFINITY, F::NAN, F::NAN),
            (F::NEG_INFINITY, F::ZERO, F::NEG_INFINITY),
            (F::NEG_INFINITY, F::NEG_ZERO, F::NEG_INFINITY),
            (F::NEG_INFINITY, F::ONE, F::NEG_INFINITY),
            (F::NEG_INFINITY, F::NEG_ONE, F::NEG_INFINITY),
            (F::NEG_INFINITY, F::INFINITY, F::NEG_INFINITY),
            (F::NEG_INFINITY, F::NEG_INFINITY, F::NEG_INFINITY),
            (F::NEG_INFINITY, F::NAN, F::NAN),
            (F::NAN, F::ZERO, F::NAN),
            (F::NAN, F::NEG_ZERO, F::NAN),
            (F::NAN, F::ONE, F::NAN),
            (F::NAN, F::NEG_ONE, F::NAN),
            (F::NAN, F::INFINITY, F::NAN),
            (F::NAN, F::NEG_INFINITY, F::NAN),
            (F::NAN, F::NAN, F::NAN),
            (F::NAN, F::SNAN, F::NAN),
        ];

        for (x, y, res) in cases {
            let val = f(x, y);
            assert_biteq!(
                val,
                res,
                "fminimum({}, {}) ({}, {})",
                Hex(x),
                Hex(y),
                Hex(x.to_bits()),
                Hex(y.to_bits()),
            );
        }

        // On platforms where operations only return a single canonical NaN (e.g. RISC-V), the
        // result may not exactly match one of the inputs which is fine.
        assert!(f(F::NAN, F::NEG_NAN).is_qnan());
        assert!(f(F::NEG_NAN, F::NAN).is_qnan());
        assert!(f(F::ZERO, F::NEG_NAN).is_qnan());
        assert!(f(F::NEG_ZERO, F::NEG_NAN).is_qnan());
        assert!(f(F::ONE, F::NEG_NAN).is_qnan());
        assert!(f(F::NEG_ONE, F::NEG_NAN).is_qnan());
        assert!(f(F::INFINITY, F::NEG_NAN).is_qnan());
        assert!(f(F::NEG_INFINITY, F::NEG_NAN).is_qnan());
        assert!(f(F::NEG_NAN, F::ZERO).is_qnan());
        assert!(f(F::NEG_NAN, F::NEG_ZERO).is_qnan());
        assert!(f(F::NEG_NAN, F::ONE).is_qnan());
        assert!(f(F::NEG_NAN, F::NEG_ONE).is_qnan());
        assert!(f(F::NEG_NAN, F::INFINITY).is_qnan());
        assert!(f(F::NEG_NAN, F::NEG_INFINITY).is_qnan());
        assert!(f(F::NEG_NAN, F::NEG_NAN).is_qnan());

        // These operations should technically return a qnan, but LLVM optimizes out our
        // `* 1.0` canonicalization.
        assert!(f(F::INFINITY, F::SNAN,).is_nan());
        assert!(f(F::NEG_INFINITY, F::SNAN,).is_nan());
        assert!(f(F::NEG_ONE, F::SNAN,).is_nan());
        assert!(f(F::NEG_SNAN, F::INFINITY).is_nan());
        assert!(f(F::NEG_SNAN, F::NEG_INFINITY).is_nan());
        assert!(f(F::NEG_SNAN, F::NEG_NAN).is_nan());
        assert!(f(F::NEG_SNAN, F::NEG_ONE).is_nan());
        assert!(f(F::NEG_SNAN, F::NEG_ZERO).is_nan());
        assert!(f(F::NEG_SNAN, F::ONE).is_nan());
        assert!(f(F::NEG_SNAN, F::ZERO).is_nan());
        assert!(f(F::NEG_ZERO, F::SNAN,).is_nan());
        assert!(f(F::ONE, F::SNAN,).is_nan());
        assert!(f(F::SNAN, F::INFINITY,).is_nan());
        assert!(f(F::SNAN, F::NEG_INFINITY,).is_nan());
        assert!(f(F::SNAN, F::NEG_ONE,).is_nan());
        assert!(f(F::SNAN, F::NEG_SNAN,).is_nan());
        assert!(f(F::SNAN, F::NEG_ZERO,).is_nan());
        assert!(f(F::SNAN, F::ONE,).is_nan());
        assert!(f(F::SNAN, F::SNAN,).is_nan());
        assert!(f(F::SNAN, F::ZERO,).is_nan());
        assert!(f(F::ZERO, F::SNAN,).is_nan());
    }

    #[test]
    #[cfg(f16_enabled)]
    fn fminimum_spec_tests_f16() {
        fminimum_spec_test::<f16>(fminimumf16);
    }

    #[test]
    fn fminimum_spec_tests_f32() {
        fminimum_spec_test::<f32>(fminimumf);
    }

    #[test]
    fn fminimum_spec_tests_f64() {
        fminimum_spec_test::<f64>(fminimum);
    }

    #[test]
    #[cfg(f128_enabled)]
    fn fminimum_spec_tests_f128() {
        fminimum_spec_test::<f128>(fminimumf128);
    }

    fn fmaximum_spec_test<F: Float>(f: impl Fn(F, F) -> F) {
        let cases = [
            (F::ZERO, F::ZERO, F::ZERO),
            (F::ZERO, F::NEG_ZERO, F::ZERO),
            (F::ZERO, F::ONE, F::ONE),
            (F::ZERO, F::NEG_ONE, F::ZERO),
            (F::ZERO, F::INFINITY, F::INFINITY),
            (F::ZERO, F::NEG_INFINITY, F::ZERO),
            (F::ZERO, F::NAN, F::NAN),
            (F::NEG_ZERO, F::ZERO, F::ZERO),
            (F::NEG_ZERO, F::NEG_ZERO, F::NEG_ZERO),
            (F::NEG_ZERO, F::ONE, F::ONE),
            (F::NEG_ZERO, F::NEG_ONE, F::NEG_ZERO),
            (F::NEG_ZERO, F::INFINITY, F::INFINITY),
            (F::NEG_ZERO, F::NEG_INFINITY, F::NEG_ZERO),
            (F::NEG_ZERO, F::NAN, F::NAN),
            (F::ONE, F::ZERO, F::ONE),
            (F::ONE, F::NEG_ZERO, F::ONE),
            (F::ONE, F::ONE, F::ONE),
            (F::ONE, F::NEG_ONE, F::ONE),
            (F::ONE, F::INFINITY, F::INFINITY),
            (F::ONE, F::NEG_INFINITY, F::ONE),
            (F::ONE, F::NAN, F::NAN),
            (F::NEG_ONE, F::ZERO, F::ZERO),
            (F::NEG_ONE, F::NEG_ZERO, F::NEG_ZERO),
            (F::NEG_ONE, F::ONE, F::ONE),
            (F::NEG_ONE, F::NEG_ONE, F::NEG_ONE),
            (F::NEG_ONE, F::INFINITY, F::INFINITY),
            (F::NEG_ONE, F::NEG_INFINITY, F::NEG_ONE),
            (F::NEG_ONE, F::NAN, F::NAN),
            (F::INFINITY, F::ZERO, F::INFINITY),
            (F::INFINITY, F::NEG_ZERO, F::INFINITY),
            (F::INFINITY, F::ONE, F::INFINITY),
            (F::INFINITY, F::NEG_ONE, F::INFINITY),
            (F::INFINITY, F::INFINITY, F::INFINITY),
            (F::INFINITY, F::NEG_INFINITY, F::INFINITY),
            (F::INFINITY, F::NAN, F::NAN),
            (F::NEG_INFINITY, F::ZERO, F::ZERO),
            (F::NEG_INFINITY, F::NEG_ZERO, F::NEG_ZERO),
            (F::NEG_INFINITY, F::ONE, F::ONE),
            (F::NEG_INFINITY, F::NEG_ONE, F::NEG_ONE),
            (F::NEG_INFINITY, F::INFINITY, F::INFINITY),
            (F::NEG_INFINITY, F::NEG_INFINITY, F::NEG_INFINITY),
            (F::NEG_INFINITY, F::NAN, F::NAN),
            (F::NAN, F::ZERO, F::NAN),
            (F::NAN, F::NEG_ZERO, F::NAN),
            (F::NAN, F::ONE, F::NAN),
            (F::NAN, F::NEG_ONE, F::NAN),
            (F::NAN, F::INFINITY, F::NAN),
            (F::NAN, F::NEG_INFINITY, F::NAN),
            (F::NAN, F::NAN, F::NAN),
            (F::NAN, F::SNAN, F::NAN),
        ];

        for (x, y, res) in cases {
            let val = f(x, y);
            assert_biteq!(
                val,
                res,
                "fmaximum({}, {}) ({}, {})",
                Hex(x),
                Hex(y),
                Hex(x.to_bits()),
                Hex(y.to_bits()),
            );
        }

        // On platforms where operations only return a single canonical NaN (e.g. RISC-V), the
        // result may not exactly match one of the inputs which is fine.
        assert!(f(F::NAN, F::NEG_NAN).is_qnan());
        assert!(f(F::NEG_NAN, F::NAN).is_qnan());
        assert!(f(F::ZERO, F::NEG_NAN).is_qnan());
        assert!(f(F::NEG_ZERO, F::NEG_NAN).is_qnan());
        assert!(f(F::ONE, F::NEG_NAN).is_qnan());
        assert!(f(F::NEG_ONE, F::NEG_NAN).is_qnan());
        assert!(f(F::INFINITY, F::NEG_NAN).is_qnan());
        assert!(f(F::NEG_INFINITY, F::NEG_NAN).is_qnan());
        assert!(f(F::NEG_NAN, F::ZERO).is_qnan());
        assert!(f(F::NEG_NAN, F::NEG_ZERO).is_qnan());
        assert!(f(F::NEG_NAN, F::ONE).is_qnan());
        assert!(f(F::NEG_NAN, F::NEG_ONE).is_qnan());
        assert!(f(F::NEG_NAN, F::INFINITY).is_qnan());
        assert!(f(F::NEG_NAN, F::NEG_INFINITY).is_qnan());
        assert!(f(F::NEG_NAN, F::NEG_NAN).is_qnan());

        // These operations should technically return a qnan, but LLVM optimizes out our
        // `* 1.0` canonicalization.
        assert!(f(F::INFINITY, F::SNAN,).is_nan());
        assert!(f(F::NEG_INFINITY, F::SNAN,).is_nan());
        assert!(f(F::NEG_ONE, F::SNAN,).is_nan());
        assert!(f(F::NEG_SNAN, F::INFINITY).is_nan());
        assert!(f(F::NEG_SNAN, F::NEG_INFINITY).is_nan());
        assert!(f(F::NEG_SNAN, F::NEG_NAN).is_nan());
        assert!(f(F::NEG_SNAN, F::NEG_ONE).is_nan());
        assert!(f(F::NEG_SNAN, F::NEG_ZERO).is_nan());
        assert!(f(F::NEG_SNAN, F::ONE).is_nan());
        assert!(f(F::NEG_SNAN, F::ZERO).is_nan());
        assert!(f(F::NEG_ZERO, F::SNAN,).is_nan());
        assert!(f(F::ONE, F::SNAN,).is_nan());
        assert!(f(F::SNAN, F::INFINITY,).is_nan());
        assert!(f(F::SNAN, F::NEG_INFINITY,).is_nan());
        assert!(f(F::SNAN, F::NEG_ONE,).is_nan());
        assert!(f(F::SNAN, F::NEG_SNAN,).is_nan());
        assert!(f(F::SNAN, F::NEG_ZERO,).is_nan());
        assert!(f(F::SNAN, F::ONE,).is_nan());
        assert!(f(F::SNAN, F::SNAN,).is_nan());
        assert!(f(F::SNAN, F::ZERO,).is_nan());
        assert!(f(F::ZERO, F::SNAN,).is_nan());
    }

    #[test]
    #[cfg(f16_enabled)]
    fn fmaximum_spec_tests_f16() {
        fmaximum_spec_test::<f16>(fmaximumf16);
    }

    #[test]
    fn fmaximum_spec_tests_f32() {
        fmaximum_spec_test::<f32>(fmaximumf);
    }

    #[test]
    fn fmaximum_spec_tests_f64() {
        fmaximum_spec_test::<f64>(fmaximum);
    }

    #[test]
    #[cfg(f128_enabled)]
    fn fmaximum_spec_tests_f128() {
        fmaximum_spec_test::<f128>(fmaximumf128);
    }
}
