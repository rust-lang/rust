use super::generic;
use crate::support::Round;

/// The square root of `x` (f16).
#[cfg(f16_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn sqrtf16(x: f16) -> f16 {
    select_implementation! {
        name: sqrtf16,
        use_arch: all(target_arch = "aarch64", target_feature = "fp16"),
        args: x,
    }

    return generic::sqrt_round(x, Round::Nearest).val;
}

/// The square root of `x` (f32).
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn sqrtf(x: f32) -> f32 {
    select_implementation! {
        name: sqrtf,
        use_arch: any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "wasm32", intrinsics_enabled),
            target_feature = "sse2"
        ),
        args: x,
    }

    generic::sqrt_round(x, Round::Nearest).val
}

/// The square root of `x` (f64).
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn sqrt(x: f64) -> f64 {
    select_implementation! {
        name: sqrt,
        use_arch: any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "wasm32", intrinsics_enabled),
            target_feature = "sse2"
        ),
        args: x,
    }

    generic::sqrt_round(x, Round::Nearest).val
}

/// The square root of `x` (f128).
#[cfg(f128_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn sqrtf128(x: f128) -> f128 {
    return generic::sqrt_round(x, Round::Nearest).val;
}

#[cfg(test)]
mod tests {
    use generic::SqrtHelper;

    use super::*;
    use crate::support::{CastInto, Float, FpResult, HInt, Status};

    /// Test behavior specified in IEEE 754 `squareRoot`.
    fn spec_test<F>()
    where
        F: Float + SqrtHelper,
        F::Int: HInt,
        F::Int: From<u8>,
        F::Int: From<F::ISet2>,
        F::Int: CastInto<F::ISet1>,
        F::Int: CastInto<F::ISet2>,
        u32: CastInto<F::Int>,
    {
        // Values that should return a NaN and raise invalid
        let nan = [F::NEG_INFINITY, F::NEG_ONE, F::NAN, F::MIN];

        // Values that return unaltered
        let roundtrip = [F::ZERO, F::NEG_ZERO, F::INFINITY];

        for x in nan {
            let FpResult { val, status } = generic::sqrt_round(x, Round::Nearest);
            assert!(val.is_nan());
            assert!(status == Status::INVALID);
        }

        for x in roundtrip {
            let FpResult { val, status } = generic::sqrt_round(x, Round::Nearest);
            assert_biteq!(val, x);
            assert!(status == Status::OK);
        }
    }

    #[test]
    #[cfg(f16_enabled)]
    fn sanity_check_f16() {
        assert_biteq!(sqrtf16(100.0f16), 10.0);
        assert_biteq!(sqrtf16(4.0f16), 2.0);
    }

    #[test]
    #[cfg(f16_enabled)]
    fn spec_tests_f16() {
        spec_test::<f16>();
    }

    #[test]
    #[cfg(f16_enabled)]
    #[allow(clippy::approx_constant)]
    fn conformance_tests_f16() {
        let cases = [
            (f16::PI, 0x3f17_u16),
            (10000.0_f16, 0x5640_u16),
            (f16::from_bits(0x0000000f), 0x13bf_u16),
            (f16::INFINITY, f16::INFINITY.to_bits()),
        ];

        for (input, output) in cases {
            assert_biteq!(
                sqrtf16(input),
                f16::from_bits(output),
                "input: {input:?} ({:#018x})",
                input.to_bits()
            );
        }
    }

    #[test]
    fn sanity_check_f32() {
        assert_biteq!(sqrtf(100.0f32), 10.0);
        assert_biteq!(sqrtf(4.0f32), 2.0);
    }

    #[test]
    fn spec_tests_f32() {
        spec_test::<f32>();
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn conformance_tests_f32() {
        let cases = [
            (f32::PI, 0x3fe2dfc5_u32),
            (10000.0f32, 0x42c80000_u32),
            (f32::from_bits(0x0000000f), 0x1b2f456f_u32),
            (f32::INFINITY, f32::INFINITY.to_bits()),
        ];

        for (input, output) in cases {
            assert_biteq!(
                sqrtf(input),
                f32::from_bits(output),
                "input: {input:?} ({:#018x})",
                input.to_bits()
            );
        }
    }

    #[test]
    fn sanity_check_f64() {
        assert_biteq!(sqrt(100.0f64), 10.0);
        assert_biteq!(sqrt(4.0f64), 2.0);
    }

    #[test]
    fn spec_tests_f64() {
        spec_test::<f64>();
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn conformance_tests_f64() {
        let cases = [
            (f64::PI, 0x3ffc5bf891b4ef6a_u64),
            (10000.0, 0x4059000000000000_u64),
            (f64::from_bits(0x0000000f), 0x1e7efbdeb14f4eda_u64),
            (f64::INFINITY, f64::INFINITY.to_bits()),
        ];

        for (input, output) in cases {
            assert_biteq!(
                sqrt(input),
                f64::from_bits(output),
                "input: {input:?} ({:#018x})",
                input.to_bits()
            );
        }
    }

    #[test]
    #[cfg(f128_enabled)]
    fn sanity_check_f128() {
        assert_biteq!(sqrtf128(100.0f128), 10.0);
        assert_biteq!(sqrtf128(4.0f128), 2.0);
    }

    #[test]
    #[cfg(f128_enabled)]
    fn spec_tests_f128() {
        spec_test::<f128>();
    }

    #[test]
    #[cfg(f128_enabled)]
    #[allow(clippy::approx_constant)]
    fn conformance_tests_f128() {
        let cases = [
            (f128::PI, 0x3fffc5bf891b4ef6aa79c3b0520d5db9_u128),
            // 10_000.0, see `f16` for reasoning.
            (
                f128::from_bits(0x400c3880000000000000000000000000),
                0x40059000000000000000000000000000_u128,
            ),
            (
                f128::from_bits(0x0000000f),
                0x1fc9efbdeb14f4ed9b17ae807907e1e9_u128,
            ),
            (f128::INFINITY, f128::INFINITY.to_bits()),
        ];

        for (input, output) in cases {
            assert_biteq!(
                sqrtf128(input),
                f128::from_bits(output),
                "input: {input:?} ({:#018x})",
                input.to_bits()
            );
        }
    }
}
