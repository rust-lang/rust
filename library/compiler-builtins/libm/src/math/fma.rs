/* SPDX-License-Identifier: MIT */
/* origin: musl src/math/fma.c, fmaf.c Ported to generic Rust algorithm in 2025, TG. */

use super::generic;
use crate::support::Round;

/* See `fmaf16.rs` for that implementation */

/// Floating multiply add (f32)
///
/// Computes `(x*y)+z`, rounded as one ternary operation (i.e. calculated with infinite precision).
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn fmaf(x: f32, y: f32, z: f32) -> f32 {
    select_implementation! {
        name: fmaf,
        use_arch: any(
            all(target_arch = "aarch64", target_feature = "neon"),
            target_feature = "sse2",
        ),
        args: x, y, z,
    }

    generic::fma_wide_round(x, y, z, Round::Nearest).val
}

/// Fused multiply add (f64)
///
/// Computes `(x*y)+z`, rounded as one ternary operation (i.e. calculated with infinite precision).
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn fma(x: f64, y: f64, z: f64) -> f64 {
    select_implementation! {
        name: fma,
        use_arch: any(
            all(target_arch = "aarch64", target_feature = "neon"),
            target_feature = "sse2",
        ),
        args: x, y, z,
    }

    generic::fma_round(x, y, z, Round::Nearest).val
}

/// Fused multiply add (f128)
///
/// Computes `(x*y)+z`, rounded as one ternary operation (i.e. calculated with infinite precision).
#[cfg(f128_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn fmaf128(x: f128, y: f128, z: f128) -> f128 {
    generic::fma_round(x, y, z, Round::Nearest).val
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::{Float, FpResult, Hex, Round, Status};

    macro_rules! cases {
        ($f:ty) => {
            [
                // Simple
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 1.0, 0.0, 1.0),
                (1.0, 0.0, 1.0, 1.0),
                // Sign checks
                (1.0, 1.0, 1.0, 2.0),
                (1.0, 1.0, -1.0, 0.0),
                (1.0, -1.0, 1.0, 0.0),
                (1.0, -1.0, -1.0, -2.0),
                (-1.0, 1.0, 1.0, 0.0),
                (-1.0, 1.0, -1.0, -2.0),
                (-1.0, -1.0, 1.0, 2.0),
                (-1.0, -1.0, -1.0, 0.0),

                // Roundtrip
                (<$f>::MAX, 1.0, 0.0, <$f>::MAX),
                (<$f>::MAX, <$f>::MAX, 1.0, <$f>::INFINITY),
                (<$f>::MAX, 1.0, -<$f>::MAX, 0.0),
                (-<$f>::MAX, 1.0, <$f>::MAX, 0.0),
                (<$f>::MIN_POSITIVE_NORMAL, 1.0, -<$f>::MIN_POSITIVE_NORMAL, 0.0),
                (-<$f>::MIN_POSITIVE_NORMAL, 1.0, <$f>::MIN_POSITIVE_NORMAL, 0.0),
                (<$f>::MIN_POSITIVE_SUBNORMAL, 1.0, -<$f>::MIN_POSITIVE_SUBNORMAL, 0.0),
                (-<$f>::MIN_POSITIVE_SUBNORMAL, 1.0, <$f>::MIN_POSITIVE_SUBNORMAL, 0.0),
                (<$f>::MAX, 1.0, -<$f>::MAX, 0.0),

                // 754-2020 says "When the exact result of (a × b) + c is non-zero yet the result of
                // fusedMultiplyAdd is zero because of rounding, the zero result takes the sign of the
                // exact result"
                (<$f>::MIN_POSITIVE_SUBNORMAL, <$f>::MIN_POSITIVE_SUBNORMAL, 0.0, 0.0),
                (<$f>::MIN_POSITIVE_SUBNORMAL, -<$f>::MIN_POSITIVE_SUBNORMAL, 0.0, -0.0),
                (-<$f>::MIN_POSITIVE_SUBNORMAL, <$f>::MIN_POSITIVE_SUBNORMAL, 0.0, -0.0),
                (-<$f>::MIN_POSITIVE_SUBNORMAL, -<$f>::MIN_POSITIVE_SUBNORMAL, 0.0, 0.0),
            ]
        };
    }

    #[track_caller]
    fn check<F: Float>(f: fn(F, F, F) -> F, cases: &[(F, F, F, F)]) {
        for &(x, y, z, exp_res) in cases {
            let val = f(x, y, z);
            assert_biteq!(
                val,
                exp_res,
                "fma({x:?}, {y:?}, {z:?}) ({} {} {})",
                Hex(x),
                Hex(y),
                Hex(z)
            );
        }
    }

    #[test]
    #[cfg(f16_enabled)]
    fn check_f16() {
        check::<f16>(super::super::fmaf16, &cases!(f16));
    }

    #[test]
    fn check_f32() {
        check::<f32>(fmaf, &cases!(f32));

        // Also do a small check that the non-widening version works for f32 (this should ideally
        // get tested some more).
        check::<f32>(
            |x, y, z| generic::fma_round(x, y, z, Round::Nearest).val,
            &cases!(f32),
        );
    }

    #[test]
    fn check_f64() {
        check::<f64>(fma, &cases!(f64));

        let expect_underflow = [
            (
                hf64!("0x1.0p-1070"),
                hf64!("0x1.0p-1070"),
                hf64!("0x1.ffffffffffffp-1023"),
                hf64!("0x0.ffffffffffff8p-1022"),
            ),
            (
                // FIXME: we raise underflow but this should only be inexact (based on C and
                // `rustc_apfloat`).
                hf64!("0x1.0p-1070"),
                hf64!("0x1.0p-1070"),
                hf64!("-0x1.0p-1022"),
                hf64!("-0x1.0p-1022"),
            ),
        ];

        for (x, y, z, res) in expect_underflow {
            let FpResult { val, status } = generic::fma_round(x, y, z, Round::Nearest);
            assert_biteq!(val, res);
            assert_eq!(status, Status::UNDERFLOW);
        }
    }

    #[test]
    #[cfg(f128_enabled)]
    fn check_f128() {
        check::<f128>(fmaf128, &cases!(f128));
    }

    #[test]
    fn issue_263() {
        let a = f32::from_bits(1266679807);
        let b = f32::from_bits(1300234242);
        let c = f32::from_bits(1115553792);
        let expected = f32::from_bits(1501560833);
        assert_eq!(fmaf(a, b, c), expected);
    }

    #[test]
    fn fma_segfault() {
        // These two inputs cause fma to segfault on release due to overflow:
        assert_eq!(
            fma(
                -0.0000000000000002220446049250313,
                -0.0000000000000002220446049250313,
                -0.0000000000000002220446049250313
            ),
            -0.00000000000000022204460492503126,
        );

        let result = fma(-0.992, -0.992, -0.992);
        //force rounding to storage format on x87 to prevent superious errors.
        #[cfg(x86_no_sse2)]
        let result = force_eval!(result);
        assert_eq!(result, -0.007936000000000007,);
    }

    #[test]
    fn fma_sbb() {
        assert_eq!(
            fma(-(1.0 - f64::EPSILON), f64::MIN, f64::MIN),
            -3991680619069439e277
        );
    }

    #[test]
    fn fma_underflow() {
        assert_eq!(
            fma(1.1102230246251565e-16, -9.812526705433188e-305, 1.0894e-320),
            0.0,
        );
    }
}
