/* SPDX-License-Identifier: MIT
 * origin: musl src/math/trunc.c */

use crate::support::{Float, FpResult, Int, IntTy, MinInt, Status};

#[inline]
pub fn trunc<F: Float>(x: F) -> F {
    trunc_status(x).val
}

#[inline]
pub fn trunc_status<F: Float>(x: F) -> FpResult<F> {
    let xi: F::Int = x.to_bits();
    let e: i32 = x.exp_unbiased();

    // The represented value has no fractional part, so no truncation is needed
    if e >= F::SIG_BITS as i32 {
        return FpResult::ok(x);
    }

    let clear_mask = if e < 0 {
        // If the exponent is negative, the result will be zero so we clear everything
        // except the sign.
        !F::SIGN_MASK
    } else {
        // Otherwise, we keep `e` fractional bits and clear the rest.
        F::SIG_MASK >> e.unsigned()
    };

    let cleared = xi & clear_mask;
    let status = if cleared == IntTy::<F>::ZERO {
        // If the to-be-zeroed portion is already zero, we have an exact result.
        Status::OK
    } else {
        // Otherwise the result is inexact and we will truncate, so indicate `FE_INEXACT`.
        Status::INEXACT
    };

    // Now zero the bits we need to truncate and return.
    FpResult::new(F::from_bits(xi ^ cleared), status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::Hexf;

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
    fn check<F: Float>(cases: &[(F, F, Status)]) {
        for &(x, exp_res, exp_stat) in cases {
            let FpResult { val, status } = trunc_status(x);
            assert_biteq!(val, exp_res, "{x:?} {}", Hexf(x));
            assert_eq!(
                status,
                exp_stat,
                "{x:?} {} -> {exp_res:?} {}",
                Hexf(x),
                Hexf(exp_res)
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
