/* SPDX-License-Identifier: MIT */
/* origin: musl src/math/ceilf.c */

//! Generic `ceil` algorithm.
//!
//! Note that this uses the algorithm from musl's `ceilf` rather than `ceil` or `ceill` because
//! performance seems to be better (based on icount) and it does not seem to experience rounding
//! errors on i386.

use crate::support::{Float, FpResult, Int, IntTy, MinInt, Status};

#[inline]
pub fn ceil<F: Float>(x: F) -> F {
    ceil_status(x).val
}

#[inline]
pub fn ceil_status<F: Float>(x: F) -> FpResult<F> {
    let zero = IntTy::<F>::ZERO;

    let mut ix = x.to_bits();
    let e = x.exp_unbiased();

    // If the represented value has no fractional part, no truncation is needed.
    if e >= F::SIG_BITS as i32 {
        return FpResult::ok(x);
    }

    let status;
    let res = if e >= 0 {
        // |x| >= 1.0
        let m = F::SIG_MASK >> e.unsigned();
        if (ix & m) == zero {
            // Portion to be masked is already zero; no adjustment needed.
            return FpResult::ok(x);
        }

        // Otherwise, raise an inexact exception.
        status = Status::INEXACT;

        if x.is_sign_positive() {
            ix += m;
        }

        ix &= !m;
        F::from_bits(ix)
    } else {
        // |x| < 1.0, raise an inexact exception since truncation will happen (unless x == 0).
        if ix & !F::SIGN_MASK == F::Int::ZERO {
            status = Status::OK;
        } else {
            status = Status::INEXACT;
        }

        if x.is_sign_negative() {
            // -1.0 < x <= -0.0; rounding up goes toward -0.0.
            F::NEG_ZERO
        } else if ix << 1 != zero {
            // 0.0 < x < 1.0; rounding up goes toward +1.0.
            F::ONE
        } else {
            // +0.0 remains unchanged
            x
        }
    };

    FpResult::new(res, status)
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
            let FpResult { val, status } = ceil_status(x);
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
