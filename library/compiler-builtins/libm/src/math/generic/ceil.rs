/* SPDX-License-Identifier: MIT */
/* origin: musl src/math/ceilf.c */

//! Generic `ceil` algorithm.
//!
//! Note that this uses the algorithm from musl's `ceilf` rather than `ceil` or `ceill` because
//! performance seems to be better (based on icount) and it does not seem to experience rounding
//! errors on i386.

use crate::support::{Float, FpResult, Int, IntTy, MinInt, Status};

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
