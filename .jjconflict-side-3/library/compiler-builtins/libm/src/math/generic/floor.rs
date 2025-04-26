/* SPDX-License-Identifier: MIT
 * origin: musl src/math/floor.c */

//! Generic `floor` algorithm.
//!
//! Note that this uses the algorithm from musl's `floorf` rather than `floor` or `floorl` because
//! performance seems to be better (based on icount) and it does not seem to experience rounding
//! errors on i386.

use crate::support::{Float, FpResult, Int, IntTy, MinInt, Status};

#[inline]
pub fn floor_status<F: Float>(x: F) -> FpResult<F> {
    let zero = IntTy::<F>::ZERO;

    let mut ix = x.to_bits();
    let e = x.exp_unbiased();

    // If the represented value has no fractional part, no truncation is needed.
    if e >= F::SIG_BITS as i32 {
        return FpResult::ok(x);
    }

    let res = if e >= 0 {
        // |x| >= 1.0
        let m = F::SIG_MASK >> e.unsigned();
        if ix & m == zero {
            // Portion to be masked is already zero; no adjustment needed.
            return FpResult::ok(x);
        }

        if x.is_sign_negative() {
            ix += m;
        }

        ix &= !m;
        F::from_bits(ix)
    } else {
        // |x| < 1.0, zero or inexact with truncation

        if (ix & !F::SIGN_MASK) == F::Int::ZERO {
            return FpResult::ok(x);
        }

        if x.is_sign_positive() {
            // 0.0 <= x < 1.0; rounding down goes toward +0.0.
            F::ZERO
        } else {
            // -1.0 < x < 0.0; rounding down goes toward -1.0.
            F::NEG_ONE
        }
    };

    FpResult::new(res, Status::INEXACT)
}
