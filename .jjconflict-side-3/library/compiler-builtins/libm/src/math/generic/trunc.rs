/* SPDX-License-Identifier: MIT
 * origin: musl src/math/trunc.c */

use crate::support::{Float, FpResult, Int, IntTy, MinInt, Status};

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
