/* SPDX-License-Identifier: MIT */
/* origin: musl src/math/rint.c */

use crate::support::{Float, FpResult, Status};

/// IEEE 754-2019 `roundToIntegralExact`, which respects rounding mode and raises inexact if
/// applicable.
#[inline]
pub fn rint_status<F: Float>(x: F) -> FpResult<F> {
    let toint = F::ONE / F::EPSILON;
    let e = x.ex();
    let positive = x.is_sign_positive();

    // On i386 `force_eval!` must be used to force rounding via storage to memory. Otherwise,
    // the excess precission from x87 would cause an incorrect final result.
    let force = |x| {
        if cfg!(x86_no_sse) && (F::BITS == 32 || F::BITS == 64) {
            force_eval!(x)
        } else {
            x
        }
    };

    let res = if e >= F::EXP_BIAS + F::SIG_BITS {
        // No fractional part; exact result can be returned.
        x
    } else {
        // Apply a net-zero adjustment that nudges `y` in the direction of the rounding mode. For
        // Rust this is always nearest, but ideally it would take `round` into account.
        let y = if positive {
            force(force(x) + toint) - toint
        } else {
            force(force(x) - toint) + toint
        };

        if y == F::ZERO {
            // A zero result takes the sign of the input.
            if positive { F::ZERO } else { F::NEG_ZERO }
        } else {
            y
        }
    };

    let status = if res == x {
        Status::OK
    } else {
        Status::INEXACT
    };
    FpResult::new(res, status)
}
