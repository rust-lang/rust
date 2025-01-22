/* SPDX-License-Identifier: MIT */
/* origin: musl src/math/rint.c */

use super::super::Float;

pub fn rint<F: Float>(x: F) -> F {
    let toint = F::ONE / F::EPSILON;
    let e = x.exp();
    let positive = x.is_sign_positive();

    // On i386 `force_eval!` must be used to force rounding via storage to memory. Otherwise,
    // the excess precission from x87 would cause an incorrect final result.
    let use_force = cfg!(x86_no_sse) && F::BITS == 32 || F::BITS == 64;

    if e >= F::EXP_BIAS + F::SIG_BITS {
        // No fractional part; exact result can be returned.
        x
    } else {
        // Apply a net-zero adjustment that nudges `y` in the direction of the rounding mode.
        let y = if positive {
            let tmp = if use_force { force_eval!(x) } else { x } + toint;
            (if use_force { force_eval!(tmp) } else { tmp } - toint)
        } else {
            let tmp = if use_force { force_eval!(x) } else { x } - toint;
            (if use_force { force_eval!(tmp) } else { tmp } + toint)
        };

        if y == F::ZERO {
            // A zero result takes the sign of the input.
            if positive { F::ZERO } else { F::NEG_ZERO }
        } else {
            y
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroes_f32() {
        assert_biteq!(rint(0.0_f32), 0.0_f32);
        assert_biteq!(rint(-0.0_f32), -0.0_f32);
    }

    #[test]
    fn sanity_check_f32() {
        assert_biteq!(rint(-1.0_f32), -1.0);
        assert_biteq!(rint(2.8_f32), 3.0);
        assert_biteq!(rint(-0.5_f32), -0.0);
        assert_biteq!(rint(0.5_f32), 0.0);
        assert_biteq!(rint(-1.5_f32), -2.0);
        assert_biteq!(rint(1.5_f32), 2.0);
    }

    #[test]
    fn zeroes_f64() {
        assert_biteq!(rint(0.0_f64), 0.0_f64);
        assert_biteq!(rint(-0.0_f64), -0.0_f64);
    }

    #[test]
    fn sanity_check_f64() {
        assert_biteq!(rint(-1.0_f64), -1.0);
        assert_biteq!(rint(2.8_f64), 3.0);
        assert_biteq!(rint(-0.5_f64), -0.0);
        assert_biteq!(rint(0.5_f64), 0.0);
        assert_biteq!(rint(-1.5_f64), -2.0);
        assert_biteq!(rint(1.5_f64), 2.0);
    }
}
