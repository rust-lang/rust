/* SPDX-License-Identifier: MIT */
/* origin: musl src/math/fmod.c. Ported to generic Rust algorithm in 2025, TG. */

use super::super::{CastFrom, Float, Int, MinInt};

#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmod<F: Float>(x: F, y: F) -> F {
    let zero = F::Int::ZERO;
    let one = F::Int::ONE;
    let mut ix = x.to_bits();
    let mut iy = y.to_bits();
    let mut ex = x.exp().signed();
    let mut ey = y.exp().signed();
    let sx = ix & F::SIGN_MASK;

    if iy << 1 == zero || y.is_nan() || ex == F::EXP_SAT as i32 {
        return (x * y) / (x * y);
    }

    if ix << 1 <= iy << 1 {
        if ix << 1 == iy << 1 {
            return F::ZERO * x;
        }
        return x;
    }

    /* normalize x and y */
    if ex == 0 {
        let i = ix << F::EXP_BITS;
        ex -= i.leading_zeros() as i32;
        ix <<= -ex + 1;
    } else {
        ix &= F::Int::MAX >> F::EXP_BITS;
        ix |= one << F::SIG_BITS;
    }

    if ey == 0 {
        let i = iy << F::EXP_BITS;
        ey -= i.leading_zeros() as i32;
        iy <<= -ey + 1;
    } else {
        iy &= F::Int::MAX >> F::EXP_BITS;
        iy |= one << F::SIG_BITS;
    }

    /* x mod y */
    while ex > ey {
        let i = ix.wrapping_sub(iy);
        if i >> (F::BITS - 1) == zero {
            if i == zero {
                return F::ZERO * x;
            }
            ix = i;
        }

        ix <<= 1;
        ex -= 1;
    }

    let i = ix.wrapping_sub(iy);
    if i >> (F::BITS - 1) == zero {
        if i == zero {
            return F::ZERO * x;
        }

        ix = i;
    }

    let shift = ix.leading_zeros().saturating_sub(F::EXP_BITS);
    ix <<= shift;
    ex -= shift as i32;

    /* scale result */
    if ex > 0 {
        ix -= one << F::SIG_BITS;
        ix |= F::Int::cast_from(ex) << F::SIG_BITS;
    } else {
        ix >>= -ex + 1;
    }

    ix |= sx;

    F::from_bits(ix)
}
