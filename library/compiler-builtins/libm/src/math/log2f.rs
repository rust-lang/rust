/* origin: FreeBSD /usr/src/lib/msun/src/e_log2f.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/*
 * See comments in log2.c.
 */

use core::f32;

const IVLN2HI: f32 = 1.4428710938e+00; /* 0x3fb8b000 */
const IVLN2LO: f32 = -1.7605285393e-04; /* 0xb9389ad4 */
/* |(log(1+s)-log(1-s))/s - Lg(s)| < 2**-34.24 (~[-4.95e-11, 4.97e-11]). */
const LG1: f32 = 0.66666662693; /* 0xaaaaaa.0p-24 */
const LG2: f32 = 0.40000972152; /* 0xccce13.0p-25 */
const LG3: f32 = 0.28498786688; /* 0x91e9ee.0p-25 */
const LG4: f32 = 0.24279078841; /* 0xf89e26.0p-26 */

/// The base 2 logarithm of `x` (f32).
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn log2f(mut x: f32) -> f32 {
    let x1p25f = f32::from_bits(0x4c000000); // 0x1p25f === 2 ^ 25

    let mut ui: u32 = x.to_bits();
    let hfsq: f32;
    let f: f32;
    let s: f32;
    let z: f32;
    let r: f32;
    let w: f32;
    let t1: f32;
    let t2: f32;
    let mut hi: f32;
    let lo: f32;
    let mut ix: u32;
    let mut k: i32;

    ix = ui;
    k = 0;
    if ix < 0x00800000 || (ix >> 31) > 0 {
        /* x < 2**-126  */
        if ix << 1 == 0 {
            return -1. / (x * x); /* log(+-0)=-inf */
        }
        if (ix >> 31) > 0 {
            return (x - x) / 0.0; /* log(-#) = NaN */
        }
        /* subnormal number, scale up x */
        k -= 25;
        x *= x1p25f;
        ui = x.to_bits();
        ix = ui;
    } else if ix >= 0x7f800000 {
        return x;
    } else if ix == 0x3f800000 {
        return 0.;
    }

    /* reduce x into [sqrt(2)/2, sqrt(2)] */
    ix += 0x3f800000 - 0x3f3504f3;
    k += (ix >> 23) as i32 - 0x7f;
    ix = (ix & 0x007fffff) + 0x3f3504f3;
    ui = ix;
    x = f32::from_bits(ui);

    f = x - 1.0;
    s = f / (2.0 + f);
    z = s * s;
    w = z * z;
    t1 = w * (LG2 + w * LG4);
    t2 = z * (LG1 + w * LG3);
    r = t2 + t1;
    hfsq = 0.5 * f * f;

    hi = f - hfsq;
    ui = hi.to_bits();
    ui &= 0xfffff000;
    hi = f32::from_bits(ui);
    lo = f - hi - hfsq + s * (hfsq + r);
    (lo + hi) * IVLN2LO + lo * IVLN2HI + hi * IVLN2HI + k as f32
}
